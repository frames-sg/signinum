// SPDX-License-Identifier: Apache-2.0

use std::{
    ffi::c_void,
    os::raw::c_int,
    sync::{Arc, OnceLock},
};

use libloading::Library;

use crate::CudaError;

type NvjpegStatus = c_int;
type NvjpegHandle = *mut c_void;
type NvjpegJpegState = *mut c_void;
type CudaStream = *mut c_void;

const NVJPEG_STATUS_SUCCESS: NvjpegStatus = 0;
const NVJPEG_OUTPUT_RGBI: c_int = 5;
const NVJPEG_MAX_COMPONENT: usize = 4;
const NVJPEG_BACKEND_GPU_HYBRID: c_int = 2;

type NvjpegCreateEx =
    unsafe extern "C" fn(c_int, *mut c_void, *mut c_void, u32, *mut NvjpegHandle) -> NvjpegStatus;
type NvjpegCreateSimple = unsafe extern "C" fn(*mut NvjpegHandle) -> NvjpegStatus;
type NvjpegDestroy = unsafe extern "C" fn(NvjpegHandle) -> NvjpegStatus;
type NvjpegJpegStateCreate =
    unsafe extern "C" fn(NvjpegHandle, *mut NvjpegJpegState) -> NvjpegStatus;
type NvjpegJpegStateDestroy = unsafe extern "C" fn(NvjpegJpegState) -> NvjpegStatus;
type NvjpegGetImageInfo = unsafe extern "C" fn(
    NvjpegHandle,
    *const u8,
    usize,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> NvjpegStatus;
type NvjpegDecode = unsafe extern "C" fn(
    NvjpegHandle,
    NvjpegJpegState,
    *const u8,
    usize,
    c_int,
    *mut NvjpegImage,
    CudaStream,
) -> NvjpegStatus;
type NvjpegDecodeBatchedInitialize =
    unsafe extern "C" fn(NvjpegHandle, NvjpegJpegState, c_int, c_int, c_int) -> NvjpegStatus;
type NvjpegDecodeBatched = unsafe extern "C" fn(
    NvjpegHandle,
    NvjpegJpegState,
    *const *const u8,
    *const usize,
    *mut NvjpegImage,
    CudaStream,
) -> NvjpegStatus;

#[repr(C)]
struct NvjpegImage {
    channel: [*mut u8; NVJPEG_MAX_COMPONENT],
    pitch: [usize; NVJPEG_MAX_COMPONENT],
}

pub(crate) struct NvjpegLibrary {
    _library: Library,
    create_ex: Option<NvjpegCreateEx>,
    create_simple: NvjpegCreateSimple,
    destroy: NvjpegDestroy,
    jpeg_state_create: NvjpegJpegStateCreate,
    jpeg_state_destroy: NvjpegJpegStateDestroy,
    get_image_info: NvjpegGetImageInfo,
    decode: NvjpegDecode,
    decode_batched_initialize: NvjpegDecodeBatchedInitialize,
    decode_batched: NvjpegDecodeBatched,
}

impl NvjpegLibrary {
    fn load() -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        const LIBRARY_CANDIDATES: &[&str] = &[
            "libnvjpeg.so.13",
            "libnvjpeg.so.12",
            "libnvjpeg.so.11",
            "libnvjpeg.so",
        ];
        #[cfg(target_os = "windows")]
        const LIBRARY_CANDIDATES: &[&str] = &[
            "nvjpeg64_130.dll",
            "nvjpeg64_120.dll",
            "nvjpeg64_110.dll",
            "nvjpeg64.dll",
        ];
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        const LIBRARY_CANDIDATES: &[&str] = &[];

        let mut last_error = None;
        for candidate in LIBRARY_CANDIDATES {
            // SAFETY: Loading nvJPEG is required for runtime-only CUDA JPEG
            // decode support. NvjpegLibrary owns the handle for all copied
            // function pointers.
            match unsafe { Library::new(candidate) } {
                Ok(library) => return Self::from_library(library),
                Err(error) => last_error = Some(error.to_string()),
            }
        }

        Err(CudaError::NvjpegUnavailable {
            message: last_error.unwrap_or_else(|| "unsupported nvJPEG host platform".to_string()),
        })
    }

    fn from_library(library: Library) -> Result<Self, CudaError> {
        Ok(Self {
            create_ex: load_optional_nvjpeg_symbol(&library, b"nvjpegCreateEx\0"),
            create_simple: load_nvjpeg_symbol(&library, b"nvjpegCreateSimple\0")?,
            destroy: load_nvjpeg_symbol(&library, b"nvjpegDestroy\0")?,
            jpeg_state_create: load_nvjpeg_symbol(&library, b"nvjpegJpegStateCreate\0")?,
            jpeg_state_destroy: load_nvjpeg_symbol(&library, b"nvjpegJpegStateDestroy\0")?,
            get_image_info: load_nvjpeg_symbol(&library, b"nvjpegGetImageInfo\0")?,
            decode: load_nvjpeg_symbol(&library, b"nvjpegDecode\0")?,
            decode_batched_initialize: load_nvjpeg_symbol(
                &library,
                b"nvjpegDecodeBatchedInitialize\0",
            )?,
            decode_batched: load_nvjpeg_symbol(&library, b"nvjpegDecodeBatched\0")?,
            _library: library,
        })
    }

    fn check(operation: &'static str, status: NvjpegStatus) -> Result<(), CudaError> {
        if status == NVJPEG_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Nvjpeg {
                operation,
                code: status,
                name: nvjpeg_status_name(status),
            })
        }
    }
}

unsafe impl Send for NvjpegLibrary {}
unsafe impl Sync for NvjpegLibrary {}

pub(crate) struct NvjpegState {
    library: Arc<NvjpegLibrary>,
    handle: NvjpegHandle,
    state: NvjpegJpegState,
}

impl NvjpegState {
    pub(crate) fn new() -> Result<Self, CudaError> {
        Self::new_with_backend(None)
    }

    pub(crate) fn new_batched() -> Result<Self, CudaError> {
        Self::new_with_backend(Some(NVJPEG_BACKEND_GPU_HYBRID))
    }

    fn new_with_backend(backend: Option<c_int>) -> Result<Self, CudaError> {
        let library = shared_library()?;
        let mut handle = std::ptr::null_mut();
        match backend {
            Some(backend) => {
                let Some(create_ex) = library.create_ex else {
                    return Err(CudaError::NvjpegUnavailable {
                        message: "nvJPEG library does not export nvjpegCreateEx".to_string(),
                    });
                };
                NvjpegLibrary::check("nvjpegCreateEx", unsafe {
                    (create_ex)(
                        backend,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        &raw mut handle,
                    )
                })?;
            }
            None => {
                NvjpegLibrary::check("nvjpegCreateSimple", unsafe {
                    (library.create_simple)(&raw mut handle)
                })?;
            }
        }
        if handle.is_null() {
            return Err(CudaError::NvjpegUnavailable {
                message: "nvJPEG handle creation returned a null handle".to_string(),
            });
        }
        let mut state = std::ptr::null_mut();
        if let Err(error) = NvjpegLibrary::check("nvjpegJpegStateCreate", unsafe {
            (library.jpeg_state_create)(handle, &raw mut state)
        }) {
            // SAFETY: handle was created above and state creation failed before
            // ownership could be moved into NvjpegState.
            let _ = unsafe { (library.destroy)(handle) };
            return Err(error);
        }
        if state.is_null() {
            // SAFETY: handle was created above and no state was returned.
            let _ = unsafe { (library.destroy)(handle) };
            return Err(CudaError::NvjpegUnavailable {
                message: "nvjpegJpegStateCreate returned a null state".to_string(),
            });
        }
        Ok(Self {
            library,
            handle,
            state,
        })
    }

    pub(crate) fn decode_rgb8(
        &mut self,
        bytes: &[u8],
        dimensions: (u32, u32),
        device_ptr: u64,
        pitch_bytes: usize,
    ) -> Result<(), CudaError> {
        self.validate_dimensions(bytes, dimensions)?;
        let mut image = rgb8_destination(device_ptr, pitch_bytes)?;

        NvjpegLibrary::check("nvjpegDecode", unsafe {
            (self.library.decode)(
                self.handle,
                self.state,
                bytes.as_ptr(),
                bytes.len(),
                NVJPEG_OUTPUT_RGBI,
                &raw mut image,
                std::ptr::null_mut(),
            )
        })
    }

    pub(crate) fn decode_rgb8_batch(
        &mut self,
        inputs: &[(&[u8], (u32, u32))],
        outputs: &[u64],
        pitches: &[usize],
    ) -> Result<(), CudaError> {
        if inputs.len() != outputs.len() || inputs.len() != pitches.len() {
            return Err(CudaError::NvjpegUnavailable {
                message: "nvJPEG batch inputs and outputs have different lengths".to_string(),
            });
        }
        let batch_size =
            c_int::try_from(inputs.len()).map_err(|_| CudaError::NvjpegUnavailable {
                message: format!("nvJPEG batch size {} does not fit c_int", inputs.len()),
            })?;
        if batch_size == 0 {
            return Ok(());
        }

        for (bytes, dimensions) in inputs {
            self.validate_dimensions(bytes, *dimensions)?;
        }

        let mut data = Vec::with_capacity(inputs.len());
        let mut lengths = Vec::with_capacity(inputs.len());
        let mut destinations = Vec::with_capacity(inputs.len());
        for ((bytes, _dimensions), (device_ptr, pitch_bytes)) in
            inputs.iter().zip(outputs.iter().zip(pitches.iter()))
        {
            data.push(bytes.as_ptr());
            lengths.push(bytes.len());
            destinations.push(rgb8_destination(*device_ptr, *pitch_bytes)?);
        }

        NvjpegLibrary::check("nvjpegDecodeBatchedInitialize", unsafe {
            (self.library.decode_batched_initialize)(
                self.handle,
                self.state,
                batch_size,
                1,
                NVJPEG_OUTPUT_RGBI,
            )
        })?;
        NvjpegLibrary::check("nvjpegDecodeBatched", unsafe {
            (self.library.decode_batched)(
                self.handle,
                self.state,
                data.as_ptr(),
                lengths.as_ptr(),
                destinations.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }

    fn validate_dimensions(&self, bytes: &[u8], dimensions: (u32, u32)) -> Result<(), CudaError> {
        let mut components = 0;
        let mut subsampling = 0;
        let mut widths = [0; NVJPEG_MAX_COMPONENT];
        let mut heights = [0; NVJPEG_MAX_COMPONENT];
        NvjpegLibrary::check("nvjpegGetImageInfo", unsafe {
            (self.library.get_image_info)(
                self.handle,
                bytes.as_ptr(),
                bytes.len(),
                &raw mut components,
                &raw mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            )
        })?;
        let actual = (
            u32::try_from(widths[0]).unwrap_or(0),
            u32::try_from(heights[0]).unwrap_or(0),
        );
        if actual != dimensions {
            return Err(CudaError::NvjpegDimensions {
                expected: dimensions,
                actual,
            });
        }
        Ok(())
    }
}

impl Drop for NvjpegState {
    fn drop(&mut self) {
        if !self.state.is_null() {
            // SAFETY: state was created by this nvJPEG handle. Drop cannot
            // report failures, so cleanup errors are ignored.
            let _ = unsafe { (self.library.jpeg_state_destroy)(self.state) };
        }
        if !self.handle.is_null() {
            // SAFETY: handle was created by nvJPEG and outlives the JPEG state
            // destroyed above.
            let _ = unsafe { (self.library.destroy)(self.handle) };
        }
    }
}

unsafe impl Send for NvjpegState {}

fn load_nvjpeg_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T, CudaError> {
    // SAFETY: Symbol names are NUL-terminated nvJPEG entry points. The symbol
    // value is copied, and NvjpegLibrary keeps the Library alive.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaError::NvjpegUnavailable {
            message: format!(
                "missing nvJPEG symbol {}: {error}",
                String::from_utf8_lossy(name)
            ),
        })
}

fn load_optional_nvjpeg_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Option<T> {
    // SAFETY: Symbol names are NUL-terminated nvJPEG entry points. The symbol
    // value is copied, and NvjpegLibrary keeps the Library alive.
    unsafe { library.get::<T>(name) }.map(|symbol| *symbol).ok()
}

fn rgb8_destination(device_ptr: u64, pitch_bytes: usize) -> Result<NvjpegImage, CudaError> {
    let mut image = NvjpegImage {
        channel: [std::ptr::null_mut(); NVJPEG_MAX_COMPONENT],
        pitch: [0; NVJPEG_MAX_COMPONENT],
    };
    image.channel[0] = usize::try_from(device_ptr).map_err(|_| CudaError::NvjpegUnavailable {
        message: "CUDA device pointer does not fit host pointer width".to_string(),
    })? as *mut u8;
    image.pitch[0] = pitch_bytes;
    Ok(image)
}

fn shared_library() -> Result<Arc<NvjpegLibrary>, CudaError> {
    static LIBRARY: OnceLock<Result<Arc<NvjpegLibrary>, String>> = OnceLock::new();
    match LIBRARY.get_or_init(|| {
        NvjpegLibrary::load()
            .map(Arc::new)
            .map_err(|error| error.to_string())
    }) {
        Ok(library) => Ok(library.clone()),
        Err(message) => Err(CudaError::NvjpegUnavailable {
            message: message.clone(),
        }),
    }
}

fn nvjpeg_status_name(status: NvjpegStatus) -> String {
    let name = match status {
        1 => "NVJPEG_STATUS_NOT_INITIALIZED",
        2 => "NVJPEG_STATUS_INVALID_PARAMETER",
        3 => "NVJPEG_STATUS_BAD_JPEG",
        4 => "NVJPEG_STATUS_JPEG_NOT_SUPPORTED",
        5 => "NVJPEG_STATUS_ALLOCATOR_FAILURE",
        6 => "NVJPEG_STATUS_EXECUTION_FAILED",
        7 => "NVJPEG_STATUS_ARCH_MISMATCH",
        8 => "NVJPEG_STATUS_INTERNAL_ERROR",
        9 => "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
        _ => return String::new(),
    };
    format!(" ({name})")
}
