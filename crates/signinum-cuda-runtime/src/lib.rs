// SPDX-License-Identifier: Apache-2.0

//! Thin CUDA Driver API runtime used by signinum CUDA adapter crates.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod kernels;
mod nvjpeg;

use std::{
    collections::HashMap,
    ffi::c_void,
    os::raw::{c_char, c_int, c_uint},
    sync::{Arc, Mutex},
};

use kernels::{
    copy_u8_launch_geometry, j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry, CudaKernel,
};
use libloading::Library;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;

type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuGetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver is unavailable: {message}")]
    Unavailable { message: String },
    #[error("CUDA driver call {operation} failed with CUresult {code}{name}")]
    Driver {
        operation: &'static str,
        code: CuResult,
        name: String,
    },
    #[error("CUDA copy output buffer too small: required {required}, have {have}")]
    OutputTooSmall { required: usize, have: usize },
    #[error("CUDA byte length is too large for kernel launch: {len}")]
    LengthTooLarge { len: usize },
    #[error("CUDA image allocation size overflow for {width}x{height}x{channels}")]
    ImageTooLarge {
        width: u32,
        height: u32,
        channels: usize,
    },
    #[error("nvJPEG is unavailable: {message}")]
    NvjpegUnavailable { message: String },
    #[error("nvJPEG call {operation} failed with nvjpegStatus_t {code}{name}")]
    Nvjpeg {
        operation: &'static str,
        code: i32,
        name: String,
    },
    #[error("nvJPEG decoded dimensions mismatch: expected {expected:?}, got {actual:?}")]
    NvjpegDimensions {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    #[error("CUDA runtime state lock is poisoned: {message}")]
    StatePoisoned { message: String },
}

struct Driver {
    _library: Library,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_get_error_name: CuGetErrorName,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    cu_ctx_synchronize: CuCtxSynchronize,
}

impl Driver {
    fn load() -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        const LIBRARY_CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so"];
        #[cfg(target_os = "windows")]
        const LIBRARY_CANDIDATES: &[&str] = &["nvcuda.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        const LIBRARY_CANDIDATES: &[&str] = &[];

        let mut last_error = None;
        for candidate in LIBRARY_CANDIDATES {
            // SAFETY: Loading the CUDA driver library is required before symbol
            // lookup. The resulting Library is owned by Driver and outlives all
            // copied function pointers.
            match unsafe { Library::new(candidate) } {
                Ok(library) => return Self::from_library(library),
                Err(error) => last_error = Some(error.to_string()),
            }
        }

        Err(CudaError::Unavailable {
            message: last_error.unwrap_or_else(|| "unsupported CUDA host platform".to_string()),
        })
    }

    fn from_library(library: Library) -> Result<Self, CudaError> {
        Ok(Self {
            cu_init: load_symbol(&library, b"cuInit\0")?,
            cu_device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
            cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
            cu_ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
            cu_ctx_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
            cu_mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
            cu_mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
            cu_memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dtoh: load_symbol(&library, b"cuMemcpyDtoH_v2\0")?,
            cu_get_error_name: load_symbol(&library, b"cuGetErrorName\0")?,
            cu_module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
            cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
            cu_ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
            _library: library,
        })
    }

    fn check(&self, operation: &'static str, result: CuResult) -> Result<(), CudaError> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Driver {
                operation,
                code: result,
                name: self.error_name(result),
            })
        }
    }

    fn error_name(&self, result: CuResult) -> String {
        let mut name = std::ptr::null();
        // SAFETY: cuGetErrorName writes a borrowed static C string pointer for
        // a CUDA result code. A failure here is non-critical for diagnostics.
        let status = unsafe { (self.cu_get_error_name)(result, &raw mut name) };
        if status == CUDA_SUCCESS && !name.is_null() {
            // SAFETY: CUDA returns a NUL-terminated static string on success.
            let cstr = unsafe { std::ffi::CStr::from_ptr(name) };
            format!(" ({})", cstr.to_string_lossy())
        } else {
            String::new()
        }
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T, CudaError> {
    // SAFETY: Symbol names are NUL-terminated CUDA Driver API entry points. The
    // symbol value is copied, and Driver keeps the Library alive.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaError::Unavailable {
            message: format!(
                "missing CUDA driver symbol {}: {error}",
                String::from_utf8_lossy(name)
            ),
        })
}

// CUDA Driver API contexts and device pointers are process resources guarded by
// the driver. They can be moved across Rust threads as opaque handles as long
// as calls set the current context before use.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

struct ContextInner {
    driver: Driver,
    context: CuContext,
    modules: Mutex<HashMap<CudaKernel, CompiledKernel>>,
    nvjpeg: Mutex<Option<nvjpeg::NvjpegState>>,
}

impl ContextInner {
    fn set_current(&self) -> Result<(), CudaError> {
        // SAFETY: context is created by cuCtxCreate_v2 and remains valid while
        // ContextInner is alive.
        self.driver.check("cuCtxSetCurrent", unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        })
    }

    fn kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        self.set_current()?;
        let mut modules = self
            .modules
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if let Some(compiled) = modules.get(&kernel) {
            return Ok(compiled.function);
        }

        let compiled = CompiledKernel::load(self, kernel)?;
        let function = compiled.function;
        modules.insert(kernel, compiled);
        Ok(function)
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let _ = self.set_current();
            let nvjpeg = match self.nvjpeg.get_mut() {
                Ok(nvjpeg) => nvjpeg,
                Err(poisoned) => poisoned.into_inner(),
            };
            drop(nvjpeg.take());
            let modules = match self.modules.get_mut() {
                Ok(modules) => modules,
                Err(poisoned) => poisoned.into_inner(),
            };
            for compiled in modules.drain().map(|(_, compiled)| compiled) {
                // SAFETY: modules were loaded into this CUDA context. Drop
                // cannot surface errors, so cleanup failures are ignored.
                let _ = unsafe { (self.driver.cu_module_unload)(compiled.module) };
            }
            // SAFETY: context was created by this ContextInner and cached
            // modules have already been unloaded.
            let _ = unsafe { (self.driver.cu_ctx_destroy)(self.context) };
        }
    }
}

unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

#[derive(Clone)]
pub struct CudaContext {
    inner: Arc<ContextInner>,
}

impl CudaContext {
    pub fn system_default() -> Result<Self, CudaError> {
        let driver = Driver::load()?;

        // SAFETY: cuInit is the CUDA Driver API process initializer.
        driver.check("cuInit", unsafe { (driver.cu_init)(0) })?;

        let mut count = 0;
        // SAFETY: CUDA writes one integer device count to the provided pointer.
        driver.check("cuDeviceGetCount", unsafe {
            (driver.cu_device_get_count)(&raw mut count)
        })?;
        if count <= 0 {
            return Err(CudaError::Unavailable {
                message: "no CUDA devices reported by driver".to_string(),
            });
        }

        let mut device = 0;
        // SAFETY: device 0 is valid when count is greater than zero.
        driver.check("cuDeviceGet", unsafe {
            (driver.cu_device_get)(&raw mut device, 0)
        })?;

        let mut context = std::ptr::null_mut();
        // SAFETY: CUDA writes a newly-created context handle for a valid device.
        driver.check("cuCtxCreate_v2", unsafe {
            (driver.cu_ctx_create)(&raw mut context, 0, device)
        })?;

        Ok(Self {
            inner: Arc::new(ContextInner {
                driver,
                context,
                modules: Mutex::new(HashMap::new()),
                nvjpeg: Mutex::new(None),
            }),
        })
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;

        let mut ptr = 0;
        let buffer = if bytes.is_empty() {
            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        } else {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, bytes.len())
            })?;

            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        };

        if !bytes.is_empty() {
            // SAFETY: ptr is a valid device allocation of bytes.len(), and the
            // host pointer is valid for bytes.len().
            self.inner.driver.check("cuMemcpyHtoD_v2", unsafe {
                (self.inner.driver.cu_memcpy_htod)(
                    ptr,
                    bytes.as_ptr().cast::<c_void>(),
                    bytes.len(),
                )
            })?;
        }

        Ok(buffer)
    }

    pub fn copy_with_kernel(&self, bytes: &[u8]) -> Result<CudaKernelOutput, CudaError> {
        let staging = self.upload(bytes)?;
        let output = self.copy_device_to_device_with_kernel(&staging)?;
        let copy_dispatches = usize::from(!bytes.is_empty());
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: copy_dispatches,
                copy_kernel_dispatches: copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    pub fn decode_jpeg_rgb8_with_nvjpeg(
        &self,
        bytes: &[u8],
        dimensions: (u32, u32),
    ) -> Result<CudaKernelOutput, CudaError> {
        self.inner.set_current()?;
        let (pitch_bytes, byte_len) = rgb8_layout(dimensions)?;
        let output = self.allocate(byte_len)?;
        if byte_len == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let mut state = self
            .inner
            .nvjpeg
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if state.is_none() {
            *state = Some(nvjpeg::NvjpegState::new()?);
        }
        let state = state.as_mut().ok_or_else(|| CudaError::NvjpegUnavailable {
            message: "nvJPEG state did not initialize".to_string(),
        })?;
        state.decode_rgb8(bytes, dimensions, output.device_ptr(), pitch_bytes)?;

        self.inner.driver.check("cuCtxSynchronize", unsafe {
            (self.inner.driver.cu_ctx_synchronize)()
        })?;

        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: true,
            },
        })
    }

    pub fn j2k_forward_rct(
        &self,
        plane0: &mut [f32],
        plane1: &mut [f32],
        plane2: &mut [f32],
    ) -> Result<CudaExecutionStats, CudaError> {
        if plane0.len() != plane1.len() || plane0.len() != plane2.len() {
            return Err(CudaError::ImageTooLarge {
                width: u32::try_from(plane0.len()).unwrap_or(u32::MAX),
                height: 1,
                channels: 3,
            });
        }
        if plane0.is_empty() {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let buffer0 = self.upload(f32_slice_as_bytes(plane0))?;
        let buffer1 = self.upload(f32_slice_as_bytes(plane1))?;
        let buffer2 = self.upload(f32_slice_as_bytes(plane2))?;
        self.launch_j2k_forward_rct_buffers(&buffer0, &buffer1, &buffer2, plane0.len())?;
        buffer0.copy_to_host(f32_slice_as_bytes_mut(plane0))?;
        buffer1.copy_to_host(f32_slice_as_bytes_mut(plane1))?;
        buffer2.copy_to_host(f32_slice_as_bytes_mut(plane2))?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    pub fn j2k_forward_dwt53(
        &self,
        samples: &[f32],
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaDwt53Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != samples.len() {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }
        if samples.is_empty() || num_levels == 0 {
            return Ok(CudaDwt53Output {
                transformed: samples.to_vec(),
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats::default(),
            });
        }

        self.inner.set_current()?;
        let buffer_a = self.upload(f32_slice_as_bytes(samples))?;
        let buffer_b = self.allocate(std::mem::size_of_val(samples))?;
        let mut current_width = width;
        let mut current_height = height;
        let mut levels = Vec::new();
        let mut dispatches = 0usize;

        for _ in 0..num_levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Horizontal,
                &buffer_a,
                &buffer_b,
                CudaDwt53Pass {
                    full_width: width,
                    current_width,
                    current_height,
                    low_extent: low_width,
                },
            )?;
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Vertical,
                &buffer_b,
                &buffer_a,
                CudaDwt53Pass {
                    full_width: width,
                    current_width,
                    current_height,
                    low_extent: low_height,
                },
            )?;
            dispatches = dispatches.saturating_add(2);
            levels.push(CudaDwt53LevelShape {
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
        }

        let mut transformed = vec![0f32; samples.len()];
        buffer_a.copy_to_host(f32_slice_as_bytes_mut(&mut transformed))?;
        Ok(CudaDwt53Output {
            transformed,
            levels,
            ll_width: current_width,
            ll_height: current_height,
            execution: CudaExecutionStats {
                kernel_dispatches: dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    fn launch_j2k_forward_rct_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kForwardRct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut len_u64).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_forward_dwt53_pass(
        &self,
        kernel: CudaKernel,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        pass: CudaDwt53Pass,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut full_width = pass.full_width;
        let mut current_width = pass.current_width;
        let mut current_height = pass.current_height;
        let mut low_extent = pass.low_extent;
        let mut params = [
            (&raw mut input_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut full_width).cast::<c_void>(),
            (&raw mut current_width).cast::<c_void>(),
            (&raw mut current_height).cast::<c_void>(),
            (&raw mut low_extent).cast::<c_void>(),
        ];
        let geometry = j2k_dwt53_launch_geometry(current_width, current_height).ok_or(
            CudaError::ImageTooLarge {
                width: pass.current_width,
                height: pass.current_height,
                channels: 1,
            },
        )?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_kernel(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        self.inner.driver.check("cuLaunchKernel", unsafe {
            (self.inner.driver.cu_launch_kernel)(
                function,
                geometry.grid.0,
                geometry.grid.1,
                geometry.grid.2,
                geometry.block.0,
                geometry.block.1,
                geometry.block.2,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        self.inner.driver.check("cuCtxSynchronize", unsafe {
            (self.inner.driver.cu_ctx_synchronize)()
        })
    }

    pub fn copy_device_to_device_with_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let dst = self.allocate(src.byte_len())?;
        if src.byte_len() == 0 {
            return Ok(dst);
        }

        let function = self.inner.kernel_function(CudaKernel::CopyU8)?;
        let mut dst_ptr = dst.device_ptr();
        let mut src_ptr = src.device_ptr();
        let mut len = u64::try_from(src.byte_len()).map_err(|_| CudaError::LengthTooLarge {
            len: src.byte_len(),
        })?;
        let mut params = [
            (&raw mut dst_ptr).cast::<c_void>(),
            (&raw mut src_ptr).cast::<c_void>(),
            (&raw mut len).cast::<c_void>(),
        ];
        let geometry =
            copy_u8_launch_geometry(src.byte_len()).ok_or(CudaError::LengthTooLarge {
                len: src.byte_len(),
            })?;

        self.launch_kernel(function, geometry, &mut params)?;

        Ok(dst)
    }

    pub fn allocate(&self, len: usize) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let mut ptr = 0;
        if len != 0 {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, len)
            })?;
        }
        Ok(CudaDeviceBuffer {
            context: self.clone(),
            ptr,
            len,
        })
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct CudaDeviceBuffer {
    context: CudaContext,
    ptr: CuDevicePtr,
    len: usize,
}

#[derive(Debug)]
pub struct CudaKernelOutput {
    buffer: CudaDeviceBuffer,
    execution: CudaExecutionStats,
}

#[derive(Debug)]
pub struct CudaDwt53Output {
    transformed: Vec<f32>,
    levels: Vec<CudaDwt53LevelShape>,
    ll_width: u32,
    ll_height: u32,
    execution: CudaExecutionStats,
}

impl CudaDwt53Output {
    pub fn transformed(&self) -> &[f32] {
        &self.transformed
    }

    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDwt53LevelShape {
    pub width: u32,
    pub height: u32,
    pub low_width: u32,
    pub low_height: u32,
    pub high_width: u32,
    pub high_height: u32,
}

#[derive(Clone, Copy, Debug)]
struct CudaDwt53Pass {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_extent: u32,
}

impl CudaKernelOutput {
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaExecutionStats {
    kernel_dispatches: usize,
    copy_kernel_dispatches: usize,
    decode_kernel_dispatches: usize,
    hardware_decode: bool,
}

impl CudaExecutionStats {
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
    }

    pub fn copy_kernel_dispatches(self) -> usize {
        self.copy_kernel_dispatches
    }

    pub fn decode_kernel_dispatches(self) -> usize {
        self.decode_kernel_dispatches
    }

    pub fn used_hardware_decode(self) -> bool {
        self.hardware_decode
    }
}

#[derive(Debug)]
struct CompiledKernel {
    module: CuModule,
    function: CuFunction,
}

impl CompiledKernel {
    fn load(context: &ContextInner, kernel: CudaKernel) -> Result<Self, CudaError> {
        context.set_current()?;
        let mut module = std::ptr::null_mut();
        // SAFETY: image is a NUL-terminated PTX string. CUDA copies or parses
        // it during module load, and the context cache unloads the module on
        // context drop.
        context.driver.check("cuModuleLoadData", unsafe {
            (context.driver.cu_module_load_data)(
                &raw mut module,
                kernel.ptx().as_ptr().cast::<c_void>(),
            )
        })?;
        let mut function = std::ptr::null_mut();
        // SAFETY: name is a NUL-terminated kernel symbol in this module.
        context.driver.check("cuModuleGetFunction", unsafe {
            (context.driver.cu_module_get_function)(
                &raw mut function,
                module,
                kernel.entrypoint().as_ptr().cast::<c_char>(),
            )
        })?;
        Ok(Self { module, function })
    }
}

unsafe impl Send for CompiledKernel {}

impl CudaDeviceBuffer {
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    pub fn byte_len(&self) -> usize {
        self.len
    }

    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        if out.len() < self.len {
            return Err(CudaError::OutputTooSmall {
                required: self.len,
                have: out.len(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }

        self.context.inner.set_current()?;
        // SAFETY: ptr is a live device allocation of self.len bytes, and out is
        // valid for at least self.len bytes.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.ptr,
                self.len,
            )
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.context.inner.set_current();
            // SAFETY: ptr was allocated by this CUDA context. Drop cannot
            // surface errors, so failures are ignored during cleanup.
            let _ = unsafe { (self.context.inner.driver.cu_mem_free)(self.ptr) };
        }
    }
}

fn f32_slice_as_bytes(samples: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns, and the output byte slice is
    // read-only with the same lifetime as the input samples.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn f32_slice_as_bytes_mut(samples: &mut [f32]) -> &mut [u8] {
    // SAFETY: the returned byte slice covers exactly the same initialized f32
    // storage and is used only for CUDA copies into the existing allocation.
    unsafe {
        std::slice::from_raw_parts_mut(
            samples.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn rgb8_layout(dimensions: (u32, u32)) -> Result<(usize, usize), CudaError> {
    let row_bytes = dimensions
        .0
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(3))
        .ok_or(CudaError::ImageTooLarge {
            width: dimensions.0,
            height: dimensions.1,
            channels: 3,
        })?;
    let byte_len =
        row_bytes
            .checked_mul(dimensions.1 as usize)
            .ok_or(CudaError::ImageTooLarge {
                width: dimensions.0,
                height: dimensions.1,
                channels: 3,
            })?;
    Ok((row_bytes, byte_len))
}
