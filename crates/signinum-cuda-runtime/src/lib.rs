// SPDX-License-Identifier: Apache-2.0

//! Thin CUDA Driver API runtime used by signinum CUDA adapter crates.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod kernels;

use std::{
    collections::HashMap,
    ffi::c_void,
    os::raw::{c_char, c_int, c_uint},
    sync::{Arc, Mutex},
};

use kernels::{copy_u8_launch_geometry, CudaKernel};
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
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: usize::from(!bytes.is_empty()),
            },
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

        // SAFETY: function is a live cached CUDA kernel for this context.
        // Kernel arguments point to stack values that live through the
        // synchronous launch. dst and src are live device buffers of at least
        // len bytes.
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
        // SAFETY: synchronizes the current CUDA context so kernel errors are
        // surfaced before returning the output buffer.
        self.inner.driver.check("cuCtxSynchronize", unsafe {
            (self.inner.driver.cu_ctx_synchronize)()
        })?;

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

impl CudaKernelOutput {
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaExecutionStats {
    kernel_dispatches: usize,
}

impl CudaExecutionStats {
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
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
