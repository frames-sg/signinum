use std::os::raw::c_uint;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    CopyU8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaLaunchGeometry {
    pub grid: (c_uint, c_uint, c_uint),
    pub block: (c_uint, c_uint, c_uint),
}

impl CudaKernel {
    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => COPY_U8_PTX,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"signinum_copy_u8\0",
        }
    }
}

pub(crate) fn copy_u8_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

const COPY_U8_THREADS: usize = 256;
const COPY_U8_THREADS_CUDA: c_uint = 256;
const COPY_U8_PTX: &[u8] = concat!(
    r"
.version 7.0
.target sm_52
.address_size 64

.visible .entry signinum_copy_u8(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 len
)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .b16 %u;

    ld.param.u64 %rd1, [dst];
    ld.param.u64 %rd2, [src];
    ld.param.u64 %rd3, [len];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.u64.u32 %rd4, %r4;
    setp.ge.u64 %p, %rd4, %rd3;
    @%p bra DONE;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u8 %u, [%rd5];
    add.u64 %rd6, %rd1, %rd4;
    st.global.u8 [%rd6], %u;
DONE:
    ret;
}
",
    "\0"
)
.as_bytes();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_u8_kernel_metadata_matches_embedded_ptx() {
        let ptx = CudaKernel::CopyU8.ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_copy_u8("));
        assert_eq!(CudaKernel::CopyU8.entrypoint(), b"signinum_copy_u8\0");
    }

    #[test]
    fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
        assert_eq!(copy_u8_launch_geometry(1).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(256).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(257).unwrap().grid, (2, 1, 1));
    }
}
