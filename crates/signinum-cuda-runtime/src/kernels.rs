use std::os::raw::c_uint;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    CopyU8,
    J2kForwardRct,
    J2kForwardDwt53Horizontal,
    J2kForwardDwt53Vertical,
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
            Self::J2kForwardRct
            | Self::J2kForwardDwt53Horizontal
            | Self::J2kForwardDwt53Vertical => J2K_ENCODE_PTX,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"signinum_copy_u8\0",
            Self::J2kForwardRct => b"signinum_j2k_forward_rct\0",
            Self::J2kForwardDwt53Horizontal => b"signinum_j2k_forward_dwt53_horizontal\0",
            Self::J2kForwardDwt53Vertical => b"signinum_j2k_forward_dwt53_vertical\0",
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
const J2K_ENCODE_THREADS_X: c_uint = 16;
const J2K_ENCODE_THREADS_Y: c_uint = 16;
const J2K_ENCODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/j2k_encode_kernels.ptx"));

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

pub(crate) fn j2k_dwt53_launch_geometry(width: u32, height: u32) -> Option<CudaLaunchGeometry> {
    let grid_x = c_uint::try_from(width.div_ceil(J2K_ENCODE_THREADS_X)).ok()?;
    let grid_y = c_uint::try_from(height.div_ceil(J2K_ENCODE_THREADS_Y)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (grid_x, grid_y, 1),
        block: (J2K_ENCODE_THREADS_X, J2K_ENCODE_THREADS_Y, 1),
    })
}

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
    fn j2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(J2K_ENCODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::J2kForwardRct.entrypoint(),
            b"signinum_j2k_forward_rct\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Horizontal.entrypoint(),
            b"signinum_j2k_forward_dwt53_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Vertical.entrypoint(),
            b"signinum_j2k_forward_dwt53_vertical\0"
        );
    }

    #[test]
    fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
        assert_eq!(copy_u8_launch_geometry(1).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(256).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(257).unwrap().grid, (2, 1, 1));
    }

    #[test]
    fn j2k_dwt53_launch_geometry_uses_16_by_16_thread_blocks() {
        let geometry = j2k_dwt53_launch_geometry(17, 33).unwrap();
        assert_eq!(geometry.grid, (2, 3, 1));
        assert_eq!(geometry.block, (16, 16, 1));
    }
}
