// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{Buffer, Device};

use crate::{
    abi::{
        JpegBaselineEncodeStatus, JpegDecodeStatus, FAST420_STATUS_HUFFMAN, FAST420_STATUS_OK,
        FAST420_STATUS_TRUNCATED, JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS,
        JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW,
    },
    buffers::{checked_buffer_slice, checked_fill_buffer_u8, new_shared_buffer},
    Error,
};

pub(super) fn jpeg_baseline_encode_status_error(status: JpegBaselineEncodeStatus) -> Error {
    let message = match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW => {
            "JPEG Baseline Metal encode entropy output exceeded capacity".to_string()
        }
        JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN => format!(
            "JPEG Baseline Metal encode missing Huffman code for symbol {}",
            status.detail
        ),
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => {
            "JPEG Baseline Metal encode received invalid kernel parameters".to_string()
        }
        other => format!("JPEG Baseline Metal encode failed with status {other}"),
    };
    Error::MetalKernel { message }
}

pub(super) fn fast_decode_status_error(status: JpegDecodeStatus) -> Error {
    let reason = match status.code {
        FAST420_STATUS_TRUNCATED => "truncated entropy stream",
        FAST420_STATUS_HUFFMAN => "invalid Huffman stream",
        _ => "unexpected Metal JPEG failure",
    };
    Error::MetalKernel {
        message: format!("{reason} at entropy byte {}", status.position),
    }
}

pub(super) fn decode_status_buffer(device: &Device, count: u32) -> Result<Buffer, Error> {
    let bytes = crate::batch_allocation::checked_count_product(
        count as usize,
        core::mem::size_of::<JpegDecodeStatus>(),
        "JPEG Metal decode status bytes",
    )?;
    let buffer = new_shared_buffer(device, bytes)?;
    checked_fill_buffer_u8(&buffer, bytes, 0, "initialize JPEG Metal decode statuses")?;
    Ok(buffer)
}

pub(super) fn first_decode_error_status(
    buffer: &Buffer,
    count: u32,
) -> Result<Option<JpegDecodeStatus>, Error> {
    #[cfg(test)]
    tests::observe_status_read();
    let statuses =
        checked_buffer_slice::<JpegDecodeStatus>(buffer, count as usize, "decode statuses")?;
    Ok(statuses
        .iter()
        .copied()
        .find(|status| status.code != FAST420_STATUS_OK))
}

pub(super) fn fast422_status_error(status: JpegDecodeStatus) -> Error {
    Error::MetalKernel {
        message: format!(
            "unexpected Metal fast422 failure at entropy byte {}",
            status.position
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    thread_local! {
        static STATUS_READ_OBSERVER: RefCell<Option<Box<dyn Fn()>>> = RefCell::new(None);
    }

    pub(super) fn observe_status_read() {
        STATUS_READ_OBSERVER.with(|observer| {
            if let Some(observer) = observer.borrow().as_ref() {
                observer();
            }
        });
    }

    struct ResetObserver;
    impl Drop for ResetObserver {
        fn drop(&mut self) {
            STATUS_READ_OBSERVER.with(|observer| *observer.borrow_mut() = None);
        }
    }

    #[test]
    fn batch_status_read_retains_scratch_ownership() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let observed_session = session.clone();
        let reads = std::rc::Rc::new(std::cell::Cell::new(0));
        let observed_reads = reads.clone();
        let _reset = ResetObserver;
        STATUS_READ_OBSERVER.with(|observer| {
            *observer.borrow_mut() = Some(Box::new(move || {
                let runtime = observed_session.runtime_result().as_ref().expect("runtime");
                assert!(
                    runtime.batch_scratch_in_use_for_test(),
                    "another submission must not reuse scratch before status consumption"
                );
                observed_reads.set(observed_reads.get() + 1);
            }));
        });
        let decoder =
            crate::Decoder::new(include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg"))
                .expect("decoder");
        let requests = [decoder.rgb8_metal_request(crate::batch::BatchOp::Full)];
        let output = crate::MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 1)
            .expect("texture output");
        super::super::batch_entry::decode_full_rgb8_batch_into_textures_with_session(
            &requests, &output, &session,
        )
        .expect("texture batch")
        .expect("supported batch");
        super::super::batch_entry::decode_full_batch_to_surfaces_with_session(&requests, &session)
            .expect("RGB batch")
            .expect("supported batch");
        assert_eq!(reads.get(), 2, "both completion paths must inspect status");
    }
}
