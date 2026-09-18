"""CPU guards and independent layout checks for DeepGEMM UE8M0 weight scales."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import builtins
import math
import unittest
from unittest.mock import PropertyMock, call, patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization import fp8 as fp8_quant
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]


def _make_params(n: int = 64, k: int = 128):
    weight = torch.nn.Parameter(torch.zeros((n, k)), requires_grad=False)
    weight_scale = torch.nn.Parameter(torch.ones((1, 1)), requires_grad=False)
    weight_scale.format_ue8m0 = False
    return weight, weight_scale


class TestDeepGemmUE8M0Requant(CustomTestCase):
    def _enabled_deepgemm_ue8m0(self):
        return patch.multiple(
            deep_gemm_wrapper,
            ENABLE_JIT_DEEPGEMM=True,
            DEEPGEMM_SCALE_UE8M0=True,
        )

    def test_helper_requants_supported_deepgemm_bf16_once(self):
        weight, weight_scale = _make_params()

        with self._enabled_deepgemm_ue8m0(), patch.object(
            fp8_utils, "requant_weight_ue8m0_inplace"
        ) as requant:
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )
            fired_again = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertTrue(fired)
        self.assertFalse(fired_again)
        self.assertTrue(weight_scale.format_ue8m0)
        requant.assert_called_once_with(weight, weight_scale, BLOCK_SIZE)

    def test_helper_skips_non_bf16_output(self):
        weight, weight_scale = _make_params()

        with self._enabled_deepgemm_ue8m0(), patch.object(
            fp8_utils, "requant_weight_ue8m0_inplace"
        ) as requant:
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.float16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_shape_deepgemm_will_not_run(self):
        weight, weight_scale = _make_params(n=96, k=128)

        with self._enabled_deepgemm_ue8m0(), patch.object(
            fp8_utils, "requant_weight_ue8m0_inplace"
        ) as requant:
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_non_deepgemm_runner(self):
        weight, weight_scale = _make_params()

        with self._enabled_deepgemm_ue8m0(), patch.object(
            fp8_utils, "requant_weight_ue8m0_inplace"
        ) as requant:
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=False,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_unsupported_block_size(self):
        weight, weight_scale = _make_params()
        unsupported_block_size = [128, 256]

        with self._enabled_deepgemm_ue8m0(), patch.object(
            fp8_utils, "requant_weight_ue8m0_inplace"
        ) as requant:
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                unsupported_block_size,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_compressed_tensors_block_processing_preserves_ue8m0_marker(self):
        scheme = CompressedTensorsW8A8Fp8.__new__(CompressedTensorsW8A8Fp8)
        scheme.strategy = QuantizationStrategy.BLOCK
        scheme.is_static_input_scheme = False
        scheme.weight_block_size = BLOCK_SIZE
        scheme.w8a8_block_fp8_linear = (
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        )

        layer = torch.nn.Module()
        layer.weight, layer.weight_scale = _make_params()
        layer.weight.data = layer.weight.data.to(torch.float8_e4m3fn)
        layer.orig_dtype = torch.bfloat16

        # This fixture models Blackwell's DeepGEMM path, not HIP's FN -> FNUZ
        # normalization. Pin the branch at its consumer so a HIP test host
        # cannot normalize/recreate the parameters between the two calls.
        with (
            self._enabled_deepgemm_ue8m0(),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w8a8_fp8.is_fp8_fnuz",
                return_value=False,
            ),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            scheme.process_weights_after_loading(layer)
            scheme.process_weights_after_loading(layer)

        self.assertTrue(layer.weight_scale.format_ue8m0)
        requant.assert_called_once_with(layer.weight, layer.weight_scale, BLOCK_SIZE)

    def test_fp8_moe_requants_standard_layer_for_deepgemm(self):
        method = fp8_quant.Fp8MoEMethod.__new__(fp8_quant.Fp8MoEMethod)
        method.convert_mxfp8_to_block = False
        method.use_mxfp8 = False
        method.is_fp4_expert = False
        method.dequant_fp4_to_fp8 = False
        method.quant_config = unittest.mock.Mock(weight_block_size=BLOCK_SIZE)

        layer = torch.nn.Module()
        layer.w13_weight, layer.w13_weight_scale_inv = _make_params()
        layer.w2_weight, layer.w2_weight_scale_inv = _make_params()

        def _mark_ue8m0(weight, weight_scale, *args, **kwargs):
            weight_scale.format_ue8m0 = True
            return True

        with patch.multiple(
            fp8_quant,
            _is_cpu=False,
            _is_fp8_fnuz=False,
            _use_aiter=False,
        ), patch.object(
            method, "is_deepgemm_moe_runner_backend_enabled", return_value=True
        ), patch.object(
            fp8_quant,
            "requant_block_scale_ue8m0_for_deepgemm",
            side_effect=_mark_ue8m0,
        ) as requant:
            method.process_weights_after_loading_block_quant(layer)

        self.assertEqual(
            requant.call_args_list,
            [
                call(
                    layer.w13_weight,
                    layer.w13_weight_scale_inv,
                    BLOCK_SIZE,
                    use_deepgemm_runner=True,
                    output_dtype=torch.bfloat16,
                    weight_shape=layer.w13_weight.shape[-2:],
                ),
                call(
                    layer.w2_weight,
                    layer.w2_weight_scale_inv,
                    BLOCK_SIZE,
                    use_deepgemm_runner=True,
                    output_dtype=torch.bfloat16,
                    weight_shape=layer.w2_weight.shape[-2:],
                ),
            ],
        )
        self.assertTrue(layer.w13_weight_scale_inv.format_ue8m0)
        self.assertTrue(layer.w2_weight_scale_inv.format_ue8m0)


class TestUE8M0PackingCpu(CustomTestCase):
    """Independent byte/layout and dequantization oracles, without DeepGEMM."""

    def setUp(self):
        original_import = builtins.__import__

        def no_deep_gemm(name, *args, **kwargs):
            if name == "deep_gemm" or name.startswith("deep_gemm."):
                raise AssertionError("CPU/torch packing must not import DeepGEMM")
            return original_import(name, *args, **kwargs)

        for guard in (
            patch("builtins.__import__", side_effect=no_deep_gemm),
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("CPU test must not initialize a GPU"),
            ),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def _scales_and_reference(mn, k, batched):
        # Powers of two are the input contract. Include both finite exponent
        # extremes and distinguish bytes across lanes, row blocks and experts.
        exponents = (1, 2, 63, 126, 127, 128, 129, 200, 253, 254)
        batches = 2 if batched else 1
        scale_rows = (mn + 127) // 128
        aligned_mn = (mn + 3) // 4 * 4
        packed_k = (k + 3) // 4
        scales = torch.empty((batches, scale_rows, k), dtype=torch.float32)
        raw = bytearray(batches * packed_k * aligned_mn * 4)
        for b in range(batches):
            for r in range(scale_rows):
                for c in range(k):
                    exponent = exponents[(b * 3 + r * k + c) % len(exponents)]
                    scales[b, r, c] = math.ldexp(1.0, exponent - 127)
                    for row in range(r * 128, min((r + 1) * 128, mn)):
                        offset = ((b * packed_k + c // 4) * aligned_mn + row) * 4
                        raw[offset + c % 4] = exponent
        return (scales if batched else scales[0]), raw

    def test_raw_bytes_shape_strides_and_padding(self):
        for mn in (1, 3, 4, 5, 127, 128, 129, 255, 256, 257):
            for k in (1, 3, 4, 5, 8, 9):
                for batched in (False, True):
                    with self.subTest(mn=mn, k=k, batched=batched):
                        scales, expected_bytes = self._scales_and_reference(
                            mn, k, batched
                        )
                        # Noncontiguous scales must retain the same logical bytes.
                        backing = torch.empty((*scales.shape, 2))
                        backing[..., 0] = scales
                        scales = backing[..., 0]
                        original = scales.clone()
                        for use_torch_impl in (False, True):
                            packed = fp8_utils.transform_scale_ue8m0(
                                scales, mn, use_torch_impl=use_torch_impl
                            )
                            aligned_mn = (mn + 3) // 4 * 4
                            packed_k = (k + 3) // 4
                            expected_shape = (mn, packed_k)
                            expected_stride = (1, aligned_mn)
                            if batched:
                                expected_shape = (2, *expected_shape)
                                expected_stride = (
                                    aligned_mn * packed_k,
                                    *expected_stride,
                                )
                            self.assertEqual(packed.dtype, torch.int32)
                            self.assertEqual(packed.device.type, "cpu")
                            self.assertEqual(tuple(packed.shape), expected_shape)
                            self.assertEqual(packed.stride(), expected_stride)
                            raw = torch.empty(0, dtype=torch.uint8).set_(
                                packed.untyped_storage()
                            )
                            self.assertEqual(raw.tolist(), list(expected_bytes))
                            torch.testing.assert_close(
                                scales, original, rtol=0, atol=0
                            )

    def test_cpu_and_rocm_dispatch_do_not_import_deepgemm(self):
        scales = torch.ones((1, 1))
        # CPU tensors must use torch even in an NVIDIA process.
        with patch.object(fp8_utils, "_is_cuda", True):
            fp8_utils.transform_scale_ue8m0(scales, 128)
        # HIP Tensor.is_cuda is True. Emulate only that property on a CPU
        # tensor; all allocation and arithmetic still take place on the CPU.
        with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as cuda:
            cuda.return_value = True
            with patch.object(fp8_utils, "_is_cuda", False):
                fp8_utils.transform_scale_ue8m0(scales, 128)
            with patch.object(fp8_utils, "_is_cuda", True):
                fp8_utils.transform_scale_ue8m0(scales, 128, use_torch_impl=True)
                # The native path remains native, and does not swallow errors
                # from a missing/broken DeepGEMM installation.
                with self.assertRaisesRegex(AssertionError, "must not import"):
                    fp8_utils.transform_scale_ue8m0(scales, 128)

    def test_inverse_and_dequantization_against_reference(self):
        from sglang.srt.utils.weight_checker_comparator import Fp8BlockComparable

        for mn in (1, 127, 128, 129, 255, 256, 257):
            for k in (1, 3, 4, 5):
                for batched in (False, True):
                    with self.subTest(mn=mn, k=k, batched=batched):
                        scales, raw = self._scales_and_reference(mn, k, batched)
                        batches = 2 if batched else 1
                        packed_k = (k + 3) // 4
                        # Construct input directly from reference bytes, NOT by
                        # calling the forward transform being tested.
                        packed = (
                            torch.tensor(list(raw), dtype=torch.uint8)
                            .view(torch.int32)
                            .view(batches, packed_k, (mn + 3) // 4 * 4)
                            .mT[:, :mn, :]
                        )
                        if not batched:
                            packed = packed[0]
                        decoded = fp8_utils.inverse_transform_scale_ue8m0(packed, mn)
                        torch.testing.assert_close(
                            decoded[..., :k], scales, rtol=0, atol=0
                        )
                        self.assertEqual(torch.count_nonzero(decoded[..., k:]), 0)
                        # +/- 0.5 keeps the largest finite scale finite in FP32.
                        # Include the smallest normal scale without BF16 rounding.
                        q = torch.full((*scales.shape[:-2], mn, k * 128), 0.5)
                        q[..., 1::2, :] = -0.5
                        q = q.to(torch.float8_e4m3fn)
                        expected = torch.empty(q.shape, dtype=torch.float32)
                        for r in range((mn + 127) // 128):
                            for c in range(k):
                                rows = slice(r * 128, (r + 1) * 128)
                                cols = slice(c * 128, (c + 1) * 128)
                                expected[..., rows, cols] = (
                                    q[..., rows, cols].float()
                                    * scales[..., r, c, None, None]
                                )
                        for scale in (packed, scales):
                            actual = Fp8BlockComparable(q, scale).dequantize(
                                torch.float32
                            )
                            torch.testing.assert_close(
                                actual, expected, rtol=0, atol=0
                            )

    def test_inverse_still_rejects_corrupted_repeated_rows(self):
        for mn in (128, 130):
            with self.subTest(mn=mn):
                scales = torch.ones(((mn + 127) // 128, 1))
                packed = fp8_utils.transform_scale_ue8m0(scales, mn)
                packed[mn - 1, 0] += 1
                with self.assertRaisesRegex(AssertionError, "sf_repeated != sf_fp32"):
                    fp8_utils.inverse_transform_scale_ue8m0(packed, mn)


if __name__ == "__main__":
    unittest.main(verbosity=3)
