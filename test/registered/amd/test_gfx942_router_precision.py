"""Router logits must be repeatable before discrete expert selection."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small")


@unittest.skipUnless(is_gfx942_supported(), "requires gfx942")
class TestGfx942RouterPrecision(unittest.TestCase):
    def test_matches_fp32_reference_and_repeats(self):
        from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm

        torch.manual_seed(123)
        weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
        for rows in (1, 7, 16, 32, 256):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                reference = F.linear(x.float(), weight.float())
                actual = aiter_dsv3_router_gemm(x, weight)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                for _ in range(5):
                    repeat = aiter_dsv3_router_gemm(x, weight)
                    torch.testing.assert_close(repeat, actual, rtol=0, atol=0)

    def test_other_architectures_retain_tuned_dispatch(self):
        import sglang.srt.layers.rocm_linear_utils as utils

        x = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(8, 128, device="cuda", dtype=torch.bfloat16)
        with patch.object(utils, "_IS_GFX942", False), patch.object(
            utils.tgemm, "mm", return_value=x
        ) as gemm:
            self.assertIs(utils.aiter_dsv3_router_gemm(x, weight), x)
            self.assertEqual(gemm.call_args.kwargs["otype"], x.dtype)


if __name__ == "__main__":
    unittest.main()
