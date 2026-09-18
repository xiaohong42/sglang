"""CPU byte-layout regressions; no GPU context is needed for these checks."""

import pytest
import torch

from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("layout", ["contiguous", "strided", "transpose"])
def test_mixed_dtype_bucket_preserves_bytes_and_metadata(layout):
    named = []
    for name, dtype in (
        ("bf16", torch.bfloat16),
        ("fn", torch.float8_e4m3fn),
        ("scale", torch.float32),
    ):
        tensor = (torch.arange(64).reshape(8, 8).float() / 8).to(dtype)
        if layout == "strided":
            tensor = tensor[:, ::2]
        elif layout == "transpose":
            tensor = tensor.T
        named.append((name, tensor))
    originals = [tensor.clone() for _, tensor in named]
    bucket = FlattenedTensorBucket(named_tensors=named)
    rebuilt = FlattenedTensorBucket(
        flattened_tensor=bucket.get_flattened_tensor(),
        metadata=bucket.get_metadata(),
    ).reconstruct_tensors()
    assert [name for name, _ in rebuilt] == [name for name, _ in named]
    for (_, original), (_, actual), saved in zip(named, rebuilt, originals):
        assert original.dtype == actual.dtype
        assert original.shape == actual.shape
        assert torch.equal(
            saved.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
        )
        assert torch.equal(
            saved.contiguous().view(torch.uint8),
            original.contiguous().view(torch.uint8),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32])
def test_zero_element_bucket(dtype):
    empty = torch.empty((0, 8), dtype=dtype)
    bucket = FlattenedTensorBucket(named_tensors=[("empty", empty)])
    [(name, actual)] = bucket.reconstruct_tensors()
    assert name == "empty"
    assert actual.dtype == dtype
    assert actual.shape == empty.shape
