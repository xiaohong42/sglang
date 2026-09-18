"""Validate gfx942 DSV4 FP8 reloads against the initialized model.

Run in a fresh process with four otherwise-idle GPUs and a local checkpoint:
  HIP_VISIBLE_DEVICES=0,1,2,3 python test/manual/dsv4/test_gfx942_fp8_reload.py \
      --model-path "$MODEL_PATH" --context-length 1024 --max-total-tokens 4096

Checks two unchanged generations, an empty update session, and three partial
original FP8 weight/scale reloads. A test-only scheduler wrapper compares ALL
registered parameters/buffers (including aliases and separate raw scales),
shape/dtype/stride/storage offset/device/data_ptr, Parameter identity, loader
identity and FP8 layout attributes. Only explicitly listed, actually
nonpersistent dynamic buffers are excluded; each rank reports this scope.
Bytes mean every logical tensor element, not unused storage padding. Complex
finite checks include both components. Transfers/comparisons are chunked; the
existing WeightChecker CPU/mmap snapshot still requires a full rank's storage.

Finally zero the small, nonquantized final RMSNorm weight via a real update
session, require exactly that raw difference and changed same-prompt logprobs,
then resend its original checkpoint tensor in a finally block. Restoration
must match the pre-change snapshot byte-for-byte and baseline tokens/logprobs.
The changed check deliberately retains the snapshot; ordinary compares consume
it only AFTER the independent raw proof. No production source files are edited.

This is not all-weight retransmission, cross-node synchronization, coverage of
unregistered derived caches, optimizer/RL convergence or benchmark accuracy.
"""

import argparse
import json
import os
from pathlib import Path
from weakref import ref

# Do NOT inherit WeightChecker's substring skips or _skip_weight_check. In
# particular DSV4 freqs_cis is static and its complex bytes must be checked.
_DYNAMIC_NONPERSISTENT_BUFFERS = {
    "expert_mask_gpu": "runtime expert-routing mask (nonpersistent only)",
}
_LAYOUT_ATTRS = ("_fp8_block_aiter_shuffled", "is_shuffled", "format_ue8m0")
_MISSING = object()
_RAW_CHUNK_BYTES = 4 * 1024 * 1024
_CHANGED_PARAMETER = "model.norm.weight"
_CHANGED_ACTION = "manual_raw_expect_changed_norm"


def _inventory(model):
    """Enumerate aliases as well as canonical names; skip no learned tensor."""
    tensors, ignored = {}, {}
    for kind, entries in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in entries:
            assert name not in tensors and name not in ignored, name
            parent, _, local_name = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            reason = _DYNAMIC_NONPERSISTENT_BUFFERS.get(local_name)
            if (
                kind == "buffer"
                and local_name in owner._non_persistent_buffers_set
                and reason is not None
            ):
                ignored[name] = {
                    "reason": reason,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
            else:
                tensors[name] = (kind, tensor)
    return tensors, ignored


def _tensor_meta(kind, tensor):
    import torch

    # Fail closed instead of silently skipping unsupported or unmaterialized state.
    assert not tensor.is_meta, "cannot prove bytes of a meta tensor"
    assert tensor.layout == torch.strided, f"unsupported layout: {tensor.layout}"
    assert not tensor.is_conj() and not tensor.is_neg(), "unresolved tensor view bits"
    attrs = []
    for attr in _LAYOUT_ATTRS:
        value = getattr(tensor, attr, _MISSING)
        assert value is _MISSING or type(value) in (bool, int, str, type(None)), attr
        attrs.append((attr, type(value), value))
    # A weak reference avoids pinning a replaced GPU Parameter, while detecting
    # replacement even if Python subsequently reuses its integer id.
    return {
        "spec": (
            kind,
            tuple(tensor.shape),
            tensor.dtype,
            tuple(tensor.stride()),
            tensor.layout,
            tensor.device,
            tensor.storage_offset(),
            tensor.data_ptr(),
        ),
        "parameter": ref(tensor) if kind == "parameter" else None,
        "parameter_id": id(tensor) if kind == "parameter" else None,
        "attrs": tuple(attrs),
        # .data views do not carry weight_loader; only inspect actual Parameters.
        "loader": (
            getattr(tensor, "weight_loader", _MISSING)
            if kind == "parameter"
            else _MISSING
        ),
    }


def _paired_chunks(expected, actual, chunk_bytes):
    """Never flatten/contiguous an entire noncontiguous expert tensor."""
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    assert chunk_bytes > 0
    limit = max(1, chunk_bytes // expected.element_size())
    if expected.numel() <= limit:
        yield expected, actual
    elif expected.is_contiguous() and actual.is_contiguous():
        expected, actual = expected.view(-1), actual.view(-1)
        for start in range(0, expected.numel(), limit):
            size = min(limit, expected.numel() - start)
            yield expected.narrow(0, start, size), actual.narrow(0, start, size)
    else:
        dim = max(range(expected.ndim), key=lambda d: expected.shape[d])
        step = max(1, limit // (expected.numel() // expected.shape[dim]))
        for start in range(0, expected.shape[dim], step):
            size = min(step, expected.shape[dim] - start)
            yield from _paired_chunks(
                expected.narrow(dim, start, size),
                actual.narrow(dim, start, size),
                chunk_bytes,
            )


def _cpu_chunk(tensor):
    return tensor.detach().contiguous().cpu().reshape(-1)


def _assert_finite(tensor, name):
    import torch

    if tensor.is_floating_point() or tensor.is_complex():
        # CPU isfinite lacks some float8 kernels; only this diagnostic converts.
        # Keep float64 precision and BOTH complex components for the actual check.
        value = tensor.float() if str(tensor.dtype).startswith("torch.float8") else tensor
        assert torch.isfinite(value).all().item(), f"nonfinite tensor: {name}"


class _RawModelProof:
    def __init__(self, checker, chunk_bytes=_RAW_CHUNK_BYTES):
        model = checker._get_model()
        tensors, self.ignored = _inventory(model)
        canonical = dict(model.named_parameters())
        canonical.update(model.named_buffers())
        assert checker._snapshot_tensors is not None, "missing WeightChecker snapshot"
        assert set(canonical) == set(checker._snapshot_tensors), "snapshot scope mismatch"
        canonical_names = {id(tensor): name for name, tensor in canonical.items()}
        self.names = {name: canonical_names[id(t)] for name, (_, t) in tensors.items()}
        self.meta = {name: _tensor_meta(kind, t) for name, (kind, t) in tensors.items()}
        self.chunk_bytes = chunk_bytes
        self.scope = {
            "registered_names": len(tensors),
            "parameters": sum(kind == "parameter" for kind, _ in tensors.values()),
            "buffers": sum(kind == "buffer" for kind, _ in tensors.values()),
            "unique_snapshot_tensors": len(set(self.names.values())),
            "unique_raw_bytes": sum(
                canonical[name].numel() * canonical[name].element_size()
                for name in set(self.names.values())
            ),
            "layout_attrs": list(_LAYOUT_ATTRS),
            "excluded_nonpersistent_buffers": self.ignored,
            "boundary": "registered tensors only; excludes unused storage padding",
        }

    def compare(self, checker, *, expect_changed=False):
        import torch

        tensors, ignored = _inventory(checker._get_model())
        assert set(tensors) == set(self.meta), (
            f"registered scope changed: missing={set(self.meta) - set(tensors)}, "
            f"added={set(tensors) - set(self.meta)}"
        )
        snapshots = checker._snapshot_tensors
        assert snapshots is not None, "baseline snapshot was prematurely released"
        differences, checked = {}, set()
        for name, (kind, tensor) in tensors.items():
            before, after = self.meta[name], _tensor_meta(kind, tensor)
            assert before["spec"] == after["spec"], f"shape/dtype/layout/pointer: {name}"
            assert before["attrs"] == after["attrs"], f"FP8 layout attributes: {name}"
            if kind == "parameter":
                assert before["parameter"]() is tensor, f"Parameter identity: {name}"
                assert before["loader"] is after["loader"], (
                    f"weight_loader identity: {name}"
                )
            canonical_name = self.names[name]
            if canonical_name in checked:
                continue
            checked.add(canonical_name)
            expected = snapshots[canonical_name]
            changed_bytes = 0
            for exp, act in _paired_chunks(expected, tensor, self.chunk_bytes):
                exp, act = _cpu_chunk(exp), _cpu_chunk(act)
                _assert_finite(exp, f"snapshot:{name}")
                _assert_finite(act, name)
                changed_bytes += torch.count_nonzero(
                    exp.view(torch.uint8) != act.view(torch.uint8)
                ).item()
                if expect_changed and name == _CHANGED_PARAMETER:
                    assert not torch.count_nonzero(act).item(), "norm update was not zero"
            if changed_bytes:
                differences[name] = changed_bytes
        if expect_changed:
            assert set(differences) == {_CHANGED_PARAMETER}, (
                f"expected ONLY zeroed {_CHANGED_PARAMETER}; raw differences={differences}"
            )
        else:
            assert not differences, f"raw byte differences: {differences}"
        return {
            **self.scope,
            "current_excluded_nonpersistent_buffers": ignored,
            "changed_bytes": differences,
        }


def _install_raw_checks(checker_class, run_checked):
    """Patch only inside scheduler children (or lightweight CPU test fixtures)."""
    original_snapshot = checker_class._snapshot
    original_compare = checker_class._compare
    original_handle = checker_class.handle

    def report(checker, stage, details):
        print(
            json.dumps({"stage": stage, "tp_rank": checker._ps.tp_rank, **details}),
            flush=True,
        )

    def snapshot(checker):
        previous_proof = getattr(checker, "_manual_raw_proof", None)

        def check():
            assert getattr(checker, "_manual_raw_proof", None) is None, (
                "refusing to overwrite an unconsumed baseline"
            )
            original_snapshot(checker)
            checker._manual_raw_proof = _RawModelProof(checker)
            report(checker, "raw-snapshot", checker._manual_raw_proof.scope)

        try:
            return run_checked(check)
        except Exception:
            if previous_proof is None:
                checker._manual_raw_proof = None
                checker._snapshot_tensors = None
                checker._snapshot_arena = None
            raise

    def compare(checker, allow_quant_error=False, skip_tensor_list=None):
        def check():
            assert not allow_quant_error and not skip_tensor_list, "strict manual gate"
            proof = checker._manual_raw_proof
            assert proof is not None, "missing manual raw snapshot"
            # Run BEFORE WeightChecker._compare releases its mmap-backed tensors.
            details = proof.compare(checker)
            original_compare(checker, allow_quant_error=False, skip_tensor_list=None)
            report(checker, "raw-unchanged-or-restored", details)

        try:
            return run_checked(check)
        finally:
            checker._manual_raw_proof = None
            checker._snapshot_tensors = None
            checker._snapshot_arena = None

    def handle(checker, action, allow_quant_error=False, skip_tensor_list=None):
        if action == _CHANGED_ACTION:

            def check():
                assert not allow_quant_error and not skip_tensor_list, (
                    "strict manual gate"
                )
                proof = checker._manual_raw_proof
                assert proof is not None, "missing controlled-change baseline"
                details = proof.compare(checker, expect_changed=True)
                report(checker, "raw-expected-norm-change", details)

            # Do not invoke the equality comparator or consume the original
            # snapshot, even on failure: the caller must still restore against it.
            return run_checked(check)
        return original_handle(checker, action, allow_quant_error, skip_tensor_list)

    checker_class._snapshot = snapshot
    checker_class._compare = compare
    checker_class.handle = handle


def _run_scheduler_with_raw_checks(*args, **kwargs):
    """Top-level, spawn-picklable target: parent-only monkeypatches are insufficient."""
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.utils.weight_checker import WeightChecker

    def run_checked(check):
        error = None
        try:
            check()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # The scheduler normally only publishes one TP response for payload=None.
        # Make a failure on ANY rank visible to every rank and thus the caller.
        group = get_tp_group().cpu_group
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        assert not any(errors), f"manual raw proof failed by TP rank: {errors}"

    _install_raw_checks(WeightChecker, run_checked)
    return run_scheduler_process(*args, **kwargs)


def _check_result(result, operation):
    if isinstance(result, tuple):
        success = result[0]
    elif isinstance(result, dict):
        success = result.get("success", False)
    else:
        success = result.success
    if not success:
        raise AssertionError(f"{operation} failed: {result}")


def _generate(engine, token_ids):
    _check_result(engine.flush_cache(), "flush cache")
    return engine.generate(
        input_ids=token_ids,
        sampling_params={"temperature": 0, "max_new_tokens": 96},
        return_logprob=True,
        logprob_start_len=0,
    )


def _logprobs(output, *, prompt_only=False):
    import numpy as np

    meta = output["meta_info"]
    values = list(meta["input_token_logprobs"])
    if not prompt_only:
        values += list(meta["output_token_logprobs"])
    positions = [(i, value[1]) for i, value in enumerate(values) if value[0] is not None]
    scores = np.asarray([value[0] for value in values if value[0] is not None])
    assert scores.size > 0 and np.isfinite(scores).all(), "empty/nonfinite logprobs"
    return positions, scores


def _assert_same(reference, actual, stage):
    import numpy as np

    assert actual["output_ids"] == reference["output_ids"], stage
    expected_positions, expected = _logprobs(reference)
    observed_positions, observed = _logprobs(actual)
    assert expected_positions == observed_positions, stage
    np.testing.assert_array_equal(observed, expected, err_msg=stage)
    print(
        json.dumps({"stage": stage, "tokens": len(observed), "max_logprob_diff": 0.0}),
        flush=True,
    )


def _assert_changed(reference, actual):
    import numpy as np

    # Compare teacher-forced scores of the SAME prompt tokens. Generated tokens
    # may change, so comparing their scores would not prove a weight effect.
    expected_positions, expected = _logprobs(reference, prompt_only=True)
    observed_positions, observed = _logprobs(actual, prompt_only=True)
    _logprobs(actual)  # Output scores must also remain finite.
    assert expected_positions == observed_positions, "changed prompt logprob alignment"
    assert expected.shape == observed.shape, "changed prompt logprob count"
    max_diff = float(np.max(np.abs(observed - expected)))
    assert max_diff > 1e-4, f"controlled norm update did not affect logprobs: {max_diff}"
    print(
        json.dumps({"stage": "controlled-change", "max_prompt_logprob_diff": max_diff}),
        flush=True,
    )


def _reload_tensors(engine, named_tensors):
    _check_result(engine.begin_weight_update(), "begin")
    try:
        for name, tensor in named_tensors:
            _check_result(engine.update_weights_from_tensor([(name, tensor)]), name)
    finally:
        _check_result(engine.end_weight_update(), "end")


def _controlled_change(engine, check_weights, original, name, baseline, token_ids):
    import torch

    assert original.ndim == 1 and original.numel() > 0, "expected final norm vector"
    assert original.dtype in (torch.float32, torch.float16, torch.bfloat16)
    _assert_finite(original, name)
    assert torch.count_nonzero(original).item(), "original norm must not be all zero"
    check_weights("snapshot")
    try:
        _reload_tensors(engine, [(name, torch.zeros_like(original))])
        check_weights(_CHANGED_ACTION)
        _assert_changed(baseline, _generate(engine, token_ids))
    finally:
        # Never take a new snapshot here: it would legitimize a corrupt restore.
        _reload_tensors(engine, [(name, original)])
        check_weights("compare")
        _assert_same(baseline, _generate(engine, token_ids), "controlled-restore")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=31917)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.context_length <= 0 or args.max_total_tokens < args.context_length:
        parser.error("require 0 < context-length <= max-total-tokens")
    os.environ.update(
        SGLANG_DSV4_FP4_EXPERTS="0",
        SGLANG_SKIP_CHECKPOINT_LOAD_CHECK="1",
        SGLANG_HACK_FLASHMLA_BACKEND="unified_kv_triton",
        SGLANG_OPT_USE_TILELANG_INDEXER="true",
        SGLANG_OPT_USE_JIT_NORM="true",
        SGLANG_OPT_USE_FUSED_COMPRESS="true",
        TORCHINDUCTOR_MAX_AUTOTUNE="0",
        TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE="0",
        AITER_BF16_FP8_MOE_BOUND="0",
    )
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
    from sglang.srt.managers.io_struct import CheckWeightsReqInput

    assert torch.cuda.device_count() >= 4, "requires four GPUs"
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942")
    model_path = Path(args.model_path)
    config = json.loads((model_path / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "fp8"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = encode_messages(
        [{"role": "user", "content": "Compute 17 * 23 and give the result."}],
        thinking_mode="chat",
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert len(token_ids) + 96 < args.context_length, "context too short for the probe"

    class RawCheckedEngine(Engine):
        run_scheduler_process_func = staticmethod(_run_scheduler_with_raw_checks)

    engine = RawCheckedEngine(
        model_path=str(model_path),
        tp_size=4,
        ep_size=4,
        attention_backend="dsv4",
        trust_remote_code=True,
        mem_fraction_static=0.75,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        chunked_prefill_size=min(2048, args.context_length),
        max_running_requests=4,
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        moe_runner_backend="triton",
        port=args.port,
        watchdog_timeout=3600,
    )
    try:

        def check_weights(action):
            result = engine.loop.run_until_complete(
                engine.tokenizer_manager.check_weights(
                    CheckWeightsReqInput(action=action)
                )
            )
            _check_result(result, f"weight {action}")

        baseline = _generate(engine, token_ids)
        for repeat in range(2):
            check_weights("snapshot")
            _assert_same(baseline, _generate(engine, token_ids), f"repeat-{repeat}")
            check_weights("compare")
        index = json.loads((model_path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        selected = [
            name
            for name in index
            if any(
                part in name
                for part in (
                    "layers.0.mlp.experts.0.",
                    "layers.0.ffn.experts.0.",
                    "layers.0.self_attn.wq_b.",
                    "layers.0.attn.wq_b.",
                )
            )
        ]
        assert any("experts" in name for name in selected), selected
        assert any(
            name.endswith((".scale", "_scale_inv")) for name in selected
        ), selected

        def checkpoint_tensors(names):
            for name in names:
                with safe_open(
                    str(model_path / index[name]), framework="pt", device="cpu"
                ) as shard:
                    yield name, shard.get_tensor(name).to("cuda:0")

        for iteration in range(4):
            check_weights("snapshot")
            _reload_tensors(engine, checkpoint_tensors(selected if iteration else []))
            check_weights("compare")
            _assert_same(baseline, _generate(engine, token_ids), f"reload-{iteration}")

        # Both published DSV4 checkpoint naming conventions map to model.norm.weight.
        norm_names = [
            name for name in ("model.norm.weight", "norm.weight") if name in index
        ]
        assert len(norm_names) == 1, f"ambiguous/missing final norm: {norm_names}"
        norm_name, original_norm = next(checkpoint_tensors(norm_names))
        _controlled_change(
            engine, check_weights, original_norm, norm_name, baseline, token_ids
        )
        print("FP8_RELOAD_PASS", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
