"""CPU-only tests of the manual gate; no SGLang/GPU/import-side-effect stubs.

Run: python -m pytest test/manual/dsv4/test_gfx942_fp8_reload_helpers.py -q
The tiny checker fixture models snapshot ownership/cleanup, not the production
quantized comparator, Engine RPC, distributed collectives or HIP execution.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "gfx942_manual_reload", Path(__file__).with_name("test_gfx942_fp8_reload.py")
)
manual = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(manual)


def _parameter(value):
    return torch.nn.Parameter(value, requires_grad=False)


@pytest.fixture
def model():
    result = torch.nn.Module()
    result.model = torch.nn.Module()
    result.model.norm = torch.nn.Module()
    result.model.norm.weight = _parameter(torch.ones(8))
    result.weight = _parameter(torch.ones(4, 4).to(torch.float8_e4m3fnuz))
    result.weight.weight_loader = lambda *args: None
    result.weight.is_shuffled = False
    result.weight._fp8_block_aiter_shuffled = False
    result.weight_scale_inv = _parameter(torch.ones(4, 4))
    result.weight_scale_inv.format_ue8m0 = False
    result.register_buffer("freqs_cis", torch.tensor([1 + 2j]), persistent=False)
    result.register_buffer(
        "view", torch.arange(60, dtype=torch.float64).view(5, 12).t()
    )
    result.register_buffer("integer", torch.tensor([2**60], dtype=torch.int64))
    result.register_buffer("flag", torch.tensor([True, False]))
    result.register_buffer("zero", torch.tensor([0.0]))
    result.register_buffer("expert_mask_gpu", torch.ones(2), persistent=False)
    result.register_buffer("unknown_nonpersistent", torch.ones(1), persistent=False)
    result.unknown_nonpersistent._skip_weight_check = True
    return result


@pytest.fixture
def checker(model):
    class TinyChecker:
        def __init__(self):
            self._get_model = lambda: model
            self._ps = SimpleNamespace(tp_rank=0)
            self._snapshot_tensors = None
            self._snapshot_arena = None
            self.compare_calls = 0

        def _snapshot(self):
            state = dict(model.named_parameters())
            state.update(model.named_buffers())
            self._snapshot_tensors = {
                name: tensor.detach().contiguous().clone()
                for name, tensor in state.items()
            }
            self._snapshot_arena = object()

        def _compare(self, allow_quant_error=False, skip_tensor_list=None):
            self.compare_calls += 1
            self._snapshot_tensors = None
            self._snapshot_arena = None

        def handle(self, action, allow_quant_error=False, skip_tensor_list=None):
            if action == "snapshot":
                return self._snapshot()
            if action == "compare":
                return self._compare(allow_quant_error, skip_tensor_list)
            raise AssertionError(action)

    manual._install_raw_checks(TinyChecker, lambda check: check())
    return TinyChecker()


@pytest.fixture(autouse=True)
def no_gpu_context():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def test_full_scope_aliases_and_cleanup(model, checker):
    model.alias = model.weight
    # Buffers' incidental Python loader attributes must not enter the contract.
    model.view.weight_loader = object()
    checker.handle("snapshot")
    proof = checker._manual_raw_proof
    assert proof.names["alias"] == proof.names["weight"]
    assert proof.meta["weight"]["parameter_id"] == id(model.weight)
    assert proof.meta["view"]["spec"][3] == tuple(model.view.stride())
    assert "freqs_cis" in proof.meta
    assert "unknown_nonpersistent" in proof.meta
    assert set(proof.ignored) == {"expert_mask_gpu"}
    model.expert_mask_gpu = torch.ones(7)
    model.view.weight_loader = object()
    checker.handle("compare")
    assert checker.compare_calls == 1
    assert checker._snapshot_tensors is None
    assert checker._snapshot_arena is None
    assert checker._manual_raw_proof is None


@pytest.mark.parametrize(
    "name",
    [
        "weight",
        "weight_scale_inv",
        "freqs_cis",
        "integer",
        "zero",
        "flag",
        "view",
        "unknown_nonpersistent",
    ],
)
def test_raw_mutations_cannot_reach_original_comparator(model, checker, name):
    checker.handle("snapshot")
    tensor = getattr(model, name)
    if name == "weight":
        tensor.view(torch.uint8)[0, 0] += 1
    elif name == "freqs_cis":
        tensor.imag.add_(1)
    elif name == "zero":
        tensor.neg_()  # +0 and -0 are numerically equal, but not byte equal.
    elif name == "flag":
        tensor.logical_not_()
    elif name == "view":
        tensor[-1, -1] += 2**-40  # float32 conversion would hide this difference.
    else:
        tensor.add_(1)
    with pytest.raises(AssertionError, match="raw byte differences"):
        checker.handle("compare")
    assert checker.compare_calls == 0
    assert checker._snapshot_tensors is None


def test_compensated_quant_weight_and_scale_still_fail_raw(model, checker):
    checker.handle("snapshot")
    model.weight.copy_(torch.full((4, 4), 2.0).to(model.weight.dtype))
    model.weight_scale_inv.mul_(0.5)
    torch.testing.assert_close(
        model.weight.float() * model.weight_scale_inv, torch.ones(4, 4)
    )
    with pytest.raises(AssertionError, match="raw byte differences"):
        checker.handle("compare")


@pytest.mark.parametrize(
    "bad",
    [
        float("inf"),
        float("nan"),
        complex(1, float("inf")),
        complex(1, float("nan")),
    ],
)
@pytest.mark.parametrize("before_snapshot", [False, True])
def test_identical_or_new_nonfinite_values_fail(model, checker, bad, before_snapshot):
    model.register_buffer("bad", torch.tensor([bad]))
    if not before_snapshot:
        model.bad.zero_()
    checker.handle("snapshot")
    model.bad.fill_(bad)
    with pytest.raises(AssertionError, match="nonfinite"):
        checker.handle("compare")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda m: setattr(m, "weight", _parameter(m.weight.detach())),
            "Parameter identity",
        ),
        (lambda m: setattr(m.weight, "data", m.weight.data.clone()), "pointer"),
        (lambda m: setattr(m.weight, "data", m.weight.data.t()), "layout"),
        (lambda m: setattr(m.weight, "data", m.weight.data.reshape(2, 8)), "shape"),
        (lambda m: setattr(m.weight, "data", m.weight.data.view(torch.uint8)), "dtype"),
        (
            lambda m: setattr(m.weight, "weight_loader", lambda *args: None),
            "weight_loader",
        ),
        (lambda m: setattr(m.weight, "is_shuffled", True), "FP8 layout attributes"),
        (
            lambda m: setattr(m.weight, "_fp8_block_aiter_shuffled", True),
            "FP8 layout attributes",
        ),
        (
            lambda m: setattr(m.weight_scale_inv, "format_ue8m0", True),
            "FP8 layout attributes",
        ),
        (lambda m: delattr(m.weight, "is_shuffled"), "FP8 layout attributes"),
        (
            lambda m: m.register_buffer("new_buffer", torch.ones(1)),
            "registered scope changed",
        ),
        (lambda m: delattr(m, "integer"), "registered scope changed"),
    ],
)
def test_metadata_or_scope_mutation_rejected(model, checker, mutation, message):
    checker.handle("snapshot")
    if "Parameter identity" == message:
        # Copy metadata onto the replacement to isolate object identity rejection.
        old_attrs = dict(model.weight.__dict__)
        mutation(model)
        model.weight.__dict__.update(old_attrs)
    else:
        mutation(model)
    with pytest.raises(AssertionError, match=message):
        checker.handle("compare")


def test_skip_name_only_excludes_nonpersistent_buffers(model, checker):
    model._non_persistent_buffers_set.remove("expert_mask_gpu")
    checker.handle("snapshot")
    assert "expert_mask_gpu" in checker._manual_raw_proof.meta
    model.expert_mask_gpu.add_(1)
    with pytest.raises(AssertionError, match="raw byte differences"):
        checker.handle("compare")


def test_meta_tensor_fails_closed():
    with pytest.raises(AssertionError, match="meta tensor"):
        manual._tensor_meta("parameter", _parameter(torch.ones(1, device="meta")))


@pytest.mark.parametrize("shape", [(3, 7, 11), (1, 43), (), (0, 5)])
def test_noncontiguous_chunk_size_and_full_coverage(shape):
    original = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(
        shape
    )
    actual = original.transpose(0, -1) if len(shape) > 1 else original
    expected = actual.contiguous()
    total = 0
    for exp, act in manual._paired_chunks(expected, actual, chunk_bytes=24):
        assert exp.numel() * exp.element_size() <= 24
        assert act.numel() * act.element_size() <= 24
        assert torch.equal(exp, act)
        total += exp.numel()
    assert total == actual.numel()


def test_changed_snapshot_retained_restore_exact(model, checker):
    checker.handle("snapshot")
    snapshots, arena = checker._snapshot_tensors, checker._snapshot_arena
    with pytest.raises(AssertionError, match="norm update was not zero"):
        checker.handle(manual._CHANGED_ACTION)  # An RPC no-op is not success.
    assert checker._snapshot_tensors is snapshots
    model.model.norm.weight.zero_()
    checker.handle(manual._CHANGED_ACTION)
    assert checker.compare_calls == 0
    assert checker._snapshot_tensors is snapshots and checker._snapshot_arena is arena
    with pytest.raises(AssertionError, match="overwrite"):
        checker.handle("snapshot")
    model.weight_scale_inv.mul_(2)
    with pytest.raises(AssertionError, match="expected ONLY"):
        checker.handle(manual._CHANGED_ACTION)
    model.weight_scale_inv.mul_(0.5)
    model.model.norm.weight.copy_(snapshots[manual._CHANGED_PARAMETER])
    checker.handle("compare")
    assert checker.compare_calls == 1 and checker._snapshot_tensors is None


def test_inexact_restore_is_not_rebaselined(model, checker):
    checker.handle("snapshot")
    original = checker._snapshot_tensors[manual._CHANGED_PARAMETER].clone()
    model.model.norm.weight.zero_()
    checker.handle(manual._CHANGED_ACTION)
    model.model.norm.weight.copy_(original)
    model.model.norm.weight[0] += 1
    with pytest.raises(AssertionError, match="raw byte differences"):
        checker.handle("compare")
    assert checker.compare_calls == 0


def test_failed_snapshot_releases_arena_without_importing_production(model, checker):
    model.weight.is_shuffled = object()  # Unexpected metadata must fail closed.
    with pytest.raises(AssertionError, match="is_shuffled"):
        checker.handle("snapshot")
    assert checker._snapshot_tensors is None
    assert checker._snapshot_arena is None
    assert checker._manual_raw_proof is None


def _output(prompt_score=-1.0, output_score=-2.0, token=9):
    return {
        "output_ids": [token],
        "meta_info": {
            "input_token_logprobs": [(None, 1, "a"), (prompt_score, 2, "b")],
            "output_token_logprobs": [(output_score, token, "c")],
        },
    }


def test_logprob_change_uses_same_prompt_not_changed_tokens():
    reference = _output()
    with pytest.raises(AssertionError, match="did not affect"):
        manual._assert_changed(reference, _output(output_score=-9.0, token=10))
    manual._assert_changed(reference, _output(prompt_score=-3.0))
    with pytest.raises(AssertionError, match="empty/nonfinite"):
        manual._assert_changed(reference, _output(prompt_score=float("nan")))
    actual = _output(prompt_score=-3.0)
    actual["meta_info"]["input_token_logprobs"][1] = (-3.0, 8, "other token")
    with pytest.raises(AssertionError, match="alignment"):
        manual._assert_changed(reference, actual)
    manual._assert_same(reference, _output(), "same")


@pytest.mark.parametrize(
    "failure", [None, "no_logprob_effect", "raw_check", "transport"]
)
def test_controlled_flow_restores_even_after_failures(model, checker, failure):
    original = model.model.norm.weight.detach().clone()
    baseline = _output()
    sessions, actions = [], []

    class Engine:
        def begin_weight_update(self):
            sessions.append("begin")
            return True, "ok"

        def end_weight_update(self):
            sessions.append("end")
            return {"success": True}

        def update_weights_from_tensor(self, items):
            name, tensor = items[0]
            assert name == "norm.weight"
            model.model.norm.weight.copy_(tensor)
            return failure != "transport" or bool(tensor.count_nonzero()), "transport"

        def flush_cache(self):
            return SimpleNamespace(success=True)

        def generate(self, **kwargs):
            changed = not bool(model.model.norm.weight.count_nonzero())
            return _output(
                prompt_score=(
                    -3.0 if changed and failure != "no_logprob_effect" else -1.0
                )
            )

    def check(action):
        actions.append(action)
        checker.handle(action)
        if failure == "raw_check" and action == manual._CHANGED_ACTION:
            raise AssertionError("simulated raw check failure")

    def run():
        manual._controlled_change(
            Engine(), check, original, "norm.weight", baseline, [1, 2]
        )

    if failure is None:
        run()
    else:
        with pytest.raises(AssertionError):
            run()
    assert sessions == ["begin", "end", "begin", "end"]
    assert actions[0] == "snapshot" and actions[-1] == "compare"
    assert actions.count("snapshot") == 1
    assert checker._snapshot_tensors is None
    torch.testing.assert_close(model.model.norm.weight, original, rtol=0, atol=0)
