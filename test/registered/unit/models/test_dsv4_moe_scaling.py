"""CPU-only DSV4 routing/scale ownership regression tests.

Load the production Python definitions without their GPU import graph: even
importing deepseek_v2 on a HIP build queries devices. GEMMs/launches alone are
replaced by CPU stand-ins; the model constructor, forward methods, HashTopK,
learned routing dispatch, and Triton combine control flow are executed verbatim.
The numerical oracle independently uses fixed, unequal expert outputs.
"""

import ast
import copy
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang"
MODEL = "srt/models/deepseek_v2.py"
TOPK = "srt/layers/moe/topk.py"
HASH = "srt/layers/moe/hash_topk.py"
FUSED = "srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
Output = namedtuple("Output", "topk_weights topk_ids router_logits")
Output.format = "standard"


@lru_cache(maxsize=None)
def _tree(path):
    return ast.parse((ROOT / path).read_text())


def _load(path, names, namespace, methods=None):
    """Keep real bodies intact, omitting import-time GPU registration only."""
    nodes = []
    for node in _tree(path).body:
        if getattr(node, "name", None) not in names:
            continue
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        if isinstance(node, ast.ClassDef) and methods is not None:
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )


class Backend:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        if name.startswith("is_"):
            return lambda: name[3:] == self.value
        raise AttributeError(name)


class DisabledEnvs:
    def __getattr__(self, _):
        return SimpleNamespace(get=lambda: False)


def _combine(namespace, weighted_outputs, scale):
    """Execute the actual final combine branches, mocking only sum kernels."""
    fn = next(
        n
        for n in _tree(FUSED).body
        if getattr(n, "name", None) == "_fused_moe_kernel_sequence"
    )
    start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "routed_scaling_factor is None"
    )
    body = copy.deepcopy(fn.body[start:])
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="routed_scaling_factor"), ast.arg(arg="intermediate_cache3")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    node = ast.FunctionDef(name="combine", args=args, body=body, decorator_list=[])
    out = torch.empty(weighted_outputs.shape[0], weighted_outputs.shape[2])
    values = dict(
        namespace,
        routed_scaling_factor=scale,
        no_combine=False,
        topk=weighted_outputs.shape[1],
        num_tokens=weighted_outputs.shape[0],
        _use_intermediate=True,
        use_fused_moe_sum_all_reduce=False,
        intermediate_cache3=weighted_outputs,
        out_hidden_states=out,
        moe_sum=lambda x, out: out.copy_(x.sum(1)),
        moe_sum_reduce=lambda x, out, s: out.copy_(x.sum(1) * s),
        moe_sum_reduce_triton=lambda x, out, s: out.copy_(x.sum(1) * s),
        moe_sum_reduce_torch_compile=lambda x, out, s: out.copy_(x.sum(1) * s),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(ROOT / FUSED),
            "exec",
        ),
        values,
    )
    return values["combine"](scale, weighted_outputs)


def _environment(aiter, platform="hip", backend="triton", a2a="none", no_combine=False):
    ns = dict(
        __name__=__name__,
        torch=torch,
        nn=nn,
        F=F,
        dataclass=dataclass,
        _is_hip=platform == "hip",
        _is_cuda=platform == "cuda",
        _is_cpu=False,
        _is_musa=False,
        _is_npu=False,
        _is_xpu=False,
        _use_aiter=aiter,
        _RENORMALIZE_SUM_EPSILON=1e-20,
        _device_sm=90,
        envs=DisabledEnvs(),
        get_exec=lambda: SimpleNamespace(
            moe=SimpleNamespace(
                enable_eplb=False, enable_waterfill=False, ep_num_redundant_experts=0
            ),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        ),
        get_parallel=lambda: SimpleNamespace(tp_size=1, moe_ep_size=1),
        get_forward=lambda: SimpleNamespace(flashinfer_trtllm_bypass=False),
        get_moe_a2a_backend=lambda: Backend(a2a),
        # Deliberately disagree with the actual runner: global flags cannot own scale.
        get_moe_runner_backend=lambda: Backend(
            "aiter" if backend == "triton" else "triton"
        ),
        is_shared_experts_fusion_disabled=lambda: True,
        has_per_rank_fused_shared_slots=lambda _: False,
        add_prefix=lambda name, prefix: name,
        RoutingMethodType=SimpleNamespace(DeepSeekV3=0),
        SboFlags=SimpleNamespace(fuse_shared_experts_inside_sbo=lambda: False),
        use_intel_amx_backend=lambda _: False,
        KTEPWrapperMethod=type("KTEPWrapperMethod", (), {}),
        TopKOutputFormat=SimpleNamespace(STANDARD="standard", BYPASSED="bypassed"),
        StandardTopKOutput=Output,
        get_global_expert_distribution_recorder=lambda: SimpleNamespace(
            on_select_experts=lambda **_: None
        ),
        topk_ids_logical_to_physical=lambda ids, *_: ids,
        _mask_topk_ids_padded_region=lambda *_: None,
        _zero_topk_weights_padded_region=lambda *_: None,
        is_hip=lambda: platform == "hip",
        should_skip_post_experts_all_reduce=lambda **_: True,
        maybe_fuse_routed_scale_and_shared_add=lambda experts, routed, shared, scale: (
            routed if shared is None else routed + shared
        ),
        expert_location_dispatch=SimpleNamespace(
            transform_select_experts_inputs=lambda **kw: (
                kw["router_logits"],
                kw["correction_bias"],
            )
        ),
        _post_process_topk_ids=lambda **kw: (
            kw["topk_ids"],
            kw["topk_weights"],
            kw["topk_ids"],
        ),
        is_gfx942_supported=lambda: platform == "hip",
        is_batch_invariant_mode_enabled=lambda: False,
        is_arch_support_pdl=lambda: False,
    )
    _load(FUSED, {"_use_moe_sum_reduce_torch_compile"}, ns)
    _load(HASH, {"HashTopK"}, ns)
    _load(
        TOPK,
        {
            "TopKConfig",
            "biased_topk_impl",
            "biased_topk_jit_kernel_impl",
            "select_experts",
        },
        ns,
    )

    class LearnedTopK:
        def __init__(self, **kw):
            self.layer_id = kw.pop("layer_id")
            for key in ("quant_config", "is_fp4_experts"):
                kw.pop(key, None)
            self.topk_config = ns["TopKConfig"](**kw)

        def __call__(self, hidden, logits, **kw):
            return ns["select_experts"](hidden, logits, self.topk_config, **kw)

    ns["TopK"] = LearnedTopK
    _load(
        MODEL,
        {"DeepseekV2MoE"},
        ns,
        methods={"__init__", "forward_normal", "forward_normal_dual_stream"},
    )
    _load(MODEL, {"MoEGate"}, ns, methods={"forward"})
    real_gate = ns["MoEGate"]
    ns["MoEGate"] = lambda **kw: SimpleNamespace(
        e_score_correction_bias=(
            None if kw["is_hash_moe"] else torch.tensor([0.2, -0.1, 0.3, 0.0])
        )
    )

    class FixedExperts:
        def __init__(self, **kw):
            self.should_fuse_routed_scaling_factor_in_topk = False
            self.quant_method = object()
            self.moe_runner_config = SimpleNamespace(
                inplace=False,
                no_combine=no_combine,
                routed_scaling_factor=kw["routed_scaling_factor"],
            )
            self.runner = SimpleNamespace(
                runner_backend=Backend(backend), config=self.moe_runner_config
            )
            self.fixed_outputs = torch.tensor(
                [[1.0, -3.0, 2.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 7.0], [8.0, -1.0, 3.0]]
            )

        def __call__(self, hidden, topk, **kw):
            weighted = self.fixed_outputs[
                topk.topk_ids.long()
            ] * topk.topk_weights.unsqueeze(-1)
            return _combine(ns, weighted, self.moe_runner_config.routed_scaling_factor)

    ns["get_moe_impl_class"] = lambda _: FixedExperts
    return ns, real_gate


def _config(**overrides):
    values = dict(
        routed_scaling_factor=1.5,
        n_shared_experts=0,
        n_routed_experts=4,
        hidden_act="silu",
        hidden_size=3,
        moe_intermediate_size=8,
        num_experts_per_tok=2,
        vocab_size=4,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_hash_layers=3,
        swiglu_limit=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _kernel_standins(ns):
    aiter = ModuleType("aiter")

    def topk_gating(weights, ids, logits, bias, renorm, scale, **kw):
        w, i = ns["biased_topk_impl"](
            torch.empty(logits.shape[0], 1),
            logits,
            bias,
            ids.shape[1],
            renorm,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=True,
        )
        weights.copy_(w)
        ids.copy_(i)

    aiter.topk_gating = topk_gating
    gate = ModuleType("sglang.kernels.ops.moe.moe_fused_gate")

    def moe_fused_gate(logits, bias, **kw):
        return ns["biased_topk_impl"](
            torch.empty(logits.shape[0], 1), logits, bias, **kw
        )

    gate.moe_fused_gate = moe_fused_gate
    return patch.dict("sys.modules", {"aiter": aiter, gate.__name__: gate})


@pytest.mark.parametrize(
    "aiter,platform", [(False, "hip"), (True, "hip"), (False, "cuda")]
)
@pytest.mark.parametrize(
    "layer_id,is_nextn", [(0, False), (2, False), (3, False), (0, True)]
)
@pytest.mark.parametrize("dual_stream", [False, True])
@pytest.mark.parametrize("num_tokens", [3, 34])
@pytest.mark.parametrize("scale", [1.0, 1.5, 2.5])
def test_fixed_experts_scale_exactly_once(
    aiter, platform, layer_id, is_nextn, dual_stream, num_tokens, scale
):
    ns, _ = _environment(aiter, platform)
    moe = ns["DeepseekV2MoE"](
        _config(routed_scaling_factor=scale),
        layer_id,
        is_nextn=is_nextn,
        is_deepseek_v4=True,
    )
    assert moe.is_hash == (layer_id < 3 and not is_nextn)
    logits = torch.tensor([[0.2, -0.5, 2.0, 0.7]]).repeat(num_tokens, 1)
    ids = torch.arange(num_tokens) % 4
    bias = moe.gate.e_score_correction_bias
    moe.gate = lambda *_: logits
    moe._maybe_quant_moe_input_once = lambda _: None
    shared = torch.tensor([[0.25, -0.75, 1.25]]).repeat(num_tokens, 1)
    moe._forward_shared_experts = lambda *args, **kw: shared.clone()
    moe.alt_stream = Mock()
    hidden = torch.zeros(num_tokens, 3)
    # Independent oracle: no production routing or scale helper is reused.
    scores = torch.sqrt(torch.log1p(torch.exp(logits.double())))
    if moe.is_hash:
        selected = torch.stack((ids, (ids + 1) % 4), dim=1)
    else:
        selected = torch.argsort(scores + bias.double(), dim=-1, descending=True)[:, :2]
    weights = scores.gather(1, selected)
    weights /= weights.sum(1, keepdim=True)
    expected = (
        moe.experts.fixed_outputs.double()[selected] * weights.unsqueeze(-1)
    ).sum(1) * scale + shared
    method = moe.forward_normal_dual_stream if dual_stream else moe.forward_normal
    with _kernel_standins(ns), patch.object(
        torch.cuda, "current_stream", return_value=Mock()
    ), patch.object(torch.cuda, "stream", return_value=nullcontext()):
        actual = method(hidden, input_ids_global=ids)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "fields,layer_id,is_nextn,expected",
    [
        ({"num_hash_layers": 3}, 2, False, True),
        ({"num_hash_layers": 3}, 3, False, False),
        ({"num_hash_layers": 3}, 0, True, False),
        ({"n_hash_layers": 3}, 2, False, True),
        ({"n_hash_layers": 3}, 3, False, False),
        ({"n_hash_layers": 3}, 0, True, False),
        ({"num_hash_layers": 0, "n_hash_layers": 3}, 0, False, False),
        ({"num_hash_layers": 1, "n_hash_layers": 3}, 1, False, False),
        ({}, 0, False, False),
    ],
)
def test_hash_config_alias_and_nextn(fields, layer_id, is_nextn, expected):
    ns, _ = _environment(True)
    cfg = _config()
    del cfg.num_hash_layers
    vars(cfg).update(fields)
    moe = ns["DeepseekV2MoE"](cfg, layer_id, is_nextn=is_nextn, is_deepseek_v4=True)
    assert moe.is_hash is expected


@pytest.mark.parametrize(
    "platform,backend,a2a,no_combine,v4,expected",
    [
        ("hip", "triton", "none", False, True, True),
        ("hip", "aiter", "none", False, True, False),
        ("hip", "triton", "deepep", False, True, False),
        ("hip", "triton", "none", True, True, False),
        ("hip", "triton", "none", False, False, False),
        ("cuda", "triton", "none", False, True, False),
        ("cpu", "triton", "none", False, True, False),
    ],
)
def test_actual_runner_owns_scale(platform, backend, a2a, no_combine, v4, expected):
    ns, _ = _environment(False, platform, backend, a2a, no_combine)
    moe = ns["DeepseekV2MoE"](_config(num_hash_layers=0), 3, is_deepseek_v4=v4)
    assert moe._hip_triton_routed_scale_finalized is expected
    assert not moe.topk.topk_config.apply_routed_scaling_factor_on_output


@pytest.mark.parametrize("ownership", ["no_runner", "topk", "ktep", "fused_shared"])
def test_existing_scale_owners_are_not_overridden(ownership):
    ns, _ = _environment(False)
    base = ns["get_moe_impl_class"](None)

    class OtherOwner(base):
        def __init__(self, **kw):
            super().__init__(**kw)
            if ownership == "no_runner":
                self.runner = None
            elif ownership == "topk":
                self.should_fuse_routed_scaling_factor_in_topk = True
            elif ownership == "ktep":
                self.quant_method = ns["KTEPWrapperMethod"]()

    ns["get_moe_impl_class"] = lambda _: OtherOwner
    cfg = _config(num_hash_layers=0)
    if ownership == "fused_shared":
        cfg.n_shared_experts = 1
        ns["is_shared_experts_fusion_disabled"] = lambda: False
    moe = ns["DeepseekV2MoE"](cfg, 3, is_deepseek_v4=True)
    assert not moe._hip_triton_routed_scale_finalized
    assert moe.topk.topk_config.apply_routed_scaling_factor_on_output == (
        ownership == "topk"
    )


@pytest.mark.parametrize("aiter", [False, True])
def test_other_hip_architecture_scale_policy_unchanged(aiter):
    ns, _ = _environment(aiter)
    ns["is_gfx942_supported"] = lambda: False
    moe = ns["DeepseekV2MoE"](_config(), 0, is_deepseek_v4=True)
    assert not moe._hip_triton_routed_scale_finalized
    assert not moe.topk.apply_routed_scaling_factor_on_output


@pytest.mark.parametrize("layer_id", [0, 3])
def test_nvidia_existing_scale_and_alias_policy_unchanged(layer_id):
    ns, _ = _environment(False, "cuda")
    moe = ns["DeepseekV2MoE"](_config(), layer_id, is_deepseek_v4=True)
    apply_scale = (
        moe.topk.apply_routed_scaling_factor_on_output
        if moe.is_hash
        else moe.topk.topk_config.apply_routed_scaling_factor_on_output
    )
    assert not apply_scale
    cfg = _config(n_hash_layers=3)
    del cfg.num_hash_layers
    assert not ns["DeepseekV2MoE"](cfg, 0, is_deepseek_v4=True).is_hash


@pytest.mark.parametrize("aiter", [False, True])
def test_gfx942_router_fp32_independent_of_aiter(aiter):
    ns, gate_cls = _environment(aiter)
    gate = gate_cls()
    gate.is_deepseek_v4 = True
    gate.weight = nn.Parameter(
        torch.tensor([[1.0, 0.00390625], [1.0, 0.0]], dtype=torch.bfloat16)
    )
    hidden = torch.ones(1, 2, dtype=torch.bfloat16)
    ns["aiter_dsv3_router_gemm"] = Mock(
        side_effect=AssertionError("must not need AITER")
    )
    actual = gate.forward(hidden)
    expected = F.linear(hidden.float(), gate.weight.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0] > actual[0, 1]
    # BF16 rounds the small but routing-relevant difference to a tie.
    assert F.linear(hidden, gate.weight)[0, 0] == F.linear(hidden, gate.weight)[0, 1]


@pytest.mark.parametrize(
    "v4,gfx942,platform",
    [(False, True, "hip"), (True, False, "hip"), (True, False, "cuda")],
)
def test_router_other_paths_unchanged(v4, gfx942, platform):
    ns, gate_cls = _environment(False, platform)
    ns["is_gfx942_supported"] = lambda: gfx942
    gate = gate_cls()
    gate.is_deepseek_v4 = v4
    gate.weight = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    hidden = torch.ones(1, 2, dtype=torch.bfloat16)
    gemm = ModuleType("sglang.kernels.ops.attention.dsv4")
    gemm.linear_bf16_fp32 = Mock(
        return_value=torch.full((1, 2), 7.0, dtype=torch.float32)
    )
    with patch.dict("sys.modules", {gemm.__name__: gemm}):
        actual = gate.forward(hidden)
    if platform == "cuda":
        gemm.linear_bf16_fp32.assert_called_once_with(hidden, gate.weight)
        assert (actual == 7).all()
    else:
        assert actual.dtype == torch.bfloat16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
