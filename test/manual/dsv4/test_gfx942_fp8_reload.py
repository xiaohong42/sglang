"""Validate gfx942 DSV4 FP8 reloads against the initialized model.

Run in a fresh process with four otherwise-idle GPUs:
  HIP_VISIBLE_DEVICES=0,1,2,3 python test/manual/dsv4/test_gfx942_fp8_reload.py \
      --model-path /models/DeepSeek-V4-Flash-FP8

Checks unchanged repeated generation, an empty update session, and three
partial FP8 weight/scale reloads. Every session compares a complete model
snapshot and requires identical generated tokens and logprobs. This is not
an RL convergence or benchmark-accuracy test.
"""

import argparse
import json
import os
from pathlib import Path


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
    engine.flush_cache()
    return engine.generate(
        input_ids=token_ids,
        sampling_params={"temperature": 0, "max_new_tokens": 96},
        return_logprob=True,
        logprob_start_len=0,
    )


def _assert_same(reference, actual, stage):
    import numpy as np

    assert actual["output_ids"] == reference["output_ids"], stage

    def logprobs(output):
        meta = output["meta_info"]
        values = meta["input_token_logprobs"] + meta["output_token_logprobs"]
        return np.asarray([value[0] for value in values if value[0] is not None])

    expected, observed = logprobs(reference), logprobs(actual)
    assert np.isfinite(observed).all(), stage
    np.testing.assert_array_equal(observed, expected, err_msg=stage)
    print(
        json.dumps({"stage": stage, "tokens": len(observed), "max_logprob_diff": 0.0}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=31917)
    args = parser.parse_args()
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
    engine = Engine(
        model_path=str(model_path),
        tp_size=4,
        ep_size=4,
        attention_backend="dsv4",
        trust_remote_code=True,
        mem_fraction_static=0.75,
        context_length=8192,
        chunked_prefill_size=2048,
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
            _assert_same(baseline, _generate(engine, token_ids), f"repeat-{repeat}")
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
        for iteration in range(4):
            check_weights("snapshot")
            _check_result(engine.begin_weight_update(), "begin")
            if iteration:
                for name in selected:
                    with safe_open(
                        str(model_path / index[name]), framework="pt", device="cpu"
                    ) as shard:
                        tensor = shard.get_tensor(name).to("cuda:0")
                    _check_result(
                        engine.update_weights_from_tensor([(name, tensor)]), name
                    )
            _check_result(engine.end_weight_update(), "end")
            check_weights("compare")
            _assert_same(baseline, _generate(engine, token_ids), f"reload-{iteration}")
        print("FP8_RELOAD_PASS", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
