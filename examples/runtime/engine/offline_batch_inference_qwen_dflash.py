"""
Usage:
python3 offline_batch_inference.py
"""

from urllib.request import urlopen

import sglang as sgl
import torch
import random
import numpy as np

# Processing the prompt.
def process_requests(llm: sgl.Engine, prompts: list[str]) -> None:
    # Create a sampling params object.
    sampling_params = {
        "temperature": 0.1,
        # "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        # "repetition_penalty": 1.05,
        # "max_new_tokens": 256,
        "max_new_tokens": 25,
    }
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt_token_ids = output["meta_info"]["prompt_tokens"]
        generated_text = output["text"]
        print(
            f"Prompt length: {prompt_token_ids}, " f"Generated text: {generated_text!r}"
        )


# Create an LLM.85St32oa!7_Z:1Rp
def initialize_engine() -> sgl.Engine:
    ddtree = False
    ddtree = True
    if ddtree:
        llm = sgl.Engine(
            model_path="Qwen/Qwen3-4B",
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="z-lab/Qwen3-4B-DFlash-b16",
            speculative_dflash_enable_ddtree=True,
            # mamba_scheduler_strategy="extra_buffer",
            speculative_num_draft_tokens=16,
            tp_size=1,
            max_running_requests=5,
            page_size=16,
            # page_size=256,
            # attention_backend="dual_chunk_flash_attn",
            # tp_size=4,
            # disable_radix_cache=True,
            # enable_mixed_chunk=False,
            # enable_torch_compile=False,
            # chunked_prefill_size=131072,
            mem_fraction_static=0.9,
            max_total_tokens=1024,
            # piecewise_cuda_graph_compiler="eager",
            # log_level="debug",
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            # enable_deterministic_inference=True,
        )
    else:
        llm = sgl.Engine(
            model_path="Qwen/Qwen3-4B",
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="AngelSlim/Qwen3-4B_eagle3",
            # speculative_dflash_enable_ddtree=True,
            # mamba_scheduler_strategy="extra_buffer",
            speculative_num_draft_tokens=16,
            speculative_num_steps=3,
            speculative_eagle_topk=5,
            tp_size=1,
            max_running_requests=5,
            page_size=16,
            # attention_backend="dual_chunk_flash_attn",
            # tp_size=4,
            # disable_radix_cache=True,
            # enable_mixed_chunk=False,
            # enable_torch_compile=False,
            # chunked_prefill_size=131072,
            mem_fraction_static=0.9,
            max_total_tokens=1024,
            # piecewise_cuda_graph_compiler="eager",
            # log_level="debug",
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            # enable_deterministic_inference=True,
        )
    return llm


def main():
    llm = initialize_engine()
    prompts = [
        "Hello, please introduce yourself",
        # "Which part of computer science do you like best",
    ]

    process_requests(llm, prompts)


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 禁用 CUDA 非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 禁用 torch 动态算子非确定性
    torch.use_deterministic_algorithms(True)
    # 环境变量锁定 CUDA 运算顺序
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    main()
