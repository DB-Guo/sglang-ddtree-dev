"""
Usage:
python3 offline_batch_inference.py
"""

from urllib.request import urlopen

import sglang as sgl
import torch

# Processing the prompt.
def process_requests(llm: sgl.Engine, prompts: list[str]) -> None:
    # Create a sampling params object.
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 256,
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
    llm = sgl.Engine(
        model_path="Qwen/Qwen3-4B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="z-lab/Qwen3-4B-DFlash-b16",
        # speculative_dflash_enable_ddtree=True,
        # mamba_scheduler_strategy="extra_buffer",
        speculative_num_draft_tokens=16,
        tp_size=1,
        max_running_requests=5,
        # context_length=1048576,
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
        log_level="debug",
        # speculative_dflash_enable_ddtree=True,
        # disable_cuda_graph=True,
        # disable_piecewise_cuda_graph=True,
    )
    # llm = sgl.Engine(
    #     model_path="Qwen/Qwen3-0.6B",
    #     # speculative_algorithm="NGRAM",
    #     # speculative_num_draft_tokens=4,
    #     # context_length=1048576,
    #     # page_size=256,
    #     # attention_backend="dual_chunk_flash_attn",
    #     # tp_size=4,
    #     # disable_radix_cache=True,
    #     enable_mixed_chunk=False,
    #     enable_torch_compile=False,
    #     # chunked_prefill_size=131072,
    #     mem_fraction_static=0.6,
    #     log_level="DEBUG",
    #     piecewise_cuda_graph_compiler="eager",
    #     trust_remote_code=True,
    #     # disable_cuda_graph=True,
    #     # disable_piecewise_cuda_graph=True,
    # )
    return llm


def main():
    llm = initialize_engine()
    # prompt = load_prompt()
    prompt = "Hello, please introduce yourself"
    prompts = [prompt]
    prompts.append("Which part of computer science do you like best")

    process_requests(llm, prompts)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=500)
    main()
