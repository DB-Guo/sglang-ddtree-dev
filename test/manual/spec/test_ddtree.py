import os
import unittest

import openai

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import gen_radix_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=302, stage="base-b", runner_config="1-gpu-small")

# ref: test/registered/spec/dflash/test_dflash.py
# TODO(DB-guo): MatchedStopMixin
class TestDFlashServerBase(CustomTestCase, MatchedStopMixin, GSM8KMixin):
    max_running_requests = 16
    attention_backend = "flashinfer"
    page_size = 1
    other_launch_args = []
    model = "Qwen/Qwen3-4B"
    draft_model = "z-lab/Qwen3-4B-DFlash-b16"
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.8

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-dflash-enable-ddtree",
            "--mem-fraction-static",
            0.9,
            "--max-total-tokens",
            5120,
            "--page-size",
            str(cls.page_size),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            with (
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
                envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
            ):
                cls.process = popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=launch_args,
                )
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # TODO(DB-guo):
    def test_early_stop(self):
        pass
    def test_eos_handling(self):
        pass

    def test_greedy_determinism(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        prompt = "The capital of France is"
        outputs = []
        for _ in range(2):
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=32,
                temperature=0,
            )
            outputs.append(response.choices[0].text)
        print(f"determinism: {outputs=}")
        self.assertEqual(outputs[0], outputs[1])
        assert self.process.poll() is None


class TestDFlashServerPage256(TestDFlashServerBase):
    page_size = 256

    def test_radix_attention(self):
        import requests

        nodes = gen_radix_tree(num_nodes=50)
        data = {
            "input_ids": [node["input_ids"] for node in nodes],
            "sampling_params": [
                {"max_new_tokens": node["decode_len"], "temperature": 0}
                for node in nodes
            ],
        }
        res = requests.post(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        assert self.process.poll() is None

# TODO(DB-guo):
# class TestDFlashServerChunkedPrefill(TestDFlashServerBase):
#     other_launch_args = ["--chunked-prefill-size", "4"]


# class TestDFlashServerNoCudaGraph(TestDFlashServerBase):
#     other_launch_args = ["--disable-cuda-graph"]

if __name__ == "__main__":
    unittest.main()
