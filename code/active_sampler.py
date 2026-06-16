import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import requests


LOCAL_MODELS = {
    "./qwen-32b",
    "./seed-36b",
    "./gpt-120b",
    "./qwen3-235b-a22b-instruct-2507",
}


def get_llm_endpoint(payload):
    local_chat_url = os.getenv("LOCAL_LLM_CHAT_URL", "http://0.0.0.0:8000/v1/chat/completions")
    local_completion_url = os.getenv("LOCAL_LLM_COMPLETION_URL", "http://0.0.0.0:8000/v1/completions")
    remote_chat_url = os.getenv("REMOTE_LLM_CHAT_URL", local_chat_url)

    if payload["model"] not in LOCAL_MODELS:
        payload["model"] = payload["model"].replace("./", "")
        return remote_chat_url

    if "prompt" in payload:
        return local_completion_url

    return local_chat_url


def call_local_llm(payload):
    base_url = get_llm_endpoint(payload)
    api_key = os.getenv("POLICY_API_KEY", "EMPTY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    retry = 10
    while retry > 0:
        print(retry)
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=360,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            retry -= 1
            if retry == 0:
                raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
            time.sleep(2)


class ActiveSampler:
    """Generate and rank candidate responses with an OpenAI-compatible LLM endpoint."""

    def __init__(self, model, KB, device: str = "cuda", max_batch_size: int = 4, dtype: str = "bfloat16"):
        self.llm = model
        self.KB = KB
        self.device = device
        self.max_batch_size = max_batch_size
        self.dtype = dtype

    def generate_candidates(self, payload: dict) -> List[str]:
        """Generate response candidates and optionally return normalized token entropy."""
        if payload["model"] not in {"./qwen-32b", "./seed-36b", "./gpt-120b"} and "prompt" in payload:
            payload["messages"] = [{"role": "user", "content": payload["prompt"]}]
            del payload["prompt"]

        data = call_local_llm(payload)
        results = []

        if "prompt" in payload:
            for choice in data["choices"]:
                text = choice["text"]
                if "</think>" in text:
                    results.append(text.split("</think>", 1)[1].strip())
                elif "</seed:think>" in text:
                    results.append(text.split("</seed:think>", 1)[1].strip())
                else:
                    results.append(text.strip())
        else:
            try:
                for choice in data["choices"]:
                    content = choice["message"]["content"]
                    if "</think>" in content:
                        results.append(content.split("</think>", 1)[1].strip())
                    elif "</seed:think>" in content:
                        results.append(content.split("</seed:think>", 1)[1].strip())
                    else:
                        results.append(content.strip())
            except Exception:
                results.append("Response generation failed.")

        if not payload.get("logprobs"):
            return results

        seq_logprobs = []
        if "prompt" in payload:
            for choice in data["choices"]:
                seq_logprobs.append(choice["logprobs"]["token_logprobs"])
        elif data["choices"][0].get("logprobs"):
            for choice in data["choices"]:
                seq_logprobs.append(
                    [token["logprob"] for token in choice["logprobs"]["content"]]
                )
        else:
            return results

        seq_entropies = []
        for seq in seq_logprobs:
            if not seq:
                seq_entropies.append(0.0)
                continue

            probs = [math.exp(lp) for lp in seq]
            total = sum(probs)
            probs = [p / total for p in probs] if total > 0 else probs
            entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
            seq_entropies.append(entropy)

        mean_entropy = sum(seq_entropies) / len(seq_entropies)
        std_entropy = math.sqrt(
            sum((x - mean_entropy) ** 2 for x in seq_entropies) / len(seq_entropies)
        )
        if std_entropy == 0:
            seq_entropies = [0.0 for _ in seq_entropies]
        else:
            seq_entropies = [(x - mean_entropy) / std_entropy for x in seq_entropies]

        return results, seq_entropies

    def compute_diversity_score(self, samples: List[str], knowledge: List[str] = None, K: int = 5) -> List[float]:
        """Score candidates by inverse similarity to known knowledge entries."""
        if not samples:
            return []

        sim_means = []
        for sample in samples:
            retrieved = self.KB.retrieve_topk_known(sample, K=K)
            if not retrieved:
                sim_means.append(0.0)
                continue

            sims = [item["score"] for item in retrieved]
            sim_means.append(1 - np.mean(sims))

        mean_val = np.mean(sim_means)
        std_val = np.std(sim_means)
        if std_val == 0:
            return [0.0 for _ in sim_means]

        return [(x - mean_val) / std_val for x in sim_means]

    def information_gain(
        self,
        samples: List[str],
        entropy_list: List[float] = None,
        knowledge_retrieved: List[str] = None,
    ) -> Dict[str, float]:
        """Combine uncertainty and knowledge diversity into a candidate score."""
        diversity_scores = self.compute_diversity_score(samples, knowledge_retrieved)
        if len(diversity_scores) != len(samples):
            raise ValueError(f"Length mismatch: {len(diversity_scores)} vs {len(samples)}")

        if entropy_list is None:
            return {sample: score for sample, score in zip(samples, diversity_scores)}

        if len(entropy_list) != len(samples):
            raise ValueError(f"Length mismatch: {len(entropy_list)} vs {len(samples)}")

        def normalize(xs):
            xs = np.array(xs)
            return (xs - xs.min()) / (xs.max() - xs.min() + 1e-8)

        tom_n = normalize(entropy_list)
        kb_n = normalize(diversity_scores)
        return {sample: entropy + diversity for sample, entropy, diversity in zip(samples, tom_n, kb_n)}

    def choose_candidate(self, scores: dict, strategy=None):
        """Choose one candidate by max, min-greedy, random, or top-k random strategy."""
        if not scores:
            raise ValueError("Cannot choose a candidate from an empty score dictionary.")

        if strategy is None or strategy == "max":
            return max(scores.items(), key=lambda x: x[1])[0]

        if strategy == "greedy":
            return min(scores.items(), key=lambda x: x[1])[0]

        if strategy == "random":
            return random.choice(list(scores.keys()))

        if strategy == "topk_random":
            k = min(3, len(scores))
            topk = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            return random.choice([seq for seq, _ in topk])

        raise ValueError(f"Unknown selection strategy: {strategy}")
