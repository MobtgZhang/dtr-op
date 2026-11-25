"""Ablation variants (paper Table 2).

DirectLLM          : no retrieval at all (zero-shot).
SingleCase         : top-1 train case, generate by analogy.
MultiCaseNoProto   : top-k train cases, no prototype, no canon, no refine.
MultiCaseCanon     : top-k train cases + canonicalization, still no prototype
                     and no refinement.
"""
from typing import Dict

from ..canonicalize import canonicalize_actions
from ..prompts import direct_prompt, ours_prompt, single_case_prompt
from .base import BaseAgent


class DirectLLM(BaseAgent):
    name = "DirectLLM"

    def query(self, task: str) -> Dict:
        prompt = direct_prompt(task, self.dataset)
        raw = self._gen(prompt)
        return {"pred": self._parse(raw), "prompt": prompt, "raw_response": raw, "retrieved_cases": []}


class SingleCaseAgent(BaseAgent):
    name = "SingleCase"

    def query(self, task: str) -> Dict:
        cases = self.library.recall(task, top_k=1, use_prototype=False) if self.library else []
        if not cases:
            return {"pred": [], "error": "empty library", "retrieved_cases": []}
        case = cases[0]
        prompt = single_case_prompt(task, case, self.dataset)
        raw = self._gen(prompt)
        plan = self._parse(raw)
        return {"pred": plan, "prompt": prompt, "raw_response": raw,
                "retrieved_cases": [{"id": case["id"], "task": case["task"], "actions": case["actions"]}]}


class MultiCaseNoProto(BaseAgent):
    name = "MultiCaseNoProto"

    def __init__(self, llm, dataset: str, top_k: int = 3, canonicalize: bool = False):
        super().__init__(llm, dataset)
        self.top_k = top_k
        self.canonicalize = canonicalize

    def query(self, task: str) -> Dict:
        cases = self.library.recall(task, top_k=self.top_k, use_prototype=False) if self.library else []
        if not cases:
            return {"pred": [], "error": "empty library", "retrieved_cases": []}
        expected_len = int(round(sum(len(c["actions"]) for c in cases) / len(cases)))
        prompt = ours_prompt(task, cases, self.dataset, expected_len)
        raw = self._gen(prompt)
        plan = self._parse(raw)
        if self.canonicalize and self.dataset == "alfworld":
            receps = [a.split("go to ", 1)[1] for c in cases for a in c["actions"] if a.startswith("go to ")]
            plan = canonicalize_actions(plan, context_receps=receps)
        return {"pred": plan, "prompt": prompt, "raw_response": raw,
                "retrieved_cases": [{"id": c["id"], "task": c["task"], "actions": c["actions"]} for c in cases]}


class MultiCaseCanon(MultiCaseNoProto):
    name = "MultiCaseCanon"

    def __init__(self, llm, dataset: str, top_k: int = 3):
        super().__init__(llm, dataset, top_k=top_k, canonicalize=True)
