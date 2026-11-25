"""StructuralAnalogicalCBR — the proposed method.

Pipeline (paper Section 3):
  Stage 1  Analogical Recall : hybrid retrieval = task-type prototype (a real
                               training trajectory) + top-k train cases.
  Stage 2  Adaptation        : LLM maps the structure of the recalled cases
                               onto the query (length-aware prompt).
  Stage 3  Refinement        : if the plan length deviates from the expected
                               length by > 2, one regeneration call.
ALFWorld predictions are additionally canonicalized to the benchmark grammar.
"""
from typing import Dict, List

from ..canonicalize import canonicalize_actions
from ..prompts import ours_prompt, refine_prompt
from .base import BaseAgent


class StructuralAnalogicalCBR(BaseAgent):
    name = "StructuralAnalogicalCBR"

    def __init__(self, llm, dataset: str, top_k: int = 3, use_refinement: bool = True,
                 use_prototype: bool = True, use_canonicalization: bool = True):
        super().__init__(llm, dataset)
        self.top_k = top_k
        self.use_refinement = use_refinement
        self.use_prototype = use_prototype
        self.use_canonicalization = use_canonicalization

    def query(self, task: str) -> Dict:
        cases = self.library.recall(task, top_k=self.top_k, use_prototype=self.use_prototype) if self.library else []
        if not cases:
            return {"pred": [], "error": "empty case library", "retrieved_cases": []}

        proto_id = ""
        if self.use_prototype and self.library is not None:
            from ..case_library import infer_alf_task_type
            if self.dataset == "alfworld":
                qg = infer_alf_task_type(task)
            else:
                qg = self.library._match_sw_family(task)
            proto = self.library.prototypes.get(qg)
            proto_id = proto["id"] if proto else ""

        expected_len = int(round(sum(len(c["actions"]) for c in cases) / len(cases)))
        prompt = ours_prompt(task, cases, self.dataset, expected_len, proto_id)
        raw = self._gen(prompt)
        plan = self._parse(raw)

        refine_response = ""
        if self.use_refinement and plan and abs(len(plan) - expected_len) > 2:
            rp = refine_prompt(task, plan, cases, expected_len, self.dataset)
            refine_response = self._gen(rp)
            refined = self._parse(refine_response)
            if refined:
                plan = refined

        if self.dataset == "alfworld" and self.use_canonicalization:
            receps = [a.split("go to ", 1)[1] for a in
                      (x for c in cases for x in c["actions"]) if a.startswith("go to ")]
            plan = canonicalize_actions(plan, context_receps=receps)

        return {
            "pred": plan,
            "prompt": prompt,
            "raw_response": raw,
            "refine_response": refine_response,
            "expected_len": expected_len,
            "retrieved_cases": [{"id": c["id"], "task": c["task"], "actions": c["actions"],
                                 "group": c.get("group", "")} for c in cases],
        }
