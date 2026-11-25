"""Offline re-implementations of published baselines.

IMPORTANT: Reflexion / CLIN / ReflAct are originally *interactive* methods that
require live environment feedback.  Because this paper evaluates text-only,
offline planning (no simulator access during planning), they are adapted here
to the offline setting as faithfully as possible:

  ReflexionStyle : generate -> verbal self-reflection -> regenerate (Shinn et
                   al. 2023, minus environment feedback).
  ClinStyle      : Reflexion + a persistent cross-task memory of distilled
                   lessons (Madaan et al. 2024 / CLIN's episodic memory, minus
                   environment feedback).
  ReflActStyle   : step-by-step ReAct-style self-imagined execution
                   (Madaan et al. 2023), capped at MAX_STEPS actions.

SYNAPSE-style (Zheng et al. 2023) and ExpeL-style (Zhao et al. 2024) are
naturally offline-compatible: trajectory retrieval as in-context demos, and
rule/insight distillation from training trajectories, respectively.

All methods use the same backbone LLM and the same real case library.
"""
import re
from typing import Dict, List

from ..case_library import infer_alf_task_type
from ..prompts import (
    clin_lesson_prompt,
    expel_plan_prompt,
    expel_rules_prompt,
    reflact_step_prompt,
    reflexion_plan_prompt,
    reflexion_reflect_prompt,
    synapse_prompt,
)
from .base import BaseAgent


class SynapseStyle(BaseAgent):
    """Retrieve top-k full trajectories, few-shot generation (no prototype,
    no length signal, no canonicalization, no refinement)."""
    name = "SYNAPSE-style"

    def __init__(self, llm, dataset: str, top_k: int = 3):
        super().__init__(llm, dataset)
        self.top_k = top_k

    def query(self, task: str) -> Dict:
        cases = self.library.recall(task, top_k=self.top_k, use_prototype=False) if self.library else []
        if not cases:
            return {"pred": [], "error": "empty library", "retrieved_cases": []}
        prompt = synapse_prompt(task, cases, self.dataset)
        raw = self._gen(prompt)
        return {"pred": self._parse(raw), "prompt": prompt, "raw_response": raw,
                "retrieved_cases": [{"id": c["id"], "task": c["task"], "actions": c["actions"]} for c in cases]}


class ExpelStyle(BaseAgent):
    """Distill rules per task group from training demos, then plan with rules."""
    name = "Expel-style"

    def __init__(self, llm, dataset: str, demos_per_group: int = 2):
        super().__init__(llm, dataset)
        self.demos_per_group = demos_per_group
        self.rules: Dict[str, str] = {}

    def bootstrap(self, library) -> None:
        super().bootstrap(library)
        groups: Dict[str, List[Dict]] = {}
        for c in library.cases:
            if c["group"]:
                groups.setdefault(c["group"], []).append(c)
        for g, cases in groups.items():
            demos = sorted(cases, key=lambda c: c["id"])[: self.demos_per_group]
            try:
                self.rules[g] = self._gen(expel_rules_prompt(g, demos, self.dataset), max_tokens=2048)
            except Exception as exc:  # keep going even if one group fails
                self.rules[g] = f"(rule extraction failed: {exc})"

    def query(self, task: str) -> Dict:
        if self.dataset == "alfworld":
            g = infer_alf_task_type(task)
        else:
            g = self.library._match_sw_family(task) if self.library else ""
        rules = self.rules.get(g, "(no rules available for this task type)")
        cases = self.library.recall(task, top_k=1, use_prototype=False) if self.library else []
        case = cases[0] if cases else {"task": "", "actions": []}
        prompt = expel_plan_prompt(task, rules, case, self.dataset)
        raw = self._gen(prompt)
        return {"pred": self._parse(raw), "prompt": prompt, "raw_response": raw,
                "retrieved_cases": [{"id": case.get("id", ""), "task": case.get("task", ""), "actions": case.get("actions", [])}],
                "rules_group": g}


class ReflexionStyle(BaseAgent):
    """Offline Reflexion: generate -> self-reflect -> regenerate."""
    name = "Reflexion-style"

    def query(self, task: str) -> Dict:
        prompt1 = reflexion_plan_prompt(task, self.dataset)
        raw1 = self._gen(prompt1)
        plan = self._parse(raw1)

        reflect_prompt = reflexion_reflect_prompt(task, plan, self.dataset)
        reflect_raw = self._gen(reflect_prompt, max_tokens=4096)

        reflection, improved = self._split_reflection(reflect_raw)
        plan2 = improved if improved else plan

        prompt2 = reflexion_plan_prompt(task, self.dataset, reflection=reflection)
        raw2 = self._gen(prompt2)
        plan3 = self._parse(raw2) or plan2

        return {"pred": plan3, "prompt": prompt2, "raw_response": raw2,
                "reflection": reflection, "first_plan": plan, "retrieved_cases": []}

    @staticmethod
    def _split_reflection(text: str):
        m = re.search(r"IMPROVED PLAN\s*:", text, re.I)
        if m:
            from ..prompts import parse_numbered
            return text[:m.start()].strip(), parse_numbered(text[m.end():])
        return text.strip(), []


class ClinStyle(ReflexionStyle):
    """Offline CLIN: Reflexion + persistent cross-task lesson memory."""
    name = "CLIN-style"

    def __init__(self, llm, dataset: str, memory_size: int = 8):
        super().__init__(llm, dataset)
        self.lessons: List[str] = []
        self.memory_size = memory_size

    def query(self, task: str) -> Dict:
        lessons = "\n".join(f"- {l}" for l in self.lessons[-self.memory_size:])
        prompt1 = reflexion_plan_prompt(task, self.dataset, lessons=lessons)
        raw1 = self._gen(prompt1)
        plan = self._parse(raw1)

        reflect_raw = self._gen(reflexion_reflect_prompt(task, plan, self.dataset), max_tokens=4096)
        reflection, improved = self._split_reflection(reflect_raw)
        plan2 = improved if improved else plan

        # distill one lesson and store it for future tasks
        try:
            lesson = self._gen(clin_lesson_prompt(task, plan2, reflection), max_tokens=512).strip().split("\n")[0]
            if lesson:
                self.lessons.append(lesson[:200])
        except Exception:
            pass

        return {"pred": plan2, "prompt": prompt1, "raw_response": raw1,
                "reflection": reflection, "lessons_used": lessons, "retrieved_cases": []}


class ReflActStyle(BaseAgent):
    """Offline ReflAct: step-by-step self-imagined execution (no demos)."""
    name = "ReflAct-style"

    def __init__(self, llm, dataset: str, max_steps: int = 0):
        super().__init__(llm, dataset)
        self.max_steps = max_steps or (10 if dataset == "alfworld" else 15)

    def query(self, task: str) -> Dict:
        history: List[Dict] = []
        actions: List[str] = []
        raw_log = []
        for _ in range(self.max_steps):
            prompt = reflact_step_prompt(task, self.dataset, history, self.max_steps)
            raw = self._gen(prompt, max_tokens=1024)
            raw_log.append(raw)
            action = self._extract_action(raw)
            if not action or action.upper() == "FINISH":
                break
            actions.append(action)
            history.append({"step": len(history) + 1, "action": action,
                            "state": self._extract_state(raw)[:120]})
        return {"pred": actions, "prompt": reflact_step_prompt(task, self.dataset, history, self.max_steps),
                "raw_response": "\n---\n".join(raw_log), "retrieved_cases": []}

    @staticmethod
    def _extract_action(text: str) -> str:
        m = re.search(r"ACTION\s*:\s*(.+)", text, re.I)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_state(text: str) -> str:
        m = re.search(r"STATE\s*:\s*(.+)", text, re.I)
        return m.group(1).strip() if m else ""
