"""Prompt templates for every method. All methods share the same backbone LLM."""
import re
from typing import Dict, List

ALF_GRAMMAR = """ALFWorld action grammar (use EXACTLY these forms):
- go to <recep>
- take <object> from <recep>
- move <object> to <recep>          (NOT "put" / "place")
- open <recep> / close <recep>
- use <lamp>
- heat <object> with <microwave>
- cool <object> with <fridge>
- clean <object> with <sinkbasin>
- examine <thing>"""

SW_GRAMMAR = """ScienceWorld commands are free-form verb phrases, e.g.:
- look around / go to <location> / open door to <room>
- pick up <object> / move <object> to <container>
- pour <substance> into <container> / connect <object> to <object>
- activate <object> / deactivate <object> / use <object> on <object>
- focus on <object> / examine <object> / wait1"""

WEBSHOP_GRAMMAR = """WebShop action grammar (use EXACTLY these forms):
- search[query] - Search for products (e.g., search[wireless headphones])
- click[product_id] - Click on a product ID (10-char code starting with 'b0')
- click[option] - Select product options like color, size (e.g., click[blue])
- click[< prev] - Go back to previous page
- click[back to search] - Return to search results
- click[buy now] - Complete the purchase

Shopping flow: search → click product → click options → click[buy now]"""


def format_demos(cases: List[Dict], mark_prototype_id: str = "") -> str:
    blocks = []
    for i, c in enumerate(cases, 1):
        tag = " (prototype)" if (mark_prototype_id and c.get("id") == mark_prototype_id) else ""
        lines = "\n".join(f"  {j+1}. {a}" for j, a in enumerate(c["actions"]))
        blocks.append(f"Example {i}{tag}:\nTask: {c['task']}\nActions:\n{lines}")
    return "\n\n".join(blocks)


def parse_numbered(text: str) -> List[str]:
    actions = []
    for line in text.split("\n"):
        m = re.match(r"^\s*\d+\s*[\.\):\-]\s*(.+)", line)
        if m:
            a = m.group(1).strip().strip('"').strip("'").strip()
            if a and len(a) > 2:
                actions.append(a)
    return actions


# ---------------------------------------------------------------------------
def ours_prompt(query: str, cases: List[Dict], dataset: str, expected_len: int,
                prototype_id: str = "") -> str:
    demos = format_demos(cases, prototype_id)
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    if dataset == "alfworld":
        extra = ("- Mirror the structure of the LONGEST demonstration (it fixes the step count).\n"
                 "- Keep intermediate state-change steps (heat/cool/clean) when the demonstrations contain them.\n"
                 "- Use short entity names like \"fridge 1\" or \"tomato 2\"; never coordinate suffixes.\n"
                 "- Include \"open <recep>\" before placing into a closed container only if the demonstrations do.")
    else:
        extra = ("- Mirror the step pattern of the LONGEST demonstration and substitute the entities of the new task.\n"
                 "- Use exactly the same verb phrases the demonstrations use.\n"
                 "- Keep every intermediate step the demonstrations show (do not skip measurements, activations, etc.).")
    return f"""You solve tasks by STRUCTURAL ANALOGY: you mirror the relational step pattern of past cases and substitute the entities of the new task.

DEMONSTRATIONS (similar tasks with their correct solutions):
{demos}

NEW TASK: {query}
The plan should have about {expected_len} actions (the average length of the demonstrations).

{grammar}

Rules:
{extra}

Output ONLY the numbered action list, starting with:
1."""


def refine_prompt(query: str, plan: List[str], cases: List[Dict], expected_len: int,
                  dataset: str) -> str:
    demos = format_demos(cases)
    plan_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(plan))
    return f"""Your previous plan has the wrong number of actions ({len(plan)}); the demonstrations suggest about {expected_len} actions.

TASK: {query}

DEMONSTRATIONS:
{demos}

YOUR PREVIOUS PLAN:
{plan_str}

Regenerate the plan with exactly {expected_len} actions, mirroring the demonstration structure. Output ONLY the numbered action list:
1."""


def direct_prompt(query: str, dataset: str) -> str:
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    return f"""Generate an action plan for the following task.

Task: {query}

{grammar}

Output ONLY the numbered action list:
1."""


def single_case_prompt(query: str, case: Dict, dataset: str) -> str:
    lines = "\n".join(f"  {j+1}. {a}" for j, a in enumerate(case["actions"]))
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    return f"""Solve the new task by ANALOGY to this single similar case: copy its step pattern and substitute the new task's entities.

CASE:
Task: {case['task']}
Actions:
{lines}

NEW TASK: {query}

{grammar}

Output ONLY the numbered action list:
1."""


def synapse_prompt(query: str, cases: List[Dict], dataset: str) -> str:
    demos = format_demos(cases)
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    return f"""Below are similar tasks and their solutions. Use them as references to solve the new task.

{demos}

NEW TASK: {query}

{grammar}

Output ONLY the numbered action list:
1."""


def expel_rules_prompt(task_type: str, demos: List[Dict], dataset: str) -> str:
    blocks = format_demos(demos)
    return f"""You are distilling reusable planning rules from demonstrations of the task type "{task_type}".

DEMONSTRATIONS:
{blocks}

Write 3-6 concise, general rules (one per line, starting with "-") that capture the common step structure and important cautions for solving ANY task of this type. Focus on structure (which steps, in which order), not surface words."""


def expel_plan_prompt(query: str, rules: str, case: Dict, dataset: str) -> str:
    lines = "\n".join(f"  {j+1}. {a}" for j, a in enumerate(case["actions"]))
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    return f"""Solve the task using the distilled rules and one concrete demonstration.

RULES for this task type:
{rules}

DEMONSTRATION:
Task: {case['task']}
Actions:
{lines}

NEW TASK: {query}

{grammar}

Output ONLY the numbered action list:
1."""


def reflexion_plan_prompt(query: str, dataset: str, reflection: str = "", lessons: str = "") -> str:
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    extra = ""
    if lessons:
        extra += f"\nLESSONS LEARNED FROM PREVIOUS TASKS (apply them):\n{lessons}\n"
    if reflection:
        extra += f"\nSELF-REFLECTION ON YOUR PREVIOUS ATTEMPT (fix all issues it raises):\n{reflection}\n"
    return f"""Generate an action plan for the following task.

Task: {query}
{extra}
{grammar}

Output ONLY the numbered action list:
1."""


def reflexion_reflect_prompt(query: str, plan: List[str], dataset: str) -> str:
    plan_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(plan))
    return f"""Task: {query}

Candidate plan:
{plan_str}

Critique this plan: identify missing steps, wrong order, wrong action grammar, or entity mistakes. Be brief (max 5 bullet points). Then output an IMPROVED numbered action list under a line "IMPROVED PLAN:".

Critique:"""


def clin_lesson_prompt(query: str, plan: List[str], reflection: str) -> str:
    plan_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(plan[:12]))
    return f"""From this experience, write ONE short general lesson (max 25 words) for solving similar tasks in the future.

Task: {query}
Plan tried:
{plan_str}
Reflection: {reflection[:600]}

Lesson:"""


def reflact_step_prompt(query: str, dataset: str, history: List[Dict], max_steps: int) -> str:
    grammar = ALF_GRAMMAR if dataset == "alfworld" else (WEBSHOP_GRAMMAR if dataset == "webshop" else SW_GRAMMAR)
    if history:
        traj = "\n".join(f"  step {h['step']}: action: {h['action']} | predicted state: {h['state']}" for h in history)
    else:
        traj = "  (none yet)"
    return f"""You are solving a task step by step (ReAct-style). At each step predict ONE action and the state you expect after it.

Task: {query}
{grammar}

Trajectory so far:
{traj}

Step {len(history)+1} of at most {max_steps}. Reply with EXACTLY two lines:
ACTION: <one action>
STATE: <short predicted state after the action>
If the task is complete, reply with:
ACTION: FINISH"""
