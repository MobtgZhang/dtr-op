#!/usr/bin/env python3
"""Optionally collect new responses for the main configurations in Table 3."""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import build_library, make_llm, offline_evaluator
from src.agents.ablations import DirectLLM, MultiCaseNoProto
from src.agents.baselines import ExpelStyle, SynapseStyle
from src.agents.ours import StructuralAnalogicalCBR
from src.config import RESULTS_DIR
from src.data.prepare import prepare
from src.evaluation.runner import run_agent_over

RESULTS = RESULTS_DIR / "main"


def run(dataset, method, workers, model):
    data = prepare(dataset)
    llm = make_llm(model)
    library = build_library(dataset, data["train"])
    methods = {
        "retrieval": SynapseStyle(llm, dataset, top_k=3),
        "procedure": MultiCaseNoProto(llm, dataset, top_k=3),
        "augmented": StructuralAnalogicalCBR(llm, dataset, top_k=3),
        "direct": DirectLLM(llm, dataset),
        "rules": ExpelStyle(llm, dataset),
    }
    agent = methods[method]
    agent.bootstrap(library)

    evaluator = offline_evaluator(dataset)
    
    # Create model-specific output directory if model is specified
    if model:
        model_safe = model.replace("/", "_").replace("-", "_")
        out_dir = RESULTS / dataset / model_safe
    else:
        out_dir = RESULTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"{agent.name.replace('/', '_').replace(' ', '_')}.jsonl"
    summary = run_agent_over(agent, data["test"], evaluator, out_file,
                             label=f"{dataset}/{agent.name}", max_workers=workers)
    summary["model"] = llm.model
    summary["n_train_cases"] = len(library.cases)
    with (out_dir / f"summary_{method}.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["alfworld", "scienceworld"], required=True)
    ap.add_argument("--method", choices=["retrieval", "procedure", "augmented", "direct", "rules"],
                    required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-opus-5")
    args = ap.parse_args()
    run(args.dataset, args.method, args.workers, args.model)


if __name__ == "__main__":
    main()
