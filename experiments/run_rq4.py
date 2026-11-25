#!/usr/bin/env python3
"""RQ4: Multi-case effectiveness analysis.

Tests SA-CBR with K=1,2,3,4,5 to demonstrate the effectiveness of multi-case 
analogical reasoning. Runs on both Claude and DeepSeek models.
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import build_library, make_llm, offline_evaluator
from src.config import RESULTS_DIR
from src.data.prepare import prepare_alfworld, prepare_scienceworld
from src.evaluation.runner import run_agent_over

RESULTS = RESULTS_DIR / "rq4"


def run_rq4(dataset: str, model: str, k_values: list = [1, 2, 3, 4, 5], workers: int = 8):
    """Run SA-CBR with different K values."""
    # Load data
    if dataset == "alfworld":
        data = prepare_alfworld()
    elif dataset == "scienceworld":
        data = prepare_scienceworld()
    else:
        raise ValueError(f"Dataset must be alfworld or scienceworld for RQ4")
    
    # Create results directory
    model_safe = model.replace("/", "_").replace("-", "_")
    out_dir = RESULTS / dataset / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    
    llm = make_llm(model)
    library = build_library(dataset, data["train"])
    evaluator = offline_evaluator(dataset)
    
    summaries = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Running SA-CBR with K={k} on {dataset} using {model}")
        print(f"{'='*60}\n")
        
        # Import here to avoid circular dependency
        from src.agents.ours import StructuralAnalogicalCBR
        
        # Create agent with specific K
        agent = StructuralAnalogicalCBR(llm, dataset, top_k=k)
        agent.bootstrap(library)
        agent.name = f"SA-CBR_K={k}"  # Override name for clarity
        
        # Run evaluation
        out_file = out_dir / f"SA_CBR_K{k}.jsonl"
        agg = run_agent_over(
            agent, 
            data["test"], 
            evaluator, 
            out_file,
            label=f"{dataset}/{model}/SA-CBR(K={k})", 
            max_workers=workers
        )
        
        # Add metadata
        agg["model"] = model
        agg["dataset"] = dataset
        agg["k"] = k
        agg["n_train_cases"] = len(library.cases)
        
        summaries.append(agg)
        
        # Save incremental results
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        
        print(f"\n==> K={k} Results: SR={agg['sr']:.2f}%, PLWSR={agg['plwsr']:.2f}%, "
              f"GC={agg['gc']:.2f}%, PLWGC={agg['plw_gc']:.2f}%\n", flush=True)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"RQ4 Summary: {dataset} / {model}")
    print(f"{'='*60}")
    print(f"{'K':<5} {'SR%':<8} {'PLWSR%':<10} {'GC%':<8} {'PLWGC%':<10}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['k']:<5} {s['sr']:<8.2f} {s['plwsr']:<10.2f} {s['gc']:<8.2f} {s['plw_gc']:<10.2f}")
    print("=" * 60)
    
    return summaries


def main():
    ap = argparse.ArgumentParser(description="RQ4: Test SA-CBR with different K values")
    ap.add_argument("--dataset", choices=["alfworld", "scienceworld"], required=True)
    ap.add_argument("--model", required=True, help="Model name (e.g., claude-opus-5, deepseek-v4-flash)")
    ap.add_argument("--k-values", default="1,2,3,4,5", help="Comma-separated K values to test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(",")]
    run_rq4(args.dataset, args.model, k_values=k_values, workers=args.workers)


if __name__ == "__main__":
    main()
