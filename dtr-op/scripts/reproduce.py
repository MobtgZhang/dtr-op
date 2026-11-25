"""Recompute the VLDB paper's empirical tables and figures offline."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_relevance import analyse, build_library, corpus_ceiling, family_heterogeneity, query_family
from check_icde_generation import reproduce_generation
from plot_paired_outcomes import DOMAINS, reproduce_paired_outcomes
from plot_curves import draw_budget, draw_coverage
from src.data.prepare import prepare


ROOT = Path(__file__).resolve().parents[1]


def verify_inputs():
    manifest = (ROOT / "SHA256SUMS").read_text(encoding="utf-8")
    for entry in manifest.splitlines():
        expected, name = entry.split("  ", 1)
        actual = sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}")


def split_accounting(domain, library, train, test):
    train_ids = {episode["task_id"] for episode in train}
    test_ids = {episode["task_id"] for episode in test}
    assert len(train_ids) == len(train) and len(test_ids) == len(test), domain
    assert train_ids.isdisjoint(test_ids), domain
    instructions = {episode["task"] for episode in train}
    action_lists = {tuple(episode["gold_actions"]) for episode in train}
    lengths = [len(episode["gold_actions"]) for episode in test]
    test_families = {episode.get("task_type") or episode.get("task_name")
                     or episode.get("group") or query_family(library, episode["task"], domain)
                     for episode in test}
    return {
        "repository_records": len(library.cases),
        "repository_families": len(library.prototypes),
        "test_episodes": len(test),
        "test_families": len(test_families),
        "median_reference_length": median(lengths),
        "mean_reference_length": round(mean(lengths), 1),
        "repeated_test_instructions": sum(episode["task"] in instructions for episode in test),
        "repeated_reference_lists": sum(tuple(episode["gold_actions"]) in action_lists
                                        for episode in test),
    }


def verify_relevance(computed):
    reference = json.loads((ROOT / "reference" / "relevance.json").read_text(encoding="utf-8"))
    for domain in DOMAINS:
        for rule, budgets in computed[domain].items():
            expected = reference[domain][rule]
            if rule == "corpus_ceiling":
                assert budgets == expected, (domain, rule)
                continue
            for budget, metrics in budgets.items():
                for metric, value in metrics.items():
                    if metric != "ms_per_query":
                        assert value == expected[budget][metric], (domain, rule, budget, metric)
    assert computed["heterogeneity"] == reference["heterogeneity"]


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    verify_inputs()

    splits, libraries, data_table = {}, {}, {}
    for domain in DOMAINS:
        splits[domain] = prepare(domain)
        libraries[domain] = build_library(domain, splits[domain])
        data_table[domain] = split_accounting(domain, libraries[domain][0], **splits[domain])
    archive = ROOT / "archive"
    generation = reproduce_generation(archive, libraries)

    relevance = {"heterogeneity": {}}
    for domain, (library, episodes) in libraries.items():
        relevance[domain] = analyse(domain, library, episodes)
        relevance[domain]["corpus_ceiling"] = corpus_ceiling(library, episodes)
        relevance["heterogeneity"][domain] = family_heterogeneity(library)
    verify_relevance(relevance)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paired = reproduce_paired_outcomes(archive, figures, relevance, libraries)
    draw_coverage(relevance, figures / "icde-retrieval.pdf")
    draw_budget(generation["domains"], figures / "icde-budget.pdf")
    write_json(output / "dataset.json", data_table)
    write_json(output / "generation.json", generation)
    write_json(output / "relevance.json", relevance)
    write_json(output / "paired.json", paired)
    print(f"Verified archived inputs and wrote results to {output}")


if __name__ == "__main__":
    main()
