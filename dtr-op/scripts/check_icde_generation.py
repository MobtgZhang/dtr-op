import gzip
from hashlib import sha256
from pathlib import Path


MAIN_FILES = {
    "retrieval": "SYNAPSE-style",
    "procedure": "MultiCaseNoProto",
    "augmented": "StructuralAnalogicalCBR",
    "direct": "DirectLLM",
    "rules": "Expel-style",
}
METRICS = {"RC": "gc", "LW-RC": "plw_gc", "Op-F1": "opf"}


def case_ids(records):
    return {
        record["task_id"]: tuple(
            case.get("case_id", case.get("id")) for case in record["retrieved_cases"]
        )
        for record in records
    }


def reproduce_generation(archive, libraries):
    from analyze_replay import replay
    from analyze_structure import ordered_operator_f1
    from audit_results import dedupe_by_task, load_jsonl
    from final_system import _score_stored, paired, score, summarize

    source_hashes = {}
    results = {}

    def load(relative_path):
        path = archive / relative_path
        with gzip.open(path.with_suffix(".jsonl.gz"), "rb") as stream:
            source_hashes[relative_path.as_posix()] = sha256(stream.read()).hexdigest()
        records = dedupe_by_task(load_jsonl(path))
        assert all(not record.get("error") and (record.get("raw_response") or record.get("pred"))
                   for record in records), path
        return records

    def add_operator_scores(rows):
        for row in rows:
            row["opf"] = ordered_operator_f1(row["pred"], row["gold"])
        return rows

    def check_retrieval(records, library, episodes, top_k, prototype):
        expected = {
            episode["task_id"]: tuple(case["id"] for case in library.recall(
                episode["task"], top_k=top_k, use_prototype=prototype))
            for episode in episodes
        }
        assert case_ids(records) == expected, (library.dataset, top_k, prototype)

    for domain, expected_count in (("alfworld", 120), ("scienceworld", 200)):
        library, episodes = libraries[domain]
        main_path = Path("main") / domain
        runs = {name: load(main_path / f"{filename}.jsonl") for name, filename in MAIN_FILES.items()}
        gold = {record["task_id"]: record["gold"] for record in runs["augmented"]}
        assert gold == {episode["task_id"]: episode["gold_actions"] for episode in episodes}, domain
        same_cases = case_ids(runs["retrieval"])
        assert same_cases == case_ids(runs["procedure"]), domain
        assert all(len(ids) == 3 and all(ids) for ids in same_cases.values()), domain
        assert all(len(ids) == 4 and len(set(ids)) == 4 and all(ids)
                   for ids in case_ids(runs["augmented"]).values()), domain
        check_retrieval(runs["retrieval"], library, episodes, top_k=3, prototype=False)
        check_retrieval(runs["augmented"], library, episodes, top_k=3, prototype=True)

        sweep_path = Path("rq4") / domain / "claude_opus_5"
        sweep_runs = {count: load(sweep_path / f"SA_CBR_K{count}.jsonl") for count in range(2, 13)}
        for count, records in sweep_runs.items():
            check_retrieval(records, library, episodes, top_k=count, prototype=True)
        for records in [*runs.values(), *sweep_runs.values()]:
            assert len(records) == expected_count
            assert {record["task_id"]: record["gold"] for record in records} == gold
        for records in [runs["augmented"], *sweep_runs.values()]:
            assert all(replay(record, domain)["pred_replay"] == record["pred"]
                       for record in records), domain
            _score_stored(domain, records)

        scored = {name: add_operator_scores(score(domain, records) if name == "augmented"
                                           else _score_stored(domain, records))
                  for name, records in runs.items()}
        sweep = {count: score(domain, records) for count, records in sweep_runs.items()}
        replacement = add_operator_scores(score(domain, runs["augmented"], disable_refinement=False))
        results[domain] = {
            "episodes": expected_count,
            "main": {
                name: {metric: round(100 * sum(row[key] for row in rows) / expected_count, 2)
                       for metric, key in METRICS.items()}
                for name, rows in scored.items()
            },
            "main_paired": {
                f"{end}_minus_{start}": {metric: paired(scored[end], scored[start], key)
                                          for metric, key in METRICS.items()}
                for end, start in (("procedure", "retrieval"), ("augmented", "procedure"),
                                   ("augmented", "retrieval"))
            },
            "sweep": {count: summarize(domain, rows) for count, rows in sweep.items()},
            "sweep_7_minus_3": {metric: paired(sweep[7], sweep[3], key)
                                for metric, key in METRICS.items() if key != "opf"},
            "replacement_triggered": sum(bool((record.get("extra") or {}).get("refine_response"))
                                         for record in runs["augmented"]),
            "replacement_minus_first": {metric: paired(replacement, scored["augmented"], key)
                                        for metric, key in METRICS.items()},
        }

    return {"domains": results, "source_sha256": source_hashes}
