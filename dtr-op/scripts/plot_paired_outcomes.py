import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from check_icde_generation import MAIN_FILES, METRICS, case_ids


DOMAINS = {"alfworld": "ALFWorld", "scienceworld": "ScienceWorld"}
LOSS, TIE, GAIN = "#a8482a", "#d6dbe1", "#006c77"
LARGE_CHANGE = 0.10


def retrieval_outcomes(domain, expected, library, episodes):
    from analyze_relevance import (
        CONSISTENT, action_type, lcs_cov, query_family, rank_lexical,
        score_set, skeleton_of,
    )

    counts = dict(both=0, deployed_only=0, lexical_only=0,
                  missed_in_library=0, below_ceiling=0)
    scores = {"lexical": [], "hybrid": []}
    for episode in episodes:
        query, gold = episode["task"], episode["gold_actions"]
        family = query_family(library, query, domain)
        lexical = rank_lexical(library, query, 4)
        deployed = library.recall(query, top_k=3, use_prototype=True)
        assert gold and len(lexical) == len(deployed) == 4
        scores["lexical"].append(score_set(lexical, gold, family)["coverage"])
        scores["hybrid"].append(score_set(deployed, gold, family)["coverage"])
        outcome = {
            (True, True): "both", (False, True): "deployed_only",
            (True, False): "lexical_only", (False, False): "missed_in_library",
        }[scores["lexical"][-1] >= CONSISTENT, scores["hybrid"][-1] >= CONSISTENT]
        if outcome == "missed_in_library":
            reference = [action_type(action) for action in gold]
            if not any(lcs_cov(skeleton_of(case), reference) >= CONSISTENT for case in library.cases):
                outcome = "below_ceiling"
        counts[outcome] += 1

    for rule, coverage in scores.items():
        published = expected[rule]["3"]
        values = np.array(coverage)
        assert len(values) == published["n"]
        assert round(100 * float(values.mean()), 1) == published["coverage_pct"]
        assert round(100 * float((values >= CONSISTENT).mean()), 1) == published["frac_ge_80_pct"]
    missing_share = round(100 * counts["below_ceiling"] / len(episodes), 1)
    assert missing_share == expected["corpus_ceiling"]["frac_no_match_pct"]
    changes = np.array(scores["hybrid"]) - np.array(scores["lexical"])
    tied = np.isclose(changes, 0, rtol=0, atol=1e-12)
    directions = {
        "lower": int(((changes < 0) & ~tied).sum()),
        "equal": int(tied.sum()),
        "higher": int(((changes > 0) & ~tied).sum()),
    }
    return counts, directions


def prompt_outcomes(domain, archive):
    from analyze_structure import ordered_operator_f1
    from audit_results import dedupe_by_task, load_jsonl
    from final_system import _score_stored

    rows = {}
    for name in ("retrieval", "procedure"):
        file = archive / "main" / domain / f"{MAIN_FILES[name]}.jsonl"
        rows[name] = dedupe_by_task(load_jsonl(file))
        assert all(not row.get("error") for row in rows[name])
    shared_cases = case_ids(rows["retrieval"])
    assert shared_cases == case_ids(rows["procedure"])
    assert all(len(ids) == 3 and all(ids) for ids in shared_cases.values())

    scored = {}
    for name, archive_rows in rows.items():
        scores = _score_stored(domain, archive_rows)
        for score in scores:
            score["opf"] = ordered_operator_f1(score["pred"], score["gold"])
        scored[name] = {score["task_id"]: score for score in scores}
    retrieval, procedure = scored["retrieval"], scored["procedure"]
    assert retrieval.keys() == procedure.keys()
    assert all(retrieval[task]["gold"] == procedure[task]["gold"] for task in retrieval)

    deltas = {
        metric: np.array([procedure[task][key] - retrieval[task][key] for task in retrieval])
        for metric, key in METRICS.items()
    }
    outcomes = {metric: bin_prompt_changes(changes) for metric, changes in deltas.items()}
    stable_rc = np.isclose(deltas["RC"], 0, rtol=0, atol=1e-12)
    return outcomes, bin_prompt_changes(deltas["LW-RC"][stable_rc])


def bin_prompt_changes(deltas):
    tied = np.isclose(deltas, 0, rtol=0, atol=1e-12)
    large = np.abs(deltas) >= LARGE_CHANGE - 1e-12
    lower = (deltas < 0) & ~tied
    higher = (deltas > 0) & ~tied
    return {
        "large_loss": int((lower & large).sum()),
        "small_loss": int((lower & ~large).sum()),
        "equal": int(tied.sum()),
        "small_gain": int((higher & ~large).sum()),
        "large_gain": int((higher & large).sum()),
    }


def stacked_shares(axis, rows, keys, colors, labels, text_colors, min_label_width):
    left = np.zeros(len(rows))
    for key, color, label, text_color in zip(keys, colors, labels, text_colors):
        widths = np.array([100 * row[key] / sum(row[item] for item in keys) for row in rows])
        axis.barh(range(len(rows)), widths, left=left, height=0.58, color=color,
                  edgecolor="white", linewidth=0.6, label=label, zorder=3)
        for index, (start, width, row) in enumerate(zip(left, widths, rows)):
            if width >= min_label_width:
                axis.text(start + width / 2, index, str(row[key]), color=text_color,
                          ha="center", va="center", fontsize=8, zorder=4)
        left += widths
    axis.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100])
    axis.tick_params(axis="both", length=0, pad=3)
    axis.spines[:].set_visible(False)
    axis.grid(axis="x", color="#d6dbe1", linewidth=0.5)
    axis.invert_yaxis()


def draw_retrieval(outcomes, output):
    figure, axis = plt.subplots(figsize=(3.35, 1.80))
    stacked_shares(axis, list(outcomes.values()),
                   ("lexical_only", "deployed_only", "both", "missed_in_library", "below_ceiling"),
                   (LOSS, GAIN, "#6c7f99", "#dce0e4", "#3c444c"),
                   ("Lexical only", "Deployed only", "Both",
                    "Neither (case exists)", "Neither (no case)"),
                   ("white", "white", "white", "#20262c", "white"), 3.8)
    axis.set_yticks(range(len(DOMAINS)), list(DOMAINS.values()))
    axis.set_xlabel("Share of episodes (%)", labelpad=3)
    handles, labels = axis.get_legend_handles_labels()
    for indices, height in (((0, 1, 2), 0.97), ((3, 4), 0.86)):
        figure.legend([handles[index] for index in indices], [labels[index] for index in indices],
                      loc="upper center", bbox_to_anchor=(0.52, height), ncol=len(indices),
                      frameon=False, handlelength=0.9, columnspacing=1.1)
    figure.subplots_adjust(left=0.24, right=0.96, top=0.66, bottom=0.23)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def draw_prompts(outcomes, output):
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(3.35, 2.66))
    threshold = f"{100 * LARGE_CHANGE:g} pp"
    for axis, (domain, metrics) in zip(axes, outcomes.items()):
        stacked_shares(axis, list(metrics.values()),
                       ("large_loss", "small_loss", "equal", "small_gain", "large_gain"),
                       (LOSS, "#e8ae94", TIE, "#9bd0d4", GAIN),
                       (f"Loss ≥{threshold}", f"Loss <{threshold}", "Unchanged",
                        f"Gain <{threshold}", f"Gain ≥{threshold}"),
                       ("white", "#20262c", "#20262c", "#20262c", "white"), 6)
        axis.set_yticks(range(len(metrics)), list(metrics))
        size = sum(next(iter(metrics.values())).values())
        axis.set_title(f"{DOMAINS[domain]} ($n$ = {size})", loc="left", pad=3)
    axes[-1].set_xlabel("Share of episodes (%)", labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    for indices, height in (((0, 1, 2), 0.98), ((3, 4), 0.895)):
        figure.legend([handles[index] for index in indices], [labels[index] for index in indices],
                      loc="upper center", bbox_to_anchor=(0.54, height), ncol=len(indices),
                      frameon=False, handlelength=0.9, columnspacing=1.1)
    figure.subplots_adjust(left=0.14, right=0.96, top=0.76, bottom=0.15, hspace=0.38)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def reproduce_paired_outcomes(archive, out, published, libraries):
    retrieval, retrieval_directions, prompts, stable_rc_lw = {}, {}, {}, {}
    for domain in DOMAINS:
        retrieval[domain], retrieval_directions[domain] = retrieval_outcomes(
            domain, published[domain], *libraries[domain])
        prompts[domain], stable_rc_lw[domain] = prompt_outcomes(domain, archive)
    for domain in DOMAINS:
        for metric in prompts[domain].values():
            assert sum(metric.values()) == sum(retrieval[domain].values())
        assert sum(stable_rc_lw[domain].values()) == prompts[domain]["RC"]["equal"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})
    draw_retrieval(retrieval, out / "paired-retrieval.pdf")
    draw_prompts(prompts, out / "paired-prompts.pdf")
    return {"retrieval": retrieval, "retrieval_directions": retrieval_directions,
            "prompt": prompts, "lw_rc_with_unchanged_rc": stable_rc_lw}
