"""Plot the retrieval coverage and generation budget curves."""

import matplotlib.pyplot as plt

from plot_paired_outcomes import DOMAINS


def draw_coverage(relevance, output):
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharey=True, layout="constrained")
    rules = {
        "lexical": ("Lexical", "o", "#0072b2"),
        "hybrid": ("Deployed", "s", "#d55e00"),
        "structural": ("Prototype alignment", "^", "#009e73"),
        "family_lexical": ("Family-restricted", "D", "#755498"),
    }
    for axis, (domain, title) in zip(axes, DOMAINS.items()):
        data = relevance[domain]
        for rule, (label, marker, color) in rules.items():
            rows = sorted(data[rule].values(), key=lambda row: row["n_cases"])
            sizes = [row["n_cases"] for row in rows]
            axis.plot(sizes, [row["coverage_pct"] for row in rows],
                      marker=marker, color=color, label=label, markersize=3)
        axis.axhline(data["corpus_ceiling"]["mean_best_coverage_pct"], linestyle=":",
                     color="#414141", label="Full-repository ceiling")
        axis.set(title=title, xlabel="Returned cases", xticks=sizes, ylim=(60, 102))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean skeleton coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(output)
    plt.close(figure)


def draw_budget(generation, output):
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 3.6), sharex=True, layout="constrained")
    for axis, (domain, title) in zip(axes, DOMAINS.items()):
        sweep = generation[domain]["sweep"]
        budgets = sorted(sweep)
        sizes = [budget + 1 for budget in budgets]
        for key, label, color, marker in (("gc", "RC", "#0072b2", "o"),
                                         ("plwgc", "LW-RC", "#d55e00", "s")):
            scores = [sweep[budget][key] for budget in budgets]
            lower = [sweep[budget][f"{key}_ci"][0] for budget in budgets]
            upper = [sweep[budget][f"{key}_ci"][1] for budget in budgets]
            axis.plot(sizes, scores, color=color, marker=marker, markersize=2.5, label=label)
            axis.fill_between(sizes, lower, upper, color=color, alpha=0.17, linewidth=0)
        axis.set(title=title, ylabel="Score (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set(xlabel="Total demonstrations (K + 1)", xticks=sizes)
    figure.savefig(output)
    plt.close(figure)
