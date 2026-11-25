# DTR-OP reproducibility artifact

This artifact accompanies *When Similar Tasks Need Different Plans: Diagnosing Trajectory Retrieval for Offline Planning*. It recomputes the experimental tables and figures from the fixed ALFWorld and ScienceWorld splits and the responses archived for the paper.

## Offline reproduction

Use CPython 3.14. The validated environment was Linux with CPython 3.14.6. From the artifact root, run the following commands.

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

After dependency installation, the run works offline and needs only a CPU. No simulator, model access, or API key is required. The full run took about two minutes on the validation machine. Run with Python assertions enabled. The script validates `SHA256SUMS`, checks the archive against the fixed test splits, and writes JSON and four PDFs into `outputs/`. Use `--output PATH` to choose a different directory.

## Paper-to-output mapping

- **Table 1** uses `dataset.json` for repository and test counts, reference lengths, and exact train/test overlaps in text and action lists.
- **Table 2 and Figure 3** use `relevance.json`. Table 2 uses key `"3"` for four returned cases. Figure 3 is redrawn as `figures/icde-retrieval.pdf` using the four access rules and the full-repository ceiling.
- **Figure 4** uses `paired.json` → `retrieval` and `figures/paired-retrieval.pdf`. On ScienceWorld, 143 episodes meet the threshold with both selectors, 29 with lexical only, 5 with deployed only, 15 with neither despite a qualifying stored case, and 8 with no qualifying stored case.
- **Tables 3 and 4** use `generation.json` → `domains` → `main` and `main_paired`, respectively. ScienceWorld procedure-minus-retrieval RC is +3.20 percentage points with a pointwise 95% interval of [+1.88, +4.59].
- **Figure 5** uses `paired.json` → `prompt` and `figures/paired-prompts.pdf`. ALFWorld has 53 LW-RC gains of at least 10 points. ScienceWorld has 137 Op-F1 gains, including 98 under 10 points.
- **Figure 6** uses `generation.json` → `domains` → `sweep` and `figures/icde-budget.pdf`. The horizontal axis is total demonstrations, `K + 1`. Each band's endpoints are the pointwise bootstrap limits for that budget. The sweep's `K=3` generations and the main four-case generations are distinct runs.
- **Table 5** uses `generation.json` → `domains` → `replacement_triggered` and `replacement_minus_first`. The trigger counts are 0/120 and 91/200, respectively.
- **Figures 1 and 2** are the paper's illustrative panel and architecture diagram, provided as `reference/icde-example.pdf` and `reference/icde-pipeline.pdf`.

## Source and inputs

- `src/case_library.py` implements the deployed retrieval operator. `src/agents/`, `src/prompts.py`, `src/canonicalize.py`, and `src/evaluation/metrics.py` contain the inference procedures, plan parser, normalization, and offline metrics from the experiment code.
- `data/processed/` contains the fixed train/test JSONL splits. ALFWorld uses the 3,553 processed training cases and 120 sampled `valid_unseen` cases (seed 42). ScienceWorld uses the 3,442 processed training cases and 200 sampled test cases (seed 43). The processed files are derived from the benchmarks described by Shridhar et al. (*ALFWorld*, ICLR 2021) and Wang et al. (*ScienceWorld*, EMNLP 2022).
- `archive/main/` contains five configurations per domain. The names `SYNAPSE-style`, `MultiCaseNoProto`, `StructuralAnalogicalCBR`, `DirectLLM`, and `Expel-style` correspond to the retrieval, procedure, augmented, direct, and rules rows. `archive/rq4/` contains the prototype-augmented runs for each `K` from 2 to 12. The compressed JSONL files preserve record order, case IDs, prompts, predictions, references, and available raw responses. Machine-specific ALFWorld `game_file` metadata was removed.
- `scripts/` contains the offline evaluation and plotting code adapted from the paper analysis scripts. Its retrieval diagnostics build each repository from the training split and use reference actions to score the returned cases. The script verifies the retrieved case order against the archived retrieval, procedure, augmented, and budget runs.
- `generation.json` includes per-run SHA-256 hashes of the **uncompressed bundled** JSONL, in addition to the `SHA256SUMS` manifest for the compressed inputs. In the event of duplicate records, usable non-error responses take precedence and the later archive line breaks ties.

The augmented and budget comparisons score the parsed first response, with ALFWorld canonicalization. Replay also reconstructs the historical prediction with the available replacement and checks it against the archived prediction. The remaining four configurations are rescored from their stored predictions. RC, LW-RC, and ordered operator F1 measure alignment with the reference plans. The 10,000-resample bootstrap conditions on the fixed generations and uses episode pairing for differences. The `ms_per_query` fields are local diagnostics and vary by machine.

## Optional live collection

`experiments/run_main.py` and `experiments/run_rq4.py` allow fresh requests with a compatible chat-completions service. The historical launch settings requested the endpoint label `claude-opus-5`. Endpoint availability and backend behavior can vary. The live augmented and budget runners preserve the historical length-triggered replacement policy. The one-call results in the paper select the first response during offline replay.

After installing `requirements-live.txt`, provide `CBR_API_BASE_URL` (the base before `/v1/chat/completions`) and `CBR_API_KEY` through your environment. Set `--model` to the model identifier on your endpoint. These commands make paid requests if your provider charges for inference.

```sh
python -m pip install -r requirements-live.txt
python experiments/run_main.py --dataset alfworld --method retrieval --model claude-opus-5
python experiments/run_rq4.py --dataset scienceworld --model claude-opus-5 --k-values 2,3,4,5,6,7,8,9,10,11,12
```

The live runner writes under `live-results/`. Use `--help` for the supported methods and options. The offline reproduction always reads the bundled archive.
