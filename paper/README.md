# CausalRAG paper drafts — version 1

This directory contains two synchronized scientific-paper drafts about multi-hop question answering with a **legal causal graph** and structural counterfactual verification:

- `en/main.tex`: English version.
- `vi/main.tex`: Vietnamese version.
- `references.bib`: shared bibliography; do not duplicate references inside a language directory.

The papers use a generic `article` layout because no target venue has been selected. They are deliberately framed as a version-1 research draft rather than a camera-ready paper.

## Build

XeLaTeX is required for Unicode Vietnamese. From the repository root, run in PowerShell:

```powershell
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -cd paper/en/main.tex
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -cd paper/vi/main.tex
```

Expected PDFs:

- `paper/en/main.pdf`
- `paper/vi/main.pdf`

If `latexmk` is unavailable but `xelatex` and `bibtex` are installed, compile each language from its own directory:

```powershell
Set-Location paper/en
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

Set-Location ../vi
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

No TeX packages are installed automatically by this repository.

## Synchronization contract

The English and Vietnamese drafts must keep the same:

1. section and subsection order;
2. equation, table, and appendix labels;
3. numerical values and experiment-version qualifiers;
4. citation keys and shared bibliography;
5. limitations, threats to validity, and visible TODO items.

Translation may change sentence structure, but it must not strengthen or weaken a scientific claim. Update both language files in the same change.

## Result provenance and interpretation

The numerical tables currently summarize the **preliminary Step 4 v4.0 artifacts**:

- `../evaluation_metrics_structural_scm/evaluation_metrics_report.json`
- `../evaluation_metrics_path_ablation/evaluation_metrics_report.json`
- `../data/pipeline_predictions_structural_scm.json`
- `../data/pipeline_predictions_path_ablation.json`
- `../data/pipeline_run_log_structural_scm.json`
- `../data/pipeline_run_log_path_ablation.json`

The current implementation is `STEP4_VERSION = "4.1-query-aware-legal-scm"`, but a complete paired v4.1 benchmark rerun has not yet been imported. Consequently:

- every current result table is explicitly labeled **preliminary v4.0**;
- projected v4.1 accuracy is not reported as an experimental result;
- the structural and path-ablation modes are reported as having identical Step 7 quality metrics, not as evidence of structural-mode superiority;
- the observed runtime overhead is reported;
- the 20 samples marked `requires_counterfactual=true` are described as direct-edge negatives, not as gold `do(M=false)` interventions;
- `path_ablation` is called a bounded node-deletion reachability baseline, never a valid implementation of Pearl's do-operator.

An auxiliary audit of the existing artifacts counted 8,882 structurally evaluated paths, zero structural fallback paths, and 3,393 emitted mediator interventions, all labeled `NECESSARY`. These counters are not currently emitted by Step 7 itself; they therefore remain diagnostic audit counts rather than official benchmark metrics.

## Mandatory TODOs before submission

- Replace author, affiliation, contact, funding, and venue placeholders.
- Rerun both modes end-to-end with Step 4 v4.1 and update every synchronized table and discussion paragraph.
- Pin the code commit and hashes of benchmark, rules, graph, memory, and predictions.
- Add a paired per-sample mode-comparison report and confidence intervals/significance tests where meaningful.
- Build a balanced structural-counterfactual benchmark containing `NECESSARY`, `NON_NECESSARY`, and `INDETERMINATE` cases with explicit factual context, intervention, and world-state gold labels.
- Obtain independent expert review of the Vietnamese legal rules, questions, answers, and citations.
- Select a venue and migrate both drafts to the same venue template only after the venue is known.

## Scope statement

The implemented `LegalSCM` is a deterministic, normative rule model over encoded legal events. The paper does **not** claim causal discovery, observational causal-effect estimation, calibrated probabilities, legal validity of every extracted rule, or a complete abduction--action--prediction model of real-world causation.
