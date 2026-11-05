# COMP8240 – Replicating *GoEmotions* and testing it on new data

Final project for **COMP8240 Applications of Data Science** (Macquarie University), by Zannatul Ferdous.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zannatulerin/comp8240-goemotions-replication/blob/main/notebooks/comp8240_goemotions_replication_v2.ipynb)

This repository replicates the evaluation in **Demszky et al. (2020), *GoEmotions: A Dataset of Fine-Grained Emotions*** (ACL 2020), applies the resulting emotion classifier to existing and newly constructed datasets, and uses an LLM as a judge to examine how well the 27-emotion taxonomy generalises beyond Reddit comments.

* Paper: <https://aclanthology.org/2020.acl-main.372>
* Dataset (authors' release): <https://github.com/google-research/google-research/tree/master/goemotions>
* Dataset (Hugging Face): <https://huggingface.co/datasets/google-research-datasets/go_emotions>

> **Status note:** items marked **TBD** are filled in as the corresponding notebook sections are completed.

## What was done

| Stage | What | Notebook section |
|---|---|---|
| Replication (analysis) | Ran the authors' `analyze_data.py` and `ppca.py` on the full dataset (58k comments, 82 raters). `ppca.py` fails on current library versions; five documented patches fix it (see `docs/ppca_fixes.md`). | 1 |
| Replication (classifier) | Evaluated a public RoBERTa checkpoint on the full GoEmotions test set (5,427 examples), macro-F1 at thresholds 0.3 and 0.5, plus per-emotion F1. | 2 |
| Existing datasets | Proxy validation of the classifier on Amazon Fine Food reviews, Coronavirus tweets, Twitter US Airline Sentiment, Customer Support on Twitter and Yelp reviews. | 3 |
| New data | 150 Reddit r/AmItheAsshole posts (Oct–Dec 2023) sampled from a public corpus; **100 hand-labelled** with the GoEmotions taxonomy. | 4–5 |
| LLM-as-judge | An LLM labels the same posts independently with the same guideline; agreement with my labels and the classifier is measured with Cohen's and Fleiss' kappa. An optional second step has the LLM judge the classifier's own predictions. | 6–9 |

## Repository layout

```
notebooks/   comp8240_goemotions_replication_v2.ipynb   main notebook (saved with outputs)
data/        labelled_master.csv                        100 hand-labelled posts + LLM + classifier labels
             judge_raw/                                 verbatim LLM-judge replies (model + date in each file)
results/     per_emotion_f1_full_test.csv               per-emotion P/R/F1 at thresholds 0.3 and 0.5
             agreement_*.csv, label_distribution.csv    kappa tables and agreement breakdowns
             existing_datasets_summary.csv              proxy-validation summary for the five existing datasets
             ppca_output.txt, ppca_emotion_agreements.csv, construction_log.json, contested_items.csv
prompts/     llm_judge_prompt.txt                       exact annotation/judge guideline given to the LLM
docs/        ppca_fixes.md                              what broke in ppca.py and how it was fixed
```

The authors' own code is **not copied** here. The notebook clones it (sparse checkout of `google-research/google-research`, folder `goemotions`) and patches it at run time.

## How to run

1. Click the **Open in Colab** badge above (a GPU runtime is recommended, e.g. T4).
2. Add Colab **Secrets** (key icon): `KAGGLE_API_TOKEN` (Kaggle datasets), optionally `GEMINI_API_KEY` (automatic LLM judge) and `GH_TOKEN` (only if you want to push results). Never paste tokens into cells.
3. Edit `GH_USER` / `GH_REPO` in the config cell if you fork the repo.
4. **Runtime → Run all.** Every section is resumable: the master label file is restored from Google Drive or from `data/labelled_master.csv` in this repo, so no labelling is repeated. The hand-labelling cell only prompts if fewer than 100 posts are labelled.
5. The LLM-judge step (Section 6) is semi-manual by design: the notebook prints the posts, a chat model labels them, and the pasted reply is imported and saved verbatim in `data/judge_raw/`. An optional cell automates this with the Gemini API.

**Environment:** Google Colab, Python 3.13; `transformers`, `datasets`, `torch`, `scikit-learn`, `scipy`, `statsmodels`, `pandas`, `seaborn` (see `requirements.txt` for the versions used). **Classifier:** [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions).

## Key results

**Replication on the full test set (5,427 examples)**

| Setting | Macro-F1 (27 emotions) | Micro-F1 | Paper (BERT) |
|---|---|---|---|
| Threshold 0.3 (default in the authors' `calculate_metrics.py`) | **0.4916** | 0.6086 | ≈ 0.46 |
| Threshold 0.5 | 0.4431 | 0.5855 | ≈ 0.46 |

* Best classes: gratitude (F1 0.918), amusement (0.826), love (0.810). Worst: relief and grief (0.000 at both thresholds; 11 and 6 test examples).
* The result is close to the paper but not identical: a different (RoBERTa) model replaces the authors' TensorFlow-1 BERT, which cannot run on current Colab; the checkpoint is third-party; sequence length was capped at 128 tokens.
* PPCA: **TBD** (compare the number of significant components with the paper).

**Existing datasets (positive-vs-negative emotion mass against each dataset's own labels):** **TBD** – see `results/existing_datasets_summary.csv`.

**New Reddit data (n = 100):** **TBD** – Cohen's κ (manual vs LLM, manual vs classifier, LLM vs classifier), Fleiss' κ and the sentiment-group κ are in `results/agreement_kappa.csv`.

## Deviations, limitations and honest caveats

* **Domain shift:** the new data are long first-person r/AmItheAsshole *posts*, not short Reddit *comments* as in GoEmotions. They cover only **Oct–Dec 2023**, because Reddit's API was no longer usable for new collection and a pre-collected public corpus was used instead.
* **Single hand label per post** (plus an optional secondary label), annotated by one person, versus 3–5 trained raters per comment in the paper. Agreement figures are therefore not directly comparable with the paper's.
* **Small sample:** 100 posts give wide confidence intervals; kappa values are reported with bootstrap 95% intervals.
* **LLM judge** was run manually through a chat model (approved by the unit convenor), so results depend on that model and prompt; model name, date, prompt and raw replies are stored for traceability.
* **Proxy validation** on existing datasets is not accuracy: those datasets have no GoEmotions labels.

## Data, ethics and licence

* `data/labelled_master.csv` contains the text of **public** Reddit posts and no usernames or links; the poster column of the source corpus was dropped. If you are the author of a post and want it removed, please open an issue.
* Source corpus: [`derek-thomas/dataset-creator-reddit-amitheasshole`](https://huggingface.co/datasets/derek-thomas/dataset-creator-reddit-amitheasshole). Other datasets are downloaded from their original sources at run time and are not redistributed here.
* Code and my annotations are released under the MIT licence (see `LICENSE`). The GoEmotions data and code remain under their own licences.
* Ethical considerations (annotator and platform bias in the original paper; privacy, consent and misuse risks when publishing new emotion-labelled data) are discussed in the project report.

## References

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., and Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of the 58th Annual Meeting of the ACL*, 4040–4054.
