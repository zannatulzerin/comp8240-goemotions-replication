# Making the authors' `ppca.py` run on current Colab

The GoEmotions authors' `ppca.py` (Principal Preserved Component Analysis, used to test whether the emotion
categories are distinguishable) did not run on Google Colab (Python 3.13, current pandas / NumPy / SciPy /
Matplotlib). This note records what failed, why, and the five patches applied. The authors' repository is **not
modified**; the notebook (Section 1) clones it and patches `ppca.py` at run time, so the patches are re-applied on
every fresh runtime.

## Symptoms

1. `AttributeError: type object 'Series' has no attribute 'nonzero'`
2. After patch 1, the script printed `Which dimensions to keep? [False] [nan]` and then crashed with
   `IndexError: invalid index to scalar variable` (`ppca.py`, `corr_mean[emotion2ppc[e]]`), after
   `RuntimeWarning: Mean of empty slice` and several `ConstantInputWarning` messages from `spearmanr`.

## Root cause of symptom 2

* `data/emotions.txt` lists 28 names, **including `neutral`**, so the ratings array has 28 columns.
* The script then **deletes every neutral rating** (`data = data[data["neutral"] == 0]`), so the `neutral` column is
  all zeros.
* The last principal component therefore has zero variance. For it, `spearmanr` returns NaN
  (`ConstantInputWarning`).
* In `LeaveOut`, a NaN partial correlation triggers `break`, so each rater's list of correlations stops at 27
  values; the following check `len(partial_corrs) < len(all_emotions)` (27 < 28) then **discards every rater**.
  `corr_per_rater` ends up empty, the Wilcoxon test returns NaN, and the plotting code indexes an empty mean.
* The authors presumably obtained a non-constant (tiny non-zero) value for that component with older NumPy/SciPy
  versions. That is a likely, untested explanation; the diagnosis above is what was verified.

How this was found: the failure was reproduced on synthetic data with exactly the GoEmotions CSV schema (28 emotion
columns, neutral removed, 3–5 raters per example), then the fix was confirmed on the real data (below).

## Patches (applied by the notebook, each checked to match exactly once)

| # | Change | Why |
|---|---|---|
| 1 | `pd.Series.nonzero(x)[0]` → `x.to_numpy().nonzero()[0]` | `Series.nonzero` was removed in pandas 2 |
| 2 | After reading `emotions.txt`, drop `neutral` from `all_emotions` (27 emotions, as in the paper) | Removes the all-zero column (root cause above) |
| 3 | `num_components = 28` → `len(v)` | The hard-coded 28 no longer matches 27 components |
| 4 | `ax.figure.colorbar(sm)` → `ax.figure.colorbar(sm, ax=ax)` | Current Matplotlib needs an explicit axis |
| 5 | Return before the t-SNE plot | `plots/color_order.txt` is not shipped in the repo; the plot is not needed for the significance test |

## Result on the real data

* Ratings were processed for all 27 emotions without errors.
* **All 27 components are significant** after Bonferroni correction (α = 0.00185; every adjusted p < 2×10⁻¹⁰).
* Mean leave-one-rater-out agreement per emotion is saved to `results/ppca_emotion_agreements.csv`: highest for
  gratitude (0.644), admiration (0.539) and amusement (0.480); lowest for grief (0.162), relief (0.165) and
  nervousness (0.172).

## What is *not* reproduced

* The t-SNE figure (patch 5).
* Exact numeric equality with the paper: the PPCA uses random splits of raters, and library versions differ, so
  small differences are expected.
