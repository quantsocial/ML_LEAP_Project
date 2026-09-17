# Does a Neural Network Beat a Classical Model?

A comparison of logistic regression and a small dense neural network on the task of predicting whether a person earns more than $50,000/year, using the Adult Census Income dataset.

---

## Problem statement

A public agency wants to use survey data to flag which households are likely to be in a higher-income bracket, so a benefit or outreach programme can be targeted. Two kinds of models are on the table: a simple, transparent one whose decisions can be explained to the public, and a more complex neural network that might — or might not — be more accurate.

**Central question:** Does the neural network perform better, and is its additional complexity worthwhile?

## Dataset

[Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult), loaded via `sklearn.datasets.fetch_openml("adult", version=2)`. Each row describes one person from U.S. Census survey data, with a mix of:

- **Numeric features:** age, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical features:** workclass, education, marital status, occupation, relationship, race, sex, native country

**Target:** whether the person earns `>50K` or `<=50K` per year (binary; encoded via `LabelEncoder`). The positive class (`>50K`) makes up roughly 24% of the data, so the task is class-imbalanced.

## Method

1. **Split first** — an 80/20 stratified train/test split, done before any preprocessing to avoid data leakage.
2. **Preprocessing** — a `ColumnTransformer` combining two pipelines:
   - Numeric: median imputation → standard scaling
   - Categorical: most-frequent imputation → one-hot encoding
3. **Classical model** — `LogisticRegression`, evaluated with 5-fold stratified cross-validation on the training set only, then fit on the full training set and scored on the held-out test set.
4. **Neural network** — a small Keras `Sequential` model (Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)), trained with early stopping on validation loss, evaluated on the same test set as the classical model.
5. **Primary metric: F1**, not accuracy, because the target is imbalanced — a model can score well on accuracy just by leaning toward the majority class, so F1 (which accounts for both precision and recall on the minority class) is a fairer measure of how well each model actually identifies higher earners.

### Extended analysis

Two additional checks were layered on top of the required comparison, to stress-test the headline result rather than take it at face value:

- **Fairness by subgroup** — F1 and accuracy computed separately by sex and by race, to check whether either model's performance is uneven across demographic groups.
- **Threshold tuning** — since both models default to a 0.5 probability cutoff, this checks whether that cutoff is actually optimal for F1, or whether it's under-crediting both models given the class imbalance.

## Results

| Model | F1 | Accuracy |
|---|---|---|
| Logistic Regression | 0.656 | 0.852 |
| Neural Network | 0.673 | 0.858 |

The network outperforms logistic regression on both metrics, but by a modest margin (+0.017 F1, +0.006 accuracy) at the set 0.5 threshold. Both models make more false negatives than false positives on the >50K class (recall 0.59 for logistic regression, 0.61 for the network). Meaning that both are more likely to miss a genuine higher earner than to wrongly flag someone, though the network narrows this slightly while also being marginally more precise.

**Threshold tuning:** the default 0.5 cutoff undersells both models. Moving to each model's own F1-optimal threshold raises logistic regression to F1 = 0.693 (threshold ≈ 0.37) and the network to F1 = 0.703 (threshold ≈ 0.30). A bigger jump than the entire gap between the two models at the default cutoff. Once thresholds are tuned, the network's real advantage narrows to about 1 F1 point.

**Fairness by subgroup:** F1 for the >50K class is consistently lower for women than men (0.60–0.63 vs. 0.67–0.68) and lower for Black individuals than White individuals (0.57–0.58 vs. 0.66–0.68). The network improves F1 slightly over logistic regression for every group tested, but the gaps between groups persist for both models.

## Interpretation

This dataset does not produce a decisive win for either approach. Logistic regression, a fast and fully transparent model, gets most of the way to the network's performance on its own, and once the decision threshold is tuned rather than left at the 0.5 default, most of the apparent gap between the two models closes, indicating that threshold choice mattered more to F1 here than model choice did.

**My assessment: the added complexity of the neural network was not worth it on this dataset.** A ~1.7-point F1 gain does not offset the loss of interpretability, the added architecture/hyperparameter decisions, and the longer training time, especially for a deployment where an agency needs to be able to explain why a household was or wasn't flagged. Many of this dataset's strongest predictors (age, education, hours worked, capital gain) relate to income in a fairly linear, additive way, which likely explains why a linear model came so close to matching a much more flexible one.

### Ethical considerations

Both models perform worse for women and Black individuals than for men and White individuals, so this model risks reproducing existing income disparities rather than correcting them. Since missing someone who deserves outreach (a false negative) is more costly than over-including someone (a false positive), and both models lean toward false negatives, this uneven, error-prone behavior makes the case for logistic regression's transparency — an opaque model's mistakes are harder to catch and explain when they affect real people.


**What could be improved with more time:**
- Average the network's results over multiple random seeds rather than a single training run, since neural network performance can vary run-to-run.
- Extend the fairness analysis to intersectional subgroups (e.g., race × sex together), not just one attribute at a time.
- Compare against a stronger classical baseline (e.g., a tree-based ensemble) to check whether the network's edge holds up against more than just logistic regression.

## Repository structure


- [`README.md`](./README.md)               — this file
- [`Codebook.ipynb`](./Codebook.ipynb)     — completed notebook, all cells run
-  [`requirements.txt`](./requirements.txt) — Python dependencies







