# AI Lab 11 — Classifier Comparison (Pandas + scikit-learn)

This project compares multiple machine learning classifiers on a Kaggle-style tabular dataset.

It follows the **Lab 11** workflow:
1. Split dataset into **X** and **y**
2. Convert categorical/object columns to **integer codes**
3. Train/test split (non-shuffled)
4. Train and evaluate multiple classifiers:
   - Bernoulli Naive Bayes
   - Random Forest
   - Gaussian Naive Bayes
   - Decision Tree
   - Multinomial Naive Bayes
   - K-Nearest Neighbors
5. Compute **Accuracy, Precision, Recall, F1**
6. Save plots and metrics to `outputs/`

## Dataset
- `data/dataset.csv` is included in the repo.
- The script expects a **classification target** column named `loan_status`.

## Run
```bash
python main.py
```

## Outputs
After running, you will get:
- `outputs/accuracy_plot.png`
- `outputs/f1_bar_chart.png`
- `outputs/results.txt`

## Notes
- The dataset included is **synthetic** (Kaggle-style) with ~5000 rows.
- Categorical columns are encoded using `pd.factorize()`.
- Train/test split is done with `shuffle=False` to match the lab idea.

## License
MIT (see `LICENSE`).

