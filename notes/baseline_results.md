Baseline Model

A Logistic Regression classifier was trained as the baseline IDS model.

Preprocessing included:
- One-hot encoding of categorical features
- Feature scaling of numeric features

Baseline Performance

Accuracy ≈ 75%

Initial evaluation showed strong precision for malicious traffic
but lower recall.

This suggested that the model correctly identifies many attacks
but also misses a substantial portion of them.

This observation motivated further threshold experiments.