Feature Importance Analysis:

Logistic regression coefficients were analyzed to identify
which features most strongly influence the classifier.

Features with large positive coefficients increase the
likelihood that traffic is classified as malicious,
while negative coefficients push predictions toward benign.

Observation:

A small number of features dominate the model’s decisions,
suggesting that the classifier relies heavily on specific
traffic patterns when identifying attacks.

This may explain why certain attack categories are detected
reliably while others are frequently missed.