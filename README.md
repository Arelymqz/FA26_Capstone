Machine Learning Intrusion Detection System Analysis
Overview:
This project evaluates how different machine learning models perform in intrusion detection systems (IDS), with a focus on threshold tuning, class imbalance, and real-world attack scenarios.

The goal is to understand how model decisions change under realistic conditions where malicious traffic is rare.

Technologies Used:
Python
Scikit-learn
Pandas / NumPy
Matplotlib
NSL-KDD Dataset

Experiments Conducted:
1. Model Comparison
Logistic Regression
Random Forest
Gradient Boosting

Compared performance across:
Precision
Recall
False Positives

2. Threshold Analysis
Tested multiple probability thresholds
Observed trade-offs between:
catching attacks (recall)
reducing false alarms (precision)

3. Realistic Attack Ratio Testing
Resampled dataset to simulate real-world conditions:
10% malicious
1% malicious
0.1% malicious
4. Cost-Sensitive Analysis
Assigned higher cost to missed attacks (false negatives)
Evaluated models under:
balanced cost
security-focused cost
critical system cost

Key Findings:
Lower attack ratios significantly reduce precision
Models behave very differently in realistic environments
Gradient Boosting performed best under extreme class imbalance
Threshold selection is critical in IDS performance

Real-World Impact:
This project shows that machine learning IDS models must be evaluated under realistic conditions.

In real networks:
attacks are rare
false positives are costly
missed attacks can be critical

This work highlights the importance of balancing detection performance with operational cost.