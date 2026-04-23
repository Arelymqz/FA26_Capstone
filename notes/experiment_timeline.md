1️⃣ Dataset Understanding (First step of the investigation)

    KEY FINDINGS:

        Train Set:
        - 53% benign
        - 46% malicious

        Test Set:
        - 43% benign
        - 57% malicious

        What this means:
        - The dataset is fairly balanced
        - Accuracy alone will not hide imbalance
        - This makes evaluation metrics like recall meaningful

    SUMMARY:
    The NSL-KDD dataset shows a relatively balanced distribution of benign and malicious traffic across both training and testing splits. This allows evaluation metrics such as recall and precision to meaningfully represent model performance without severe class imbalance bias.

2️⃣ Baseline Model Training
    
    MODEL: Logistic Regression
    
        Preprocessing:
        - One-hot encoding
        - feature scaling

        Evaluation Tools Generated:
        - ROC Curve
        - Precision-Recall Curve

        Results:
        - AUC = 0.781

        Interpretation:
        - classifier performs better than random
        - still not strong enough to be considered highly reliable

    The baseline logistic regression classifier achieved an ROC-AUC score of approximately 0.78, indicating moderate discriminative capability between benign and malicious traffic.

    Precision Recall Curve Results:
        - Average precision = 0.865

        Meaning:
        - When the model raises alerts, it is usually correct
        - But this does not guarantee that attacks are detected

3️⃣ Threshold Tradeoff Experiment (Main experiment)
    
    Varied decision threshold from 0.05 → 0.95

    METRICS TRACKED:
    - recall
    - precision
    - false negatives
    - false positives

    KEY OBSERVATIONS:

        When Threshold Increases:
        - Recall ↓
        - Precision ↑
        - False positives ↓

        Importance Insight: lower thresholds catch more attacks but generate more alerts

4️⃣ Missed Attacks Analysis

    KEY DISCOVERY:
    even with high precision, the system misses thousands of attacks
    - supports thesis that good metrics can hide operational risks

5️⃣ Attack Category Analysis (Most important finding)
    
    DETECTION BY CATEGORY
    - DoS: 82%
    - Probe: 75%
    - U2R: 8.5%
    - R2L: 1.1%

    IDS detects network attacks well but fails almost completely on R2L attacks
    - R2L attacks often look similar to legitimate traffic, making them harder for statistical models to detect

6️⃣ Category vs Threshold Analysis

    FINDINGS:
    Increading threshold destroys detection fro difficult attack types

    MEANING:
    A system optimized for precision would completely miss R2L attacks

7️⃣ Feature Importance Analysis

    INTERPRETATION:
    - A small number of features dominate the model
    - This means the classifier relies heavily on specific network behaviors

    This explains why...
    - DoS attacks are detected easily
    - R2L attacks are missed
    