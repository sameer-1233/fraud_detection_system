Fraud Detection Model
📌 Problem Statement

A bank wants to develop a real-time fraud detection system that identifies fraudulent transactions while minimizing false positives.
The model should capture relationships between transaction features and provide probabilistic outputs to support decision-making.

Dataset: Google Sheet Provided

🛠️ Approach
1. Data Preprocessing

Handled categorical features (Location, MerchantCategory) with OneHotEncoding.

Standardized numeric features using StandardScaler.

Built a ColumnTransformer + Pipeline to ensure consistent preprocessing during training & inference.

Addressed class imbalance (very few fraud cases) using:

Class weights (balanced)

SMOTE oversampling

2. Model Development

Three models were evaluated:

Logistic Regression (baseline, interpretable, probability outputs)

Random Forest (ensemble, nonlinear feature capture, combined with SMOTE)

XGBoost (gradient boosting, imbalance handling with scale_pos_weight)

3. Performance Evaluation

Metrics: Precision, Recall, F1, ROC AUC, Precision-Recall AUC (better for imbalance).

Tried threshold tuning (varied cutoff from 0.1 → 0.9).

Used Cross-validation (StratifiedKFold) for stable estimates.

📊 Results Summary
Model	Handling Imbalance	Precision (Fraud=1)	Recall (Fraud=1)	F1 (Fraud=1)	ROC AUC	PR AUC
Logistic Regression (pipeline)	Class weights	0.06	0.60	0.12	~0.41	Low
Random Forest + SMOTE	SMOTE + class_weight	Higher than LR	Moderate	Moderate	~0.62	Better
XGBoost (scale_pos_weight)	Built-in imbalance	0.00	0.00	0.00	~0.48	Very low
XGBoost + SMOTE (CV)	SMOTE inside CV	–	–	–	–	0.061 ± 0.012

⚠️ With only ~5 fraud cases in the test set, metrics are unstable.

📝 Discussion

Fraud detection is an extremely imbalanced classification problem.

Logistic Regression detected some fraud but had very low precision.

Random Forest with SMOTE showed slightly more balanced results.

XGBoost struggled due to too few fraud cases.

PR AUC scores were very low across models, highlighting the challenge.

🚀 Future Work

Collect more fraud transactions (current dataset too small).

Engineer better features (transaction velocity, user history, merchant risk scoring).

Explore anomaly detection or hybrid supervised + unsupervised approaches.

Align threshold tuning with business objectives (recall vs. false positives).

✅ Conclusion

Best performing model here: Random Forest with SMOTE.

However, performance is limited by data scarcity, not just algorithm choice.

For real-world deployment, the bank must:

Continuously collect fraud data

Engineer strong behavioral features

Optimize thresholds based on business tolerance for false positives

📂 Files in This Submission

fraud_detection_system.ipynb → Full Colab notebook (code + results).

README.md → Project summary & results (this file).

fraud_detection_system.pdf → Exported report (optional).