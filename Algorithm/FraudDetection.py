"""
XGBoost algorithm for fraud detection on bank transaction data 
Loads the Preprocessed dataset and trains the binary classifier
to identify fraudulent transactions

"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import os 

#Load PreProcessed data 
print("="*50)
print("LOADING PREPROCESSED DATA")
print("="*50)

DATA_PATH = "/Users/joe.alcock/Documents/Advanced-Topics-in-AI/Data/Processed/preprocessed_transactions.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
print(f"Fraud cases : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"Legit cases : {(df['isFraud'] == 0).sum():,}  ({(1 - df['isFraud'].mean())*100:.2f}%)\n")

X = df.drop(columns=['isFraud'])
Y = df['isFraud']

#Split Features and Target 
print("=" * 50)
print("Splitting Features and Target")
print("=" * 50)

print(f"Feature Matrix X : {X.shape} ({X.shape[1]} features)")
print(f"Target Vector Y : {Y.shape}")
print(f"Features Used : {X.columns.tolist()}\n")

#Train / Test Split (80% Training / 20% test)
print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size = 0.2,
    random_state = 42,
    stratify= Y
)

print(f"Training Set : {X_train.shape[0]:,} rows ({Y_train.sum():,} fraud cases)")
print(f"Test Set : {X_test.shape[0]:,} rows {Y_test.sum():,} fraud cases")

#Calculate class imbalance weight

print("=" * 50)
print("CALCULATE CLASS WEIGHT")
print("=" * 50)

legit_count = (Y_train == 0).sum()
fraud_count = (Y_train == 1).sum()
scale_pos_weight = legit_count / fraud_count

print(f"Legitimate transaction : {legit_count:,}")
print(f"Fraud Transactions : {fraud_count:,}")
print(f"scale_pos_weight : {scale_pos_weight:.2f}")
print(f"Each fraud cause weighted as {scale_pos_weight:.1f}x a legitimate case")

#Build and Train the XGBoost Model
print("=" * 50)
print("BUILDING AND TRAINING THE XGBOOST MODEL")
print("=" * 50)

model = XGBClassifier(
    n_estimators = 300,
    max_depth = 6,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight,
    eval_metric = "aucpr",
    random_state = 42,
    verbosity = 0
)

print("TRAINING XGBBOOST MODEL")
print(f"Building {model.n_estimators} trees, max depth {model.max_depth}, learning rate {model.learning_rate}")

model.fit(
    X_train, Y_train,
    eval_set = [(X_test, Y_test)],
    verbose = False
)

print("Training Complete")

#Generate Predictions 
print("=" * 50)
print("GENERATING PREDICTIONS")
print("=" * 50)

Y_pred = model.predict(X_test)
Y_pred_prob = model.predict_proba(X_test)[:,1]

print(f"Transactions in test set : {len(Y_test):,}")
print(f"Predicted as Fraud : {Y_pred.sum():,}")
print(f"Actually Fraud : {Y_test.sum():,}\n")


#Model Evaluation
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

roc_auc = roc_auc_score(Y_test, Y_pred_prob)

print("CLASSIFICATION REPORT:")
print(" " + "-" * 50)
print(classification_report(Y_test, Y_pred, target_names=["legititmate", "fraud"]))
print(f"ROC_AUC SCORE: {roc_auc:.4f}")
print(f"(1.0 = perfect, 0.5 = random guessing)\n")

#Confusion Matrix
print("=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(Y_test, Y_pred)
tn,fp,fn,tp = cm.ravel()

print(f"True Negatives (correctly identified as legitimate) : {tn:,}")
print(f"False Positives (incorrectly identified as fraud) : {fp:,}")
print(f"False Negatives (fraud missed by model) : {fn:,}")
print(f"True Positives (correctly identified as fraud) : {tp:,}\n")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
    ax=ax
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — XGBoost Fraud Detection", fontsize=13, fontweight="bold")
plt.tight_layout()


#Feature Importance 
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)
importance_df = pd.DataFrame({
    "Feature" : X.columns,
    "Importance" : model.feature_importances_
}).sort_values("Importance", ascending = False)

print("\nFeature Importance Ranking: ")
print(" " + " - " * 35)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 200)
    print(f" {row['Feature']:<25} {row['Importance']:.4f} {bar}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue"
)
ax.set_title("Feature Importance — XGBoost Fraud Detection", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()

print("=" * 50)
print("XGBOOST FRAUD DETECTION COMPLETE")
print("=" * 50)
print(f"FINAL ROC-AUC SCORE: {roc_auc:.4f}")
print(f"FRAUD CASES CAUGHT: {tp:,} / {tp+fn:,} ({tp/(tp+fn)*100:.1f}% recall)")
print(f"FALSE ALARMS: {fp:,}")
print(f"MISSED FRAUDS: {fn:,}")
