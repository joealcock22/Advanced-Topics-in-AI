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

