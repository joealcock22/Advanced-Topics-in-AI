#PreProcessing.py
#processes the dataset from kaggle, should be able to remove irrelevant data 
#keeping only data to be used by the machine learning algorithm 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 

#Open raw CSV file for pandas 

print("Loading Dataset...")
df = pd.read_csv("Data/Raw/bank_transactions_data_2_augmented_clean_2.csv")
print("Successfully Loaded Data")


