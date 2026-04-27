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

#Drop The Identifier Columns 
id_columns = ["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"]
df.drop(columns=id_columns, inplace = True)
print(f"Dropped ID columns: {id_columns}")
print(f"Remaining Columns: {df.columns.tolist()}\n")

#Parse and Extract date and time features
#Extracted Features: 
#Transaction hours, Transaction day, Transaction Month, isWeekend

print("Parsing and Extracting date/time features")
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="mixed")

df["TransactionHour"] = df["TransactionDate"].dt.hour
df["TransactionDay"] = df["TransactionDate"].dt.dayofweek
df["TransactionMonth"] = df["TransactionDate"].dt.month
df["isWeekend"] = (df["TransactionDay"] >= 5).astype(int)

#drop original date column as raw string is no longer needed 
df.drop(columns = {"TransactionDate"}, inplace = True)
print("Extracted : TransationHour, TransactionDay, TransactionMonth, isWeekend\n")

#Create Fraud Label
#Uses real-world Fraud indicators to classify fraudulent transactions 
print("Creating Fraud Label...")
q95_amount = df["TransactionAmount"].quantile(0.95)
rule_multiple_logins = df["LoginAttempts"] > 1
rule_drain_account = (
    (df["TransactionAmount"] > q95_amount) &
    (df["TransactionAmount"] > df["AccountBalance"] * 0.90)
)

df["isFraud"] = (rule_multiple_logins | rule_drain_account).astype(int)

#Calculates and returns the number of fraudulent and legitimate transactions
#as well as the percentage rates of each
fraud_count = df["isFraud"].sum()
total = len(df)
fraud_rate = fraud_count / total * 100
print(f"Fraud label created")
print(f"Fraud Transactions : {fraud_count:,} ({fraud_rate:.2f}%)")
print(f"Legitimate Transactions: {total - fraud_count:,} ({100 - fraud_rate:.2f}%)")

#Feature Engineering 
print("Engineering Predictive Features...")
df["AmountToBalanceRatio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1e-19)
df["IsHighLoginAttempt"] = (df["LoginAttempts"] > 1).astype(int)
print("Created: AmountToBalanceRatio and IsHighLoginAttempt")

#Encode categoric labels
print("encoding categoric labels")
catergoric_cols = ["TransactionType", "Channel", "CustomerOccupation", "Location"]
le = LabelEncoder()

for col in catergoric_cols:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded {col}")
print()

#Scale Numeric Features 
print("Scaling numerics")
numeric_cols = [
    "TransactionAmount",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "CustomerAge", 
    "AmountToBalanceRatio"
]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"Scaled: {numeric_cols}\n")

#Final Checks and Save 
print("Final Dataset Overview")
print(f"Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Columns : {df.columns.tolist()}")
print(f"Missing Values : {df.isnull().sum().sum()}")
print()

os.makedirs("/Users/joe.alcock/Documents/Advanced-Topics-in-AI/Data/Processed/", exist_ok = True)
output_path = "/Users/joe.alcock/Documents/Advanced-Topics-in-AI/Data/Processed/preprocessed_transactions.csv"
df.to_csv(output_path, index = False)
print("PreProcessing Successful!")




