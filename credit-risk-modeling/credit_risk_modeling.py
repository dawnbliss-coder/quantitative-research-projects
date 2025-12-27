#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

# Load data 
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check default rate
default_rate = df['default'].mean()
print(f"\nDefault rate: {default_rate:.2%}")
print(f"Defaulters: {df['default'].sum()}")
print(f"Non-defaulters: {(df['default']==0).sum()}")

# Visualize some relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Income vs Default
axes[0, 0].hist([df[df['default']==0]['income'], df[df['default']==1]['income']], 
                label=['No Default', 'Default'], bins=30)
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Income Distribution by Default Status')
axes[0, 0].legend()

# FICO Score vs Default
axes[0, 1].hist([df[df['default']==0]['fico_score'], df[df['default']==1]['fico_score']], 
                label=['No Default', 'Default'], bins=30)
axes[0, 1].set_xlabel('FICO Score')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('FICO Score Distribution by Default Status')
axes[0, 1].legend()

# Total Debt vs Default
axes[1, 0].hist([df[df['default']==0]['total_debt_outstanding'], 
                 df[df['default']==1]['total_debt_outstanding']], 
                label=['No Default', 'Default'], bins=30)
axes[1, 0].set_xlabel('Total Debt Outstanding')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Debt Distribution by Default Status')
axes[1, 0].legend()

# Years Employed vs Default
axes[1, 1].hist([df[df['default']==0]['years_employed'], 
                 df[df['default']==1]['years_employed']], 
                label=['No Default', 'Default'], bins=30)
axes[1, 1].set_xlabel('Years Employed')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Employment Duration by Default Status')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data_exploration.png')  # Save instead of show for script
print("\nData exploration plot saved as 'data_exploration.png'")

# Separate features (X) and target (y)
X = df.drop(['customer_id', 'default'], axis=1)
y = df['default']

print("\nFeatures (X):", X.columns.tolist())
print("Target (y):", y.name)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete!")

# ===== TRAIN LOGISTIC REGRESSION =====
print("\n=== Training Logistic Regression ===")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

if not lr_model.n_iter_[0] < lr_model.max_iter:
    print("WARNING: Model may not have converged!")

lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred_proba)
lr_logloss = log_loss(y_test, lr_pred_proba)

print(f"Accuracy: {lr_accuracy:.4f}")
print(f"ROC-AUC: {lr_auc:.4f}")
print(f"Log Loss: {lr_logloss:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['No Default', 'Default']))

# ===== TRAIN RANDOM FOREST =====
print("\n=== Training Random Forest ===")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)
rf_logloss = log_loss(y_test, rf_pred_proba)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"ROC-AUC: {rf_auc:.4f}")
print(f"Log Loss: {rf_logloss:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Default', 'Default']))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# ===== MODEL COMPARISON =====
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'ROC-AUC': [lr_auc, rf_auc],
    'Log Loss': [lr_logloss, rf_logloss]
})

print("\n=== Model Comparison ===")
print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

metrics = ['Accuracy', 'ROC-AUC', 'Log Loss']
for i, metric in enumerate(metrics):
    axes[i].bar(['Logistic Regression', 'Random Forest'], 
                comparison[metric], color=['blue', 'green'])
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylim([0, max(comparison[metric]) * 1.2])

plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel comparison plot saved as 'model_comparison.png'")

# Select best model
best_model_name = comparison.loc[comparison['ROC-AUC'].idxmax(), 'Model']
print(f"\nBest model based on ROC-AUC: {best_model_name}")

# ===== SET FINAL MODEL =====
# Store the trained models and scaler for later use
final_model = lr_model  # Using logistic regression as final model
final_scaler = scaler

print("\n" + "="*50)
print("MODELS TRAINED AND READY FOR PREDICTIONS")
print("="*50)

# ===== PREDICTION FUNCTION =====
def predict_expected_loss(borrower_data, loan_amount, recovery_rate=0.10):
    """
    Predict expected loss for a loan
    
    Parameters:
    - borrower_data: dict with keys matching feature names
    - loan_amount: The loan amount (Exposure at Default)
    - recovery_rate: Expected recovery if default occurs (default 10%)
    
    Returns:
    - expected_loss: Expected loss in dollars
    - probability_default: Probability of default (0 to 1)
    """
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([borrower_data])
    
    # Ensure columns are in correct order
    input_df = input_df[X.columns]
    
    # Scale the input data
    input_df_scaled = pd.DataFrame(
            final_scaler.transform(input_df),
            columns=input_df.columns
        )
    
    # Predict probability of default
    probability_default = final_model.predict_proba(input_df_scaled)[0][1]
    
    # Calculate Loss Given Default
    lgd = 1 - recovery_rate
    
    # Calculate Expected Loss
    expected_loss = probability_default * loan_amount * lgd
    
    return expected_loss, probability_default


# ===== TEST THE FUNCTION =====
print("\n=== Example Prediction ===")

test_borrower = {
    'credit_lines_outstanding': 8,
    'loan_amt_outstanding': 45000,
    'total_debt_outstanding': 80000,
    'income': 30000,      # Low income
    'years_employed': 1,   # Short employment
    'fico_score': 520      # Poor credit
}

test_loan_amount = 15000

loss, pd_prob = predict_expected_loss(test_borrower, test_loan_amount)

print(f"Borrower details: {test_borrower}")
print(f"Loan amount: ${test_loan_amount:,}")
print(f"Probability of Default: {pd_prob:.2%}")
print(f"Expected Loss: ${loss:,.2f}")

# Additional test cases
print("\n=== Additional Test Cases ===")

# Good borrower
good_borrower = {
    'credit_lines_outstanding': 2,      # Low
    'loan_amt_outstanding': 10000,      # Low  
    'total_debt_outstanding': 10000,    # Very low
    'income': 130000,                   # Very high!
    'years_employed': 2,                # Decent
    'fico_score': 800                   # Excellent!
}

loss_good, pd_good = predict_expected_loss(good_borrower, 15000)
print(f"\nGood Borrower:")
print(f"  Probability of Default: {pd_good:.2%}")
print(f"  Expected Loss: ${loss_good:,.2f}")

# Average borrower
avg_borrower = {
    'credit_lines_outstanding': 3,
    'loan_amt_outstanding': 20000,
    'total_debt_outstanding': 30000,
    'income': 80000,
    'years_employed': 7,
    'fico_score': 750
}

loss_avg, pd_avg = predict_expected_loss(avg_borrower, 15000)
print(f"\nAverage Borrower:")
print(f"  Probability of Default: {pd_avg:.2%}")
print(f"  Expected Loss: ${loss_avg:,.2f}")

print("\n" + "="*50)
print("SCRIPT EXECUTION COMPLETE")
print("="*50)