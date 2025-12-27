# Quantitative Research Projects

A collection of quantitative finance and risk modeling projects demonstrating skills in time series forecasting, financial derivatives pricing, and credit risk assessment.

## Projects Overview

### 1. Commodity Price Forecasting
**Objective**: Extrapolate monthly natural gas price data to daily granularity and forecast future prices for storage contract valuation.

**Techniques**:
- Time series decomposition (trend + seasonality)
- Sine/cosine cyclical feature engineering
- Linear regression for trend modeling
- Price interpolation and extrapolation

**Key Files**:
- `commodity-price-forecasting/commodity-price-forecasting.ipynb`
- `commodity-price-forecasting/Nat_Gas.csv.csv`

---

### 2. Storage Contract Valuation
**Objective**: Build a pricing model for natural gas storage contracts accounting for injection/withdrawal costs, storage fees, and seasonal price differentials.

**Techniques**:
- Cash flow analysis
- Contract valuation modeling
- Inventory constraint validation
- Multi-period optimization

**Key Files**:
- `storage-contract-valuation/contract_pricing_model.ipynb`

**Results**: Created a production-ready pricing function that calculates expected contract value given injection/withdrawal dates, volumes, and cost parameters.

---

### 3. Credit Risk Modeling
**Objective**: Predict probability of default for personal loan borrowers and calculate expected losses.

**Techniques**:
- Logistic Regression
- Random Forest Classifier
- Model comparison (Accuracy, ROC-AUC, Log Loss)
- Expected loss calculation (PD × EAD × LGD)

**Key Files**:
- `credit-risk-modeling/credit-risk-modeling.ipynb`
- `credit-risk-modeling/Task 3 and 4_Loan_Data.csv`

**Results**: Achieved 99.9% accuracy with ROC-AUC of 0.9999 in predicting loan defaults. Built function to estimate expected loss for any borrower profile.

---

### 4. FICO Score Bucketing
**Objective**: Develop optimal bucketing strategy to convert continuous FICO scores into categorical risk ratings.

**Techniques**:
- Quantization optimization
- Mean Squared Error (MSE) minimization
- Log-Likelihood maximization
- Dynamic programming approach

**Key Files**:
- `bucket_generator/bucket_generator.ipynb`
- `bucket_generator/Task 3 and 4_Loan_Data.csv.csv`

**Results**: Implemented two bucketing methods with comparative analysis, creating optimal risk rating system for mortgage portfolio.

---

## Technical Stack

**Languages**: Python 3.11+

**Core Libraries**:
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib
- **Optimization**: scipy


