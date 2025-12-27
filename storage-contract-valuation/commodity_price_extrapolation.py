#!/usr/bin/env python
# coding: utf-8

# ## Data Source
# This analysis uses monthly natural gas price data from October 2020 to September 2024.
# The data file `Nat_Gas.csv` should be in the same directory as this notebook.

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[49]:


df = pd.read_csv('Nat_Gas.csv')


# In[50]:


df['Dates'] = pd.to_datetime(df['Dates']) #to adjust date column in datetime format so that pd not think it as just text
df = df.sort_values('Dates') #for dates in correct order


# In[51]:


df['Month'] = df['Dates'].dt.month
df['Year'] = df['Dates'].dt.year
df['Day'] = df['Dates'].dt.dayofyear


# In[54]:


#seasons repeat in a cycle(like a wave), so using cosine and sine functions for mapping days to an angle
df['sin_day'] = np.sin(2 * np.pi * df['Day'] / 365.25) #365.25 instead of 366 for considering leap year
df['cos_day'] = np.cos(2 * np.pi * df['Day'] / 365.25)


# In[55]:


# Features: year, and cyclical day features
X = df[['Year', 'sin_day', 'cos_day']].values
y = df['Prices'].values

model = LinearRegression()
model.fit(X, y)  


# In[56]:


def estimate_price(input_date):
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d')

    year = input_date.year
    day_of_year = input_date.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
 
    X_pred = np.array([[year, sin_day, cos_day]])
    predicted_price = model.predict(X_pred)[0]
    
    return predicted_price


# In[65]:


def price_contract(injection_dates, withdrawal_dates, 
                   injection_volumes, withdrawal_volumes,
                   injection_rate, withdrawal_rate, 
                   storage_rate, max_storage):
    """
    injection_dates: list of dates to inject gas
    withdrawal_dates: list of dates to withdraw gas
    injection_volumes: list of volumes (MMBtu) for each injection
    withdrawal_volumes: list of volumes (MMBtu) for each withdrawal
    injection_rate: cost per MMBtu to inject
    withdrawal_rate: cost per MMBtu to withdraw
    storage_rate: cost per month to store (fixed monthly fee)
    max_storage: maximum storage capacity (MMBtu)
    """
    
    # Define first and last dates
    first_injection = min(injection_dates)
    last_withdrawal = max(withdrawal_dates)
    
    # Calculate purchase costs (market price + injection fees)
    purchase_cost = sum(estimate_price(date) * volume + injection_rate * volume 
                    for date, volume in zip(injection_dates, injection_volumes))

    # Calculate sale revenue (market price - withdrawal fees)
    sale_revenue = sum(estimate_price(date) * volume - withdrawal_rate * volume 
                   for date, volume in zip(withdrawal_dates, withdrawal_volumes))
    
    # Calculate storage duration in months
    storage_months = ((last_withdrawal.year - first_injection.year) * 12 + 
                      (last_withdrawal.month - first_injection.month))
    storage_cost = storage_rate * storage_months
    
    # Create list of all transactions with (date, volume, type)
    transactions = []
    
    # Add injections (positive volumes)
    for date, volume in zip(injection_dates, injection_volumes):
        transactions.append((date, volume, 'injection'))
    
    # Add withdrawals (negative volumes)
    for date, volume in zip(withdrawal_dates, withdrawal_volumes):
        transactions.append((date, -volume, 'withdrawal'))
    
    # Sort by date
    transactions.sort(key=lambda x: x[0])
    
    # Track running inventory
    current_inventory = 0
    
    for date, volume, transaction_type in transactions:
        current_inventory += volume
        
        # Check if inventory goes negative
        if current_inventory < 0:
            return f"Error: Inventory goes negative on {date.date()}. Trying to withdraw more than available."
        
        # Check if exceeds capacity
        if current_inventory > max_storage:
            return f"Error: Inventory exceeds capacity on {date.date()}. Current: {current_inventory}, Max: {max_storage}"
    
    # Check if all gas is withdrawn by the end
    if current_inventory != 0:
        print(f"Warning: {current_inventory} MMBtu remains in storage at contract end.")
    
    # Calculate final contract value
    contract_value = sale_revenue - purchase_cost - storage_cost
    
    return contract_value


# In[ ]:


# Interactive input for contract pricing
print("=== Natural Gas Storage Contract Pricing ===\n")

# Get number of injections
num_injections = int(input("How many injection dates? "))
injection_dates = []
injection_volumes = []

for i in range(num_injections):
    date_str = input(f"Injection date {i+1} (YYYY-MM-DD): ")
    injection_dates.append(pd.Timestamp(date_str))
    volume = float(input(f"Injection volume {i+1} (MMBtu): "))
    injection_volumes.append(volume)

# Get number of withdrawals
num_withdrawals = int(input("\nHow many withdrawal dates? "))
withdrawal_dates = []
withdrawal_volumes = []

for i in range(num_withdrawals):
    date_str = input(f"Withdrawal date {i+1} (YYYY-MM-DD): ")
    withdrawal_dates.append(pd.Timestamp(date_str))
    volume = float(input(f"Withdrawal volume {i+1} (MMBtu): "))
    withdrawal_volumes.append(volume)

# Get contract parameters
injection_rate = float(input("Injection rate ($ per MMBtu): "))
withdrawal_rate = float(input("Withdrawal rate ($ per MMBtu): "))
storage_rate = float(input("Storage cost ($ per month): "))
max_storage = float(input("Maximum storage capacity (MMBtu): "))

# Calculate contract value
value = price_contract(injection_dates, withdrawal_dates,
                      injection_volumes, withdrawal_volumes,
                      injection_rate, withdrawal_rate,
                      storage_rate, max_storage)

if isinstance(value, str):  # Error message
    print(value)
else:
    print(f"\nContract Value: {value:,.2f}")

