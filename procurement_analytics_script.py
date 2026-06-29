import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import linprog

def run_procurement_analytics():
    print("Executing End-to-End Procurement Analytics Suite...")
    
    # 1. Load Data
    file_path = "IT_Procurement_Project.xlsx"
    df_po = pd.read_excel(file_path, sheet_name="Purchase_Orders")
    df_items = pd.read_excel(file_path, sheet_name="Items")
    df_suppliers = pd.read_excel(file_path, sheet_name="Suppliers")
    
    df_po['PO_Date'] = pd.to_datetime(df_po['PO_Date'])
    df_items = df_items.rename(columns={'Category': 'Item_Category'})
    df_suppliers = df_suppliers.rename(columns={'Category': 'Supplier_Category'})

    # 2. Time Series Forecasting (Nada Sanders Framework)
    df_po['Year_Month'] = df_po['PO_Date'].dt.to_period('M')
    ts_data = df_po.groupby('Year_Month')['Total_Amount'].sum().reset_index()
    ts_data['Year_Month'] = ts_data['Year_Month'].dt.to_timestamp()
    ts_data.set_index('Year_Month', inplace=True)
    ts_data = ts_data.asfreq('MS')

    model = ExponentialSmoothing(ts_data['Total_Amount'], trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    forecast_values = fitted_model.forecast(steps=6)
    print("\n[1/3] Time Series Forecasting Complete. 6-Month rolling targets generated.")

    # 3. Should-Cost Regression (Kirit Pandit Framework)
    df_reg = df_po.merge(df_items[['Item_ID', 'Standard_Price', 'Item_Category']], on='Item_ID', how='left')
    df_reg = df_reg.merge(df_suppliers[['Supplier_ID', 'Risk_Score']], on='Supplier_ID', how='left')
    df_reg = df_reg[df_reg['Approval_Status'] == 'Approved'].dropna()

    X = df_reg[['Quantity', 'Lead_Time', 'Standard_Price', 'Risk_Score']]
    X = sm.add_constant(X)
    y = df_reg['Unit_Price']
    model_ols = sm.OLS(y, X).fit()
    
    df_reg['Predicted_Should_Cost'] = model_ols.predict(X)
    df_reg['Price_Variance'] = df_reg['Unit_Price'] - df_reg['Predicted_Should_Cost']
    print("[2/3] OLS Pricing Regression Complete. Price Leakage vectors mapped.")

    # 4. Prescriptive Optimization (Christian Mandl Framework)
    df_opt = pd.read_excel(file_path, sheet_name="Optimization_Model").dropna(subset=['Supplier_ID', 'Unit Cost($)'])
    unit_costs = df_opt['Unit Cost($)'].values
    capacity_limits = df_opt['Capacity Limit'].values
    
    c = unit_costs
    A_ub = np.eye(len(unit_costs))
    b_ub = capacity_limits
    A_eq = np.ones((1, len(unit_costs)))
    b_eq = np.array([25000]) # Target Demand
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*len(unit_costs), method='highs')
    print("[3/3] Prescriptive LP Solver Execution Complete.")
    print(f"\nOptimized Minimum Operational Spend Allocation: ${result.fun:,.2f}")

if __name__ == "__main__":
    run_procurement_analytics()
