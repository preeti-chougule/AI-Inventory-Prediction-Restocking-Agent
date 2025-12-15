import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# --- Global Configuration ---
RANDOM_STATE = 42
# Z-score for a 95% service level
SERVICE_LEVEL_Z = 1.65 
# The dataset's default filename
FILE_NAME = "retail_store_inventory.csv" 
# NOTE: Update this path to where you save the file
FILE_PATH = f"/Users/pruthatrivedi/Desktop/AI_inventory Management/dataset/retail_store_inventory.csv"
# --- End Configuration ---

# --------------------------------
# 1️⃣ Load & Clean Data
# --------------------------------
def load_and_clean_data(filepath):
    """Loads, cleans, and prepares data for feature engineering."""
    print("1/6: Loading and cleaning data...")
    if not os.path.exists(filepath):
        # Fallback path assumption if the user's path is not correctly set.
        fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FILE_NAME)
        if os.path.exists(fallback_path):
             filepath = fallback_path
        else:
             raise FileNotFoundError(f"File not found. Please ensure '{FILE_NAME}' is present and update the FILE_PATH variable in main.py.")
        
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace(' ', '_') # Clean column names
    
    # Convert date and sort
    df['Date'] = pd.to_datetime(df['Date']) 
    df = df.sort_values(by=['Store_ID', 'Product_ID', 'Date']).reset_index(drop=True)
    
    # Drop rows with missing crucial information
    df = df.dropna(subset=['Units_Sold', 'Inventory_Level']).copy()

    # Rename columns to match the general flow
    df.rename(columns={'Units_Sold': 'Demand', 'Inventory_Level': 'Current_Stock_Start'}, inplace=True)
    
    print(f"Loaded {len(df)} daily records.")
    return df

# --------------------------------
# 2️⃣ Feature Engineering
# --------------------------------
def create_features(df):
    """Creates time-based, lag, and categorical features."""
    print("2/6: Creating features...")
    
    # Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Grouping key for unique product-store combinations
    group_key = ['Store_ID', 'Product_ID']

    # Lag and Rolling Mean features (critical for time series)
    df['Demand_lag_1'] = df.groupby(group_key)['Demand'].shift(1)
    df['Demand_lag_7'] = df.groupby(group_key)['Demand'].shift(7)
    df['Rolling_Mean_7'] = df.groupby(group_key)['Demand'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )

    # One-Hot Encoding for categorical features
    potential_cols = ['Region', 'Category', 'Weather_Condition', 'Holiday/Promotion', 'Seasonality']
    
    # Only encode columns that actually exist in this specific dataset
    categorical_cols = [col for col in potential_cols if col in df.columns]
    
    # FIX #1: Preserve 'Category' for final output because get_dummies drops the original
    if 'Category' in df.columns:
        df['Category_Output'] = df['Category']

    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)

    # Restore the original 'Category' column name
    if 'Category_Output' in df.columns:
        df.rename(columns={'Category_Output': 'Category'}, inplace=True)

    # Drop NaNs created by lagging/rolling features (first 7 days of data for each group)
    df = df.dropna().reset_index(drop=True)
    return df

# --------------------------------
# 3️⃣ Forecast Sales (Demand)
# --------------------------------
def forecast_demand(df, feature_cols, target_col='Demand'):
    """Trains a Random Forest model and generates demand forecasts."""
    print("3/6: Training model and forecasting demand...")
    
    # Create a combined key for iteration
    df['Group'] = df['Store_ID'].astype(str) + '_' + df['Product_ID'].astype(str)
    groups = df['Group'].unique()
    results = []

    for group in groups:
        group_data = df[df['Group'] == group].copy().reset_index(drop=True)
        
        # Skip if not enough data for train/test split
        if len(group_data) < 20: continue 

        # Time-series split: Train on 90% history, Test on last 10%
        train_size = int(len(group_data) * 0.9)
        
        X_train = group_data[feature_cols].iloc[:train_size]
        X_test = group_data[feature_cols].iloc[train_size:]
        y_train = group_data[target_col].iloc[:train_size]
        y_test = group_data[target_col].iloc[train_size:]

        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        
        # Save results
        store_results = group_data.iloc[train_size:].copy()
        store_results['Actual_Demand'] = y_test.values
        store_results['Predicted_Demand'] = y_pred
        
        results.append(store_results)

    forecast_df = pd.concat(results).reset_index(drop=True)
    return forecast_df

# --------------------------------
# 4️⃣ Generate Inventory Recommendations (Dynamic Simulation)
# --------------------------------
def generate_inventory_recommendations(forecast_df, service_level_z):
    """
    Simulates daily inventory dynamically for each Product-Store group.
    """
    print("4/6: Generating dynamic inventory recommendations...")
    
    forecast = forecast_df.copy()

    # --- Step 4a: Calculate Safety Stock (SS) and Reorder Level (ROL) ---
    
    # Compute forecast error (Standard Deviation of Error)
    forecast['error'] = forecast['Actual_Demand'] - forecast['Predicted_Demand']
    
    group_key = ['Store_ID', 'Product_ID']
    store_product_stats = forecast.groupby(group_key)['error'].std().reset_index()
    store_product_stats.rename(columns={'error': 'forecast_std'}, inplace=True)
    
    # Merge std deviation back to the main dataframe
    forecast = forecast.merge(store_product_stats, on=group_key, how='left')

    # Safety Stock (SS) = Z-score * Std Dev of Error
    forecast['safety_stock'] = forecast['forecast_std'] * service_level_z
    # Reorder Level (ROL) = Predicted Demand + Safety Stock
    forecast['reorder_level'] = forecast['Predicted_Demand'] + forecast['safety_stock']
    
    all_recommendations = []
    groups = forecast['Group'].unique()

    # --- Step 4b: Dynamic Inventory Simulation (Iterative) ---
    for group in groups:
        group_data = forecast[forecast['Group'] == group].sort_values('Date').copy() 
        
        # Initialize the stock using the first available stock level from the test set
        current_stock = group_data['Current_Stock_Start'].iloc[0]
        
        current_stock_list, stock_after_sales_list = [], []
        # Correctly initialized as two empty lists
        restock_qty_list, restock_needed_list = [], []
        stock_after_restock_list = []
        
        for index, row in group_data.iterrows():
            
            current_stock_list.append(current_stock)
            
            # Stock after sales (using Actual Demand)
            stock_after_sales = current_stock - row['Actual_Demand']
            # Ensure stock doesn't go below zero (simulating a stockout)
            stock_after_sales = max(0, stock_after_sales)
            stock_after_sales_list.append(stock_after_sales)
            
            # Check if restock is needed (stock drops below ROL)
            restock_needed = stock_after_sales < row['reorder_level']
            restock_needed_list.append(restock_needed)
            
            # Calculate restock quantity
            restock_qty = 0
            if restock_needed:
                # Restock up to the Reorder Level
                restock_qty = row['reorder_level'] - stock_after_sales
                restock_qty = max(0, restock_qty) 
                
            restock_qty_list.append(restock_qty)
            
            # Stock after restock
            stock_after_restock = stock_after_sales + restock_qty
            stock_after_restock_list.append(stock_after_restock)
            
            # Update current stock for the NEXT day
            current_stock = stock_after_restock 
            
        # Append simulated columns
        group_data['current_stock'] = current_stock_list
        group_data['stock_after_sales'] = stock_after_sales_list
        group_data['restock_needed'] = restock_needed_list
        # FIX #2: Convert list to numpy array before rounding
        group_data['restock_qty'] = np.array(restock_qty_list).round(0) 
        group_data['stock_after_restock'] = stock_after_restock_list

        all_recommendations.append(group_data)

    inventory_df = pd.concat(all_recommendations).reset_index(drop=True)
    
    output_cols = [
        'Date', 'Store_ID', 'Product_ID', 'Category', 
        'Predicted_Demand', 'Actual_Demand',
        'safety_stock', 'reorder_level',
        'current_stock', 'stock_after_sales',
        'restock_needed', 'restock_qty', 'stock_after_restock'
    ]
    return inventory_df[output_cols]

# --------------------------------
# 5️⃣ Save Outputs
# --------------------------------
def save_outputs(forecast_df, inventory_df):
    """Saves the final forecast and inventory recommendations to CSV files."""
    print("5/6: Saving outputs...")
    
    # Drop the temporary 'Group' column before saving
    for df in [forecast_df, inventory_df]:
        if 'Group' in df.columns:
            df.drop(columns=['Group'], inplace=True)
        if 'Current_Stock_Start' in df.columns:
             df.drop(columns=['Current_Stock_Start'], inplace=True)
             
    forecast_df.to_csv("RetailInventory_Demand_Forecast.csv", index=False)
    inventory_df.to_csv("RetailInventory_Recommendations.csv", index=False)
    print("Forecast saved to RetailInventory_Demand_Forecast.csv")
    print("Inventory recommendations saved to RetailInventory_Recommendations.csv")

# --------------------------------
# 6️⃣ Main Pipeline
# --------------------------------
def main():
    """Executes the entire AI Inventory Prediction Agent pipeline."""
    
    # Load and process data
    df = load_and_clean_data(FILE_PATH)
    df = create_features(df)

    # Automatically generate feature columns based on what's left after cleaning/OHE
    # FIX #3: Added 'Category' to exclude_cols because we preserved it as text for output,
    # but the model cannot train on text.
    exclude_cols = ['Date', 'Store_ID', 'Product_ID', 'Demand', 'Current_Stock_Start', 'Group', 'error', 'Category']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('Unnamed:')]
    
    target_col = 'Demand'
    
    # Diagnostic print to ensure strings are gone
    # print(f"Features used for training: {feature_cols}")

    # Forecast
    forecast_df = forecast_demand(df, feature_cols, target_col)

    # Inventory recommendations
    inventory_df = generate_inventory_recommendations(
        forecast_df, 
        service_level_z=SERVICE_LEVEL_Z
    )

    # Save outputs
    save_outputs(forecast_df, inventory_df)
    
    print("\n6/6: Pipeline execution complete. Ready for API deployment.")

# --------------------------------
if __name__ == "__main__":
    main()