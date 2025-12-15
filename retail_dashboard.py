import streamlit as st
import pandas as pd
import requests
import altair as alt

st.set_page_config(page_title="AI Inventory Dashboard", layout="wide")
st.title("Retail Inventory Management Dashboard")

# --- Sidebar Configuration ---
st.sidebar.header("Connection Settings")
# Allow user to configure API URL dynamically to handle port mismatches (8000 vs 8001)
base_api_url = st.sidebar.text_input(
    "API Base URL", 
    value="http://127.0.0.1:8001",
    help="Ensure this matches the port uvicorn is running on (default 8001)."
)

# --- Data Fetching ---
@st.cache_data
def fetch_data(api_url):
    """Fetches all forecast and inventory data from the API."""
    forecast_api = f"{api_url}/forecast"
    inventory_api = f"{api_url}/inventory"
    
    try:
        # 1. Fetch Forecast
        forecast_response = requests.get(forecast_api)
        forecast_response.raise_for_status()
        forecast_df = pd.DataFrame(forecast_response.json())
        
        # 2. Fetch Inventory
        inventory_response = requests.get(inventory_api)
        inventory_response.raise_for_status()
        inventory_df = pd.DataFrame(inventory_response.json())
        
        # 3. Data Type Conversions (Defensive)
        if not forecast_df.empty:
            if 'Date' in forecast_df.columns:
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            else:
                st.error(f"Forecast data missing 'Date' column. Columns found: {list(forecast_df.columns)}")

        if not inventory_df.empty:
            if 'Date' in inventory_df.columns:
                inventory_df['Date'] = pd.to_datetime(inventory_df['Date'])
            else:
                st.error(f"Inventory data missing 'Date' column. Columns found: {list(inventory_df.columns)}")
        
        return forecast_df, inventory_df
        
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Connection Refused. Is the API running at `{api_url}`?")
        st.info("💡 Tip: check your terminal to see if uvicorn is running and on which port.")
        return pd.DataFrame(), pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- Load Data ---
# We pass the URL here so the cache invalidates if the user changes the URL
forecast_df, inventory_df = fetch_data(base_api_url)

if forecast_df.empty or inventory_df.empty:
    st.warning("⚠️ No data loaded. Please check the API connection and try again.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🛒 Select Product & Store")

# 1. Store Selector
if 'Store_ID' in forecast_df.columns:
    store_list = sorted(forecast_df['Store_ID'].unique())
    selected_store = st.sidebar.selectbox("Select Store ID", store_list)
else:
    st.error("Error: The loaded data is missing the 'Store_ID' column. Cannot filter.")
    st.stop()


# Filter data by selected store
forecast_store = forecast_df[forecast_df['Store_ID'] == selected_store]
inventory_store = inventory_df[inventory_df['Store_ID'] == selected_store]

# 2. Product Selector (Only show products available in the selected store)
if 'Product_ID' in forecast_store.columns:
    product_list = sorted(forecast_store['Product_ID'].unique())
    selected_product = st.sidebar.selectbox("Select Product ID", product_list)
else:
    st.error("Error: The loaded data is missing the 'Product_ID' column. Cannot filter.")
    st.stop()

# Final filter by product
forecast_final = forecast_store[forecast_store['Product_ID'] == selected_product].sort_values('Date')
inventory_final = inventory_store[inventory_store['Product_ID'] == selected_product].sort_values('Date')


# --- Main Content ---

st.header(f"Store {selected_store} / Product {selected_product}")

# Display Key Metrics
col1, col2, col3 = st.columns(3)

if not inventory_final.empty:
    latest_data = inventory_final.iloc[-1]
    latest_category = latest_data.get('Category', "N/A")

    with col1:
        st.metric("Latest Category", latest_category)

    with col2:
        st.metric("Predicted Next Demand", f"{latest_data.get('Predicted_Demand', 0):.0f} units")

    with col3:
        st.metric("Current Stock Level", f"{latest_data.get('current_stock', 0):.0f} units")
else:
    st.info("No detailed inventory data available for this selection.")


# --- Chart: Demand Forecast & Inventory Levels ---
st.subheader("Demand Forecast & Inventory History")

if not inventory_final.empty and 'Date' in inventory_final.columns:
    # 1. Demand Chart (Predicted vs Actual)
    demand_chart = alt.Chart(forecast_final).mark_line().encode(
        x=alt.X('Date:T', title="Date"),
        y=alt.Y('Predicted_Demand:Q', title="Demand (Units)"),
        color=alt.value("#3498db"),
        tooltip=['Date', 'Predicted_Demand', 'Actual_Demand']
    ).properties(title="Predicted vs Actual Demand")

    actual_demand = alt.Chart(forecast_final).mark_circle(color='red').encode(
        x='Date:T',
        y='Actual_Demand:Q',
        tooltip=['Date', 'Actual_Demand']
    )
    
    # 2. Inventory Chart (Stock vs Reorder Level)
    inventory_base = alt.Chart(inventory_final).encode(
        x=alt.X('Date:T')
    )
    
    stock_line = inventory_base.mark_line(color='#27ae60').encode(
        y=alt.Y('current_stock:Q', title="Stock Level (Units)"),
        tooltip=['Date', 'current_stock', 'reorder_level']
    ).properties(title="Stock Level vs Reorder Threshold")
    
    reorder_line = inventory_base.mark_line(color='#e74c3c', strokeDash=[5, 5]).encode(
        y='reorder_level:Q'
    )
    
    # Combine charts using columns for better display
    st.altair_chart((demand_chart + actual_demand) | (stock_line + reorder_line), use_container_width=True)
elif not inventory_final.empty:
    st.warning("Cannot display charts. Data is loaded but is missing the 'Date' column.")


# --- Inventory Recommendations & Alerts ---
st.subheader("Inventory Recommendations & Restock Alerts")

# Filter for needed restock
restock_alerts = inventory_final[inventory_final['restock_needed']==True]

if not restock_alerts.empty:
    st.error(f"🚨 ACTION REQUIRED: {len(restock_alerts)} restock days detected!")
    
    display_cols = [
        'Date', 'Category', 'Predicted_Demand', 'reorder_level', 
        'current_stock', 'stock_after_sales', 'restock_qty'
    ]
    
    # Filter display columns to only those that exist
    display_cols = [col for col in display_cols if col in restock_alerts.columns]
    
    st.dataframe(
        restock_alerts[display_cols].rename(columns={'restock_qty': 'Restock Quantity', 'current_stock': 'Stock at Start'}), 
        use_container_width=True
    )
else:
    st.success("✅ No immediate restock needed based on current projections.")

with st.expander("View Full Inventory Simulation Data"):
    st.dataframe(inventory_final, use_container_width=True)