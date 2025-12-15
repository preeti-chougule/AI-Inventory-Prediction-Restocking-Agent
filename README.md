# AI-Inventory-Prediction-Restocking-Agent
An end-to-end AI-driven inventory forecasting and restocking system that transforms raw retail sales data into predictive insights and actionable recommendations. This project integrates machine learning, FastAPI, and a Streamlit dashboard to enable proactive, data-driven inventory management.

🚀 Project Overview

Retail inventory decisions are often reactive, leading to stockouts, overstocking, and lost revenue. This project introduces a predictive inventory agent that:

Forecasts future demand

Detects low-stock risks early

Recommends optimal reorder quantities

Visualizes insights through an interactive dashboard

The system demonstrates how ML pipelines and GenAI-style automation can be deployed in a real-world business scenario.



🧠 Core Features

Sales Data Ingestion – Automated ingestion and preprocessing of retail inventory data

Demand Forecasting – ML-based demand prediction using historical sales trends

Low-Stock Detection – Identifies products nearing stockout thresholds

Restocking Recommendations – Suggests when and how much to reorder

API-Driven Architecture – Clean separation between ML, backend, and UI layers

Interactive Dashboard – Business-friendly Streamlit UI for decision-making



🏗️ System Architecture
Raw Sales Data (CSV)
        ↓
ML Pipeline (Training + Forecasting)
        ↓
Forecast & Recommendation CSVs
        ↓
FastAPI Backend
        ↓
Streamlit Dashboard (Visualization & Alerts)



📂 Repository Structure
AI-Inventory-Agent/
│
├── retail_store_inventory.csv        # Input dataset
├── retail_main.py                    # ML pipeline (training + forecasting)
├── retail_api.py                     # FastAPI backend
├── retail_dashboard.py               # Streamlit dashboard
├── AI Inventory Agent Streamlit App.docx
├── AI Inventory Final.pdf             # Project documentation
├── requirements.txt                  # Python dependencies
└── README.md                          # Project documentation



⚙️ Tech Stack

Programming: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn (forecasting models)

Backend: FastAPI, Uvicorn

Frontend / UI: Streamlit

Visualization: Altair



🧪 How to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/your-username/AI-Inventory-Agent.git
cd AI-Inventory-Agent

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the ML Pipeline

This step trains the model and generates forecast & recommendation files.

python retail_main.py

Expected output:

RetailInventory_Demand_Forecast.csv

RetailInventory_Recommendations.csv

4️⃣ Start the FastAPI Backend

uvicorn retail_api:app --reload --port 8001

API will be available at:

http://127.0.0.1:8001

5️⃣ Launch the Streamlit Dashboard

streamlit run retail_dashboard.py

A browser window will automatically open with the interactive dashboard.



📊 Dashboard Capabilities

Store & product-level filtering

Actual vs predicted demand visualization

Stock level vs reorder threshold tracking

Automated low-stock alerts

Data-driven restocking recommendations



💡 Business Impact

Reduces stockouts and excess inventory

Enables proactive replenishment decisions

Improves operational efficiency

Demonstrates applied ML for real-world retail analytics



📌 Future Enhancements

Real-time data ingestion

Cloud deployment (AWS / GCP)

Advanced deep learning forecasting (LSTM)

Authentication & role-based access


