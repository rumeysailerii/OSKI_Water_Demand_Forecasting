# OSKI_Water_Demand_Forecasting
End-to-end Machine Learning Dashboard for water demand forecasting, resource optimization, and operational logistics using Streamlit and Scikit-Learn.
# 💧 OSKI Water Demand & Logistics Forecasting Dashboard
## 📌 Project Overview
This project is an interactive, data-driven **Decision Support System (DSS)** developed for the Ordu Water and Sewerage Administration (OSKİ). It transforms complex, static infrastructure data into actionable business insights. By integrating machine learning models, the dashboard forecasts future logistics demands (e.g., water tanker requirements) and categorizes operational zones to optimize resource allocation.

## 🚀 Key Features
* **Dynamic Demand Forecasting:** Utilizes Supervised Learning algorithms (`Random Forest`, `K-Nearest Neighbors`, `Linear Regression`) to predict the 2026 logistical load across major districts (Altınordu, Fatsa, Ünye). 
* **Market Segmentation (Clustering):** Implements `K-Means Clustering` (Unsupervised Learning) combined with StandardScaler to objectively group districts based on their growth rates and subscriber volumes.
* **Interactive Data Visualization:** Features dynamic Bubble Charts, Sunburst diagrams, and time-series line charts built with `Plotly` and `Seaborn` to visualize the correlation between subscriber growth and water consumption.
* **Scenario Analysis:** Includes an interactive control panel (via `Streamlit`) allowing executives to adjust growth thresholds and simulate different operational scenarios in real-time.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest, KNN, Linear Regression, K-Means, Cross-Validation)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly, Matplotlib, Seaborn

## 📊 Business Impact & ROI
Rather than relying on subjective estimations, this tool provides a robust, mathematical foundation for strategic planning. The Random Forest model achieved a highly reliable Cross-Validation accuracy, enabling the administration to:
1. Prevent over-allocation of resources.
2. Minimize operational costs in logistics (tanker dispatching).
3. Prioritize infrastructure investments based on data-driven growth matrices.

## 💻 How to Run Locally
1. Clone the repository: `git clone https://github.com/YourUsername/OSKI_Water_Demand_Forecasting.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m streamlit run dashboard.py`

⚠️Data Privacy Notice: Due to strict Non-Disclosure Agreements (NDA) and corporate data privacy policies of the Ordu Water and Sewerage Administration (OSKİ), the datasets used in this project are strictly confidential and cannot be shared publicly. This repository contains the core Machine Learning pipelines and Streamlit application code, provided solely for portfolio and demonstration purposes.

<img width="628" height="342" alt="image" src="https://github.com/user-attachments/assets/5a78ce03-dc74-4450-bcd9-38b455846cb2" />

<img width="625" height="325" alt="image" src="https://github.com/user-attachments/assets/3f30152e-e435-42ad-b678-6102c0787e56" />
<img width="402" height="221" alt="image" src="https://github.com/user-attachments/assets/d71d6ddb-5f92-45a3-9879-f5165f288110" />
<img width="634" height="316" alt="image" src="https://github.com/user-attachments/assets/39c66365-5612-4a77-b8a1-5ca7e3cd4d19" />
<img width="613" height="295" alt="image" src="https://github.com/user-attachments/assets/772d9f4c-10d6-4b75-95a6-ab5e37c4ad0a" />
<img width="612" height="357" alt="image" src="https://github.com/user-attachments/assets/30f2b1c3-a25e-4038-bece-2fe4aa6d0151" />



---
*Developed by Rumeysa Ileri as part of a comprehensive data science and strategic planning initiative.*



