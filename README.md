# Retail-Banking-Customer-Segmentation-and-Portfolio-Intelligence
<img width="980" height="738" alt="image" src="https://github.com/user-attachments/assets/f973f9b3-5b33-4f04-8990-063230c92f6d" />
<img width="978" height="465" alt="image" src="https://github.com/user-attachments/assets/2d47e7e8-ed52-4934-80f9-bfd1a6ca495d" />


An end-to-end Machine Learning pipeline and executive dashboard designed to solve a critical retail banking problem: moving away from blanket marketing to hyper-targeted, data-driven customer strategies.

This project synthesizes realistic banking behavior (RFM metrics), applies K-Means clustering to discover hidden customer segments, and translates those mathematical clusters into an interactive, actionable dashboard for a Chief Marketing Officer (CMO) or Head of Retail Banking.

## Business Problem

In retail banking, treating a high-net-worth individual the same as a micro-transacting student wastes marketing spend and accelerates customer churn. By analyzing behavioral data (Recency, Frequency, Monetary value, and Channel preferences), banks can predict needs, optimize cost-to-serve, and offer the right product at the right time.

## System Architecture

This project is broken down into three core components:

1. **rfm_pipeline.py** (Data Engineering & Synthesis)

* Generates a realistic, 8,000-row synthetic dataset of retail banking customers.

* Models complex distributions for Recency, Frequency, Monetary Value, Age, and Microloan dependency based on real-world mobile money and branch banking trends.

2. **clustering_engine.py** (Machine Learning)

* Preprocesses and encodes categorical features.

* Applies StandardScaler to ensure monetary values don't overpower frequency/recency metrics.

* Uses K-Means Clustering to mathematically group the portfolio into 5 distinct behavioral segments.

3. **segmentation_dashboard.py** (Business Intelligence)

* A high-performance Streamlit application.

* Translates raw ML clusters into enterprise banking tiers (Platinum, Gold, Silver, Bronze, Standard).

* Visualizes channel strategy using Plotly (Donut charts, stacked horizontal bars).

* Provides a dynamic "Strategic Action Playbook" mapping specific marketing actions to each segment.

## Tech Stack

Language: Python 3.x

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (K-Means, StandardScaler, MinMaxScaler)

## Dashboard Highlights

* **Executive KPIs**: Real-time tracking of Total Portfolio Volume, ARPU (Average Revenue Per User), Active Credit Lines, and Digital Penetration.

* **Portfolio Distribution**: Visualizes the exact revenue concentration across the 5 wealth/behavioral tiers.

* **Channel Strategy**: Maps exactly where each segment prefers to transact (Mobile App vs. USSD vs. Branch).

* **Actionable Playbook**: Dynamically generates strategic recommendations (e.g., "Migrate Bronze tier away from expensive branch networks to the lite mobile app").

Visualization: Plotly (Plotly Express & Graph Objects)

Frontend/Dashboard: Streamlit
