<h1 align="left"><b>AI Sales Forecasting & Product Demand</b></h1>

An advanced AI-powered Sales Forecasting Dashboard built with Streamlit, combining Bi-Directional LSTM and Transformer Attention architectures to predict future revenue and analyze product demand trends.This interactive dashboard enables businesses to explore sales insights, visualize demand patterns, and generate 30-day AI-based revenue projections.
<h2>Overview</h2><br>

The system enables organizations to:-

1.Monitor revenue KPIs in real time

2.Identify high-performing product categories

4.Analyze temporal demand patterns

5.Generate AI-driven 30-day revenue forecasts

6.Compare category-level future projections

It bridges Deep Learning Forecasting with Business Intelligence visualization in a unified, production-ready interface.
<h2>AI Model Architecture</h2><br>

The forecasting engine combines:

1.Bi-Directional LSTM for capturing temporal dependencies

2.Transformer Attention Block for contextual sequence learning

3.Dense layers for multi-quantile regression output
<h3> Core Features</h3> 

| Category | Capabilities |
|-----------|-------------|
| **Business Intelligence** | Revenue KPIs, Category Leaderboard, Demand Classification, Market Share Visualization, Heatmap |
| **Forecasting Engine** | 80/20 Split, Multi-Quantile Output (P10/P50/P90), 30-Day Projection, Inverse Scaling |
| **Interactive Analytics** | Category Filtering, Comparative Forecasting, Plotly Charts, Dark UI |
<h2 align="left">Architecture Flow</h2>
<p align="left">
Input Data<br>
↓<br>
MinMax Scaling<br>
↓<br>
Sequence Generation<br>
↓<br>
BiLSTM<br>
↓<br>
Transformer Attention<br>
↓<br>
LSTM + Dense Layers<br>
↓<br>
P10 | P50 | P90 Forecast Output<br>
↓<br>
Interactive Dashboard
</p>
<h3>Tech Stack</h3>

| Layer | Technology |
|-------|------------|
| **UI Framework** | Streamlit |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Preprocessing** | Scikit-learn |
| **Language** | Python 3.9+ |
<h2>Installation & Setup</h2>

<b>1. Clone the Repository</b>

<pre>
git clone https://github.com/your-username/ai-sales-forecasting.git
cd ai-sales-forecasting
</pre>

<b>2. Install Dependencies</b>

<pre>
pip install -r requirements.txt
# or install manually
pip install streamlit tensorflow pandas numpy plotly scikit-learn
</pre>

<b>3️. Run the Application</b>

<pre>
streamlit run app.py
</pre>

The application will launch at:

<pre>
http://localhost:8501
</pre>

<h3>Business Applications</h3><Br>

- **Revenue Forecasting & Financial Planning**  
- **Inventory Optimization**  
- **Demand Intelligence**  
- **Executive Decision Support**  
- **AI-Powered Retail Analytics**  

<h3>Roadmap</h3><br>

- Model Persistence & Checkpoint Saving  
- MLOps Pipeline Integration  
- Cloud Deployment (AWS / GCP / Azure)  
- Model Benchmarking (Prophet / XGBoost / LSTM Baseline)  
- Real-Time Streaming Data Integration  

<h3> License</h3><br>
This project is licensed under the **MIT License**.

<h3>Author</h3><br>
Developed as a production-ready **AI Sales Forecasting & Product Demand**.
