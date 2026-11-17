# AI Sales Forecasting & Demand Prediction

A sales forecasting dashboard built with Streamlit. It combines a Bidirectional LSTM and a Transformer Attention block to predict future revenue and analyse product demand trends. The dashboard lets users explore sales insights, view demand patterns and generate 30-day revenue projections.

**Live demo:** [add your Streamlit app link here]

## Overview

The dashboard lets users:

1. Track revenue KPIs
2. Identify high-performing product categories
3. Analyse demand patterns over time
4. Generate 30-day revenue forecasts
5. Compare forecasts across product categories

## Model Architecture

The forecasting model combines:

1. **Bidirectional LSTM** to capture temporal dependencies
2. **Transformer Attention block** for contextual sequence learning
3. **LSTM + Dense layers** with multi-quantile regression output (P10 / P50 / P90)

### Pipeline

```
Input Data (sales_data.csv)
        ↓
MinMax Scaling
        ↓
Sequence Generation
        ↓
Bidirectional LSTM
        ↓
Transformer Attention
        ↓
LSTM + Dense Layers
        ↓
P10 | P50 | P90 Forecast Output
        ↓
Interactive Streamlit Dashboard
```

## Features

| Category | Capabilities |
|---|---|
| **Business Intelligence** | Revenue KPIs, Category Leaderboard, Demand Classification, Market Share Visualisation, Heatmap |
| **Forecasting** | 80/20 train-test split, Multi-Quantile Output (P10/P50/P90), 30-Day Projection, Inverse Scaling |
| **Interactive Analytics** | Category Filtering, Comparative Forecasting, Plotly Charts, Dark UI |

## Results

| Metric | Value |
|---|---|
| RMSE (test set) | ≈ ₹2,880 |
| MAE (test set) | ≈ ₹2,510 |

Average of 3 runs (15 epochs, sequence length 30, 80/20 split, 194 test samples). Exact values vary slightly between runs.

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly |
| Preprocessing | Scikit-learn |
| Language | Python 3.9+ |

## Project Structure

```
├── app.py            # Streamlit dashboard and model code
├── sales_data.csv    # Sales dataset
├── requirements.txt  # Python dependencies
└── README.md
```

## Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/sandraem849-sys/AI-Sales-Forcasting-Demend-Product.git
cd AI-Sales-Forcasting-Demend-Product
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Business Applications

- Revenue forecasting and financial planning
- Inventory optimisation
- Demand intelligence
- Retail analytics

## Roadmap

- Model benchmarking against Prophet, XGBoost and a plain LSTM baseline
- Model persistence and checkpoint saving
- Cloud deployment
- Streaming data integration

## License

This project is licensed under the MIT License.

## Author

Sandra E M | [LinkedIn](https://linkedin.com/in/sandra-e-m-829858349) | [GitHub](https://github.com/sandraem849-sys)
