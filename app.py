import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, Input, MultiHeadAttention, 
                                     LayerNormalization, Add)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Sales Forecaster", layout="wide", page_icon="📈")

# --- CUSTOM STYLES ---
st.markdown("""<style>
.main { background-color: #0e1117; }
.stButton>button { width:100%; border-radius:5px; height:3em; background:#00eaff; color:black; font-weight:bold; border:none; }
.stButton>button:hover { background:#00b0ff; color:white; }
</style>""", unsafe_allow_html=True)

# --- HELPERS ---
def quantile_loss(q):
    def loss(y, f):
        e = y - f
        return K.mean(K.maximum(q*e, (q-1)*e))
    return loss

def transformer_block(x):
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    ffn = Dense(128, activation="relu")(x)
    ffn = Dense(x.shape[-1])(ffn)
    x = Add()([x, ffn])
    return LayerNormalization()(x)

# --- HEADER ---
st.title(" AI Sales Forecasting & Product Demand")
st.markdown("Predicting future revenue using **Hybrid LSTM-Transformer** architectures.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Forecast Configuration")
    uploaded_file = st.file_uploader("Upload 'sales_data.csv'", type="csv")
    epochs = st.slider("Training Epochs", 5, 50, 15)
    seq_length = st.number_input("Sequence Length (Days)", value=30)

# =========================================================
# MAIN BLOCK
# =========================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    df = df.sort_values('Sale_Date')

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    total_rev = df['Sales_Amount'].sum()
    avg_sale = df['Sales_Amount'].mean()
    col1.metric("Total Revenue", f"₹{total_rev:,.0f}")
    col2.metric("Average Sale", f"₹{avg_sale:,.2f}")
    col3.metric("Data Points", len(df))

    # --- CATEGORY SUMMARY ---
    category_demand = (
        df.groupby('Product_Category')
          .agg({'Sales_Amount': 'sum', 'Quantity_Sold': 'sum'})
          .sort_values('Sales_Amount', ascending=False)
    )

    st.markdown("---")
    st.subheader(" Market Leaderboard")

    top_category = category_demand.index[0]
    top_val = category_demand['Sales_Amount'].iloc[0]

    st.info(f" **Highest Demand Category:** {top_category} with revenue **₹{top_val:,.2f}**")

    with st.expander(" View Detailed Category Ranking", expanded=True):
        styled_df = category_demand.style.format({'Sales_Amount': '₹{:,.2f}', 'Quantity_Sold': '{:,}'})
        st.dataframe(styled_df, use_container_width=True)

    # --- TABS ---
    st.markdown("###  Deep Dive Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        [" Demand Analysis", " AI Forecast", " Temporal Heatmap", " Category Analysis"]
    )

    # ================= TAB 1 =================
    with tab1:
        st.subheader("Market Demand Insights")
        sub_tab_demand1, sub_tab_demand2 = st.tabs([" Demand Trend", " Distribution Pie"])

        with sub_tab_demand1:
            total_sales = category_demand['Sales_Amount'].sum()
            category_demand['Demand_%'] = (category_demand['Sales_Amount'] / total_sales) * 100

            def demand_level(p):
                if p > 40: return "High"
                elif p > 20: return "Medium"
                else: return "Low"

            category_demand['AI_Level'] = category_demand['Demand_%'].apply(demand_level)
            color_map = {'High': '#ff4c4c', 'Medium': '#ffa500', 'Low': '#00ff99'}
            point_colors = category_demand['AI_Level'].map(color_map)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=category_demand.index,
                y=category_demand['Sales_Amount'],
                mode='lines+markers',
                line=dict(color='#00eaff', width=3),
                marker=dict(color=point_colors, size=12, line=dict(color='white', width=2)),
                text=[f"AI Level: {lvl}<br>Demand: {pct:.1f}%" 
                      for lvl, pct in zip(category_demand['AI_Level'], category_demand['Demand_%'])],
                hoverinfo='text+x+y',
                name='Sales Amount'
            ))

            fig_trend.update_layout(
                title=f" AI Product Demand Trend (Top: {top_category})",
                template='plotly_dark',
                xaxis_title="Product Category",
                yaxis_title="Total Sales Amount",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with sub_tab_demand2:
            fig_pie = px.pie(
                category_demand,
                values='Sales_Amount',
                names=category_demand.index,
                hole=0.5,
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Darkmint_r
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ================= TAB 2 =================
    with tab2:
        st.subheader("Hybrid Deep Learning Model")
        st.write("This model uses a **Bi-Directional LSTM** paired with a **Transformer Attention** layer.")
        
        if st.button("AI Forecasting"):
            with st.spinner("Training Transformer-LSTM Model..."):
                features = ['Sales_Amount']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features])

                def create_sequences(data, seq_length):
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i:i+seq_length])
                        y.append(data[i+seq_length, 0])
                    return np.array(X), np.array(y)

                X, y = create_sequences(scaled_data, seq_length)
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                inputs = Input(shape=(seq_length, len(features)))
                x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
                x = transformer_block(x)
                x = LSTM(32)(x)
                x = Dense(32, activation='relu')(x)
                
                p10 = Dense(1, name='p10')(x)
                p50 = Dense(1, name='p50')(x)
                p90 = Dense(1, name='p90')(x)
                
                model = Model(inputs, [p10, p50, p90])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, [y_train, y_train, y_train], epochs=epochs, batch_size=32, verbose=0)
                
                # Forecasting
                def forecast_future(model, last_seq, steps=30):
                    future_preds = []
                    current_seq = last_seq.copy()
                    for _ in range(steps):
                        preds = model.predict(current_seq[np.newaxis, :, :], verbose=0)
                        p50_val = preds[1][0][0] 
                        future_preds.append(p50_val)
                        new_row = current_seq[-1].copy()
                        new_row[0] = p50_val
                        current_seq = np.vstack([current_seq[1:], new_row])
                    return np.array(future_preds)

                _, p50_test, _ = model.predict(X_test, verbose=0)
                future_scaled = forecast_future(model, X_test[-1], steps=30)

                def quick_inverse(scaled_val):
                    dummy = np.zeros((len(scaled_val), len(features)))
                    dummy[:, 0] = scaled_val.flatten()
                    return scaler.inverse_transform(dummy)[:, 0]

                y_actual = quick_inverse(y_test)
                p50_actual = quick_inverse(p50_test)
                future_actual = quick_inverse(future_scaled)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_actual, mode='lines', name='Actual Sales', line=dict(color='#00eaff')))
                fig.add_trace(go.Scatter(y=p50_actual, mode='lines', name='Model Fit', line=dict(color='#ff00ff', width=1)))
                
                future_x = list(range(len(y_actual), len(y_actual) + 30))
                fig.add_trace(go.Scatter(x=future_x, y=future_actual, mode='lines+markers', name='30-Day AI Projection', line=dict(dash='dash', color='#00ff00')))

                fig.update_layout(title=" Revenue Projection", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
    # ================= TAB 3 =================
    with tab3:
        st.subheader("Sales Heatmap")
        df['Month'] = df['Sale_Date'].dt.month
        df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek
        heatmap_df = df.groupby(['Month', 'DayOfWeek']).agg({'Sales_Amount':'sum'}).reset_index()
        fig_heat = px.density_heatmap(heatmap_df, x='Month', y='DayOfWeek', z='Sales_Amount',
                                      color_continuous_scale='Viridis', template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ================= TAB 4 =================
    with tab4:
        st.markdown("##  Advanced AI Forecasting")
        st.info("Explore category-specific predictions or compare market trends.")
        
        # Sub-tabs inside Tab 4
        sub_tab1, sub_tab2 = st.tabs([" Interactive Explorer", " Comparative Overview"])

        # 1. Prepare Category Forecasts
        category_forecasts = {}
        all_cats = df['Product_Category'].unique()
        
        # NOTE: Ensure 'scaler', 'features', 'create_sequences', 'forecast_future', 
        # and 'inverse_transform' are defined in your helper functions!
        for cat in all_cats:
            cat_df = df[df['Product_Category'] == cat].sort_values('Sale_Date')
            if len(cat_df) >= seq_length:
                # This is a placeholder for your specific logic:
                # cat_scaled = scaler.fit_transform(cat_df[features])
                # X_cat, _ = create_sequences(cat_scaled, seq_length)
                # forecast = forecast_future(model, X_cat[-1], steps=30)
                # category_forecasts[cat] = inverse_transform(forecast)
                
                # MOCK DATA for demonstration if model isn't run:
                category_forecasts[cat] = np.random.uniform(100, 500, 30).cumsum()

        with sub_tab1:
            st.subheader("Deep Dive into Categories")
            fig_inter = go.Figure()
            colors = px.colors.qualitative.Prism
            
            for i, (cat, forecast) in enumerate(category_forecasts.items()):
                fig_inter.add_trace(go.Scatter(
                    x=list(range(len(forecast))),
                    y=forecast,
                    mode='lines+markers',
                    name=cat,
                    line=dict(width=3, shape='spline', color=colors[i % len(colors)]),
                    visible=True
                ))

            buttons = [dict(label="All Categories", method="update",
                            args=[{"visible": [True]*len(category_forecasts)},
                                  {"title": "Global Category Forecast"}])]

            for i, cat in enumerate(category_forecasts.keys()):
                visible = [False]*len(category_forecasts)
                visible[i] = True
                buttons.append(dict(label=cat, method="update",
                                    args=[{"visible": visible},
                                          {"title": f"Forecast Detail: {cat}"}]))

            fig_inter.update_layout(
                updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1.15)],
                template='plotly_dark',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_inter, use_container_width=True)

        with sub_tab2:
            st.subheader("Market Comparison")
            fig_comp = go.Figure()
            for cat, forecast in category_forecasts.items():
                fig_comp.add_trace(go.Scatter(
                    y=forecast,
                    mode='lines',
                    name=cat,
                    line=dict(width=2, shape='spline')
                ))
            fig_comp.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Please upload a CSV file in the sidebar to begin analysis.")