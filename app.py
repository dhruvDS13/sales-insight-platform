import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
from io import BytesIO

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')  # Updated to latest stable

# Page config
st.set_page_config(page_title="InsightForge", layout="wide", initial_sidebar_state="expanded")

# === Enhanced Vibrant Custom CSS ===
st.markdown("""
<style>
    /* Main Header - Vibrant Gradient */
    .insightforge-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .insightforge-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Chat Bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1e293b;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        font-weight: 500;
    }

    /* Section Headers */
    h2, h3 {
        color: #4c1d95 !important;
        font-weight: 700;
    }

    /* Metrics Styling */
    .stMetric > label {
        color: #764ba2 !important;
        font-weight: bold;
    }

    /* Sidebar Styling */
    .css-1d391kg { 
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# === Vibrant Header ===
st.markdown("""
<div class="insightforge-header">
    <div class="insightforge-title">
        🔨 InsightForge
    </div>
    <p style="font-size:1.3rem; margin:0; opacity:0.9;">Hammer Sales Data into Gold with AI!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV/Excel file", type=['csv', 'xlsx'])

persona = st.sidebar.selectbox("👤 Business Persona", ["Executive", "Sales Manager", "Analyst"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            content = uploaded_file.read()
            try:
                decoded_content = content.decode('utf-8')
            except:
                decoded_content = content.decode('latin-1', errors='replace')
            df = pd.read_csv(BytesIO(decoded_content.encode('utf-8')))
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        
        required_cols = ['Order Date', 'Sales', 'Profit', 'Category', 'Region', 'Segment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}. Expected: {required_cols}")
        else:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Profit'] = df['Profit'].fillna(0)
            df['Sales'] = df['Sales'].clip(lower=0)
            if 'Quantity' not in df.columns:
                df['Quantity'] = (df['Sales'] / df['Profit'].replace(0, 1)).round(0)
            df = df.dropna(subset=['Sales', 'Profit'])
            
            st.session_state.df_clean = df
            st.success(f"✅ Loaded {len(df):,} rows from {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    except Exception as e:
        st.error(f"Upload failed: {e}")

# Main App
if 'df_clean' in st.session_state:
    df = st.session_state.df_clean
    
    # Filters
    st.markdown("### 📅 Filters")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['Order Date'].min().date())
    with col2:
        end_date = st.date_input("End Date", df['Order Date'].max().date())
    
    df_filtered = df[(df['Order Date'] >= pd.to_datetime(start_date)) & (df['Order Date'] <= pd.to_datetime(end_date))]
    
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Region", options=df['Region'].unique(), default=df['Region'].unique())
    with col2:
        category_filter = st.multiselect("Category", options=df['Category'].unique(), default=df['Category'].unique())
    
    df_filtered = df_filtered[
        df_filtered['Region'].isin(region_filter) & 
        df_filtered['Category'].isin(category_filter)
    ]

    @st.cache_data
    def analyze_data(df_filt):
        analysis = {}
        analysis['total_sales'] = df_filt['Sales'].sum()
        analysis['total_profit'] = df_filt['Profit'].sum()
        analysis['avg_profit_margin'] = (analysis['total_profit'] / analysis['total_sales'] * 100) if analysis['total_sales'] > 0 else 0

        df_filt['Month'] = df_filt['Order Date'].dt.to_period('M')
        monthly_sales = df_filt.groupby('Month')['Sales'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)
        analysis['monthly_sales'] = monthly_sales

        group_col = 'Sub-Category' if 'Sub-Category' in df_filt.columns else 'Category'
        top_products = df_filt.groupby(group_col)['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        top_declines = df_filt.groupby(group_col)['Sales'].sum().sort_values().head(3).reset_index()
        analysis['top_products'] = top_products
        analysis['top_declines'] = top_declines
        analysis['group_col'] = group_col

        analysis['region_perf'] = df_filt.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
        analysis['category_contrib'] = df_filt.groupby('Category')['Sales'].sum() / analysis['total_sales'] * 100
        analysis['region_contrib'] = df_filt.groupby('Region')['Sales'].sum() / analysis['total_sales'] * 100

        mean_monthly = monthly_sales['Sales'].mean()
        std_monthly = monthly_sales['Sales'].std()
        anomalies = monthly_sales[abs(monthly_sales['Sales'] - mean_monthly) > 2 * std_monthly]
        analysis['anomalies'] = anomalies if not anomalies.empty else None

        return analysis

    analysis = analyze_data(df_filtered)

    # === Colorful Dashboard ===
    st.markdown("### 📊 Performance Dashboard")

    # Persona-based KPIs with colors
    colors = ["#667eea", "#764ba2", "#f093fb"]
    if persona == "Executive":
        cols = st.columns(3)
        cols[0].metric("Total Sales", f"${analysis['total_sales']:,.0f}", delta=None)
        cols[1].metric("Total Profit", f"${analysis['total_profit']:,.0f}", delta=None)
        cols[2].metric("Profit Margin", f"{analysis['avg_profit_margin']:.1f}%", delta=None)
    elif persona == "Sales Manager":
        cols = st.columns(3)
        top_region = analysis['region_perf'].loc[analysis['region_perf']['Sales'].idxmax()]
        cols[0].metric("Top Region", f"{top_region['Region']}", f"${top_region['Sales']:,.0f}")
        cols[1].metric("Top Products", len(analysis['top_products']))
        cols[2].metric("Anomalies Detected", len(analysis['anomalies']) if analysis['anomalies'] is not None else 0)
    else:
        cols = st.columns(3)
        cols[0].metric("Avg Monthly Sales", f"${analysis['monthly_sales']['Sales'].mean():,.0f}")
        cols[1].metric("Highest Month", analysis['monthly_sales'].loc[analysis['monthly_sales']['Sales'].idxmax(), 'Month'])
        cols[2].metric("Profit Margin", f"{analysis['avg_profit_margin']:.1f}%")

    # === Vibrant Charts ===
    st.markdown("### 📈 Visual Insights")

    row1 = st.columns(2)
    with row1[0]:
        fig_trend = px.line(
            analysis['monthly_sales'], x='Month', y='Sales',
            title="Sales Trend Over Time",
            color_discrete_sequence=['#667eea']
        )
        fig_trend.update_layout(template="plotly_white", title_font_size=18, hovermode="x unified")
        fig_trend.update_traces(line=dict(width=4))
        st.plotly_chart(fig_trend, use_container_width=True)

    with row1[1]:
        fig_bar = px.bar(
            analysis['top_products'],
            x='Sales', y=analysis['group_col'],
            orientation='h',
            title="Top 10 Products by Sales",
            color='Sales',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    row2 = st.columns(2)
    with row2[0]:
        fig_region = px.bar(
            analysis['region_perf'],
            x='Region', y=['Sales', 'Profit'],
            barmode='group',
            title="Region Performance",
            color_discrete_sequence=['#a8edea', '#fed6e3']
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with row2[1]:
        fig_pie_cat = px.pie(
            values=analysis['category_contrib'].values,
            names=analysis['category_contrib'].index,
            title="Sales by Category",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_pie_cat, use_container_width=True)

    # Anomalies
    if analysis['anomalies'] is not None:
        st.warning("⚠️ Sales Anomalies Detected!")
        fig_anom = px.bar(analysis['anomalies'], x='Month', y='Sales', color='Sales', color_continuous_scale='Reds')
        st.plotly_chart(fig_anom, use_container_width=True)

    # === AI Insights ===
    st.markdown("### 🤖 AI-Powered Insights")
    if st.button("✨ Generate Insights", type="primary"):
        top_decline = analysis['top_declines'][analysis['group_col']].tolist()
        top_growth = analysis['top_products'].head(3)[analysis['group_col']].tolist()
        anomalies = analysis['anomalies']['Month'].astype(str).tolist() if analysis['anomalies'] is not None else ["None"]

        summary = f"Sales: ${analysis['total_sales']:,.0f}, Profit: ${analysis['total_profit']:,.0f}, Margin: {analysis['avg_profit_margin']:.1f}%. Top declines: {', '.join(top_decline)}. Growth drivers: {', '.join(top_growth)}. Anomalies: {', '.join(anomalies)}."

        if persona == "Executive":
            prompt = f"Executive summary: {summary}. Provide 3 high-impact strategic actions. Concise, bold, visionary."
        elif persona == "Sales Manager":
            prompt = f"Sales manager briefing: {summary}. Give 3 immediate, actionable tactics with owners."
        else:
            prompt = f"Deep analysis: {summary}. Explain drivers, correlations, and 3 data-backed recommendations."

        prompt += " Use bullet points. Be insightful and specific."

        with st.spinner("Generating insights..."):
            try:
                response = model.generate_content(prompt)
                st.markdown("**🔥 AI Insights**")
                st.markdown(f"<div class='ai-bubble'>{response.text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI failed: {e}")

    # === Colorful Chat Interface ===
    st.markdown("### 💬 Ask Your Data Anything")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Type your question (e.g., 'Why did Tech sales drop in Q4?')", key="query")
    
    col1, col2 = st.columns([1, 6])
    with col1:
        ask_btn = st.button("Send 🚀", type="primary")
    with col2:
        show_data = st.checkbox("Show data used for answer")

    if ask_btn and user_query:
        context = f"Date range: {start_date} to {end_date}. Regions: {region_filter}. Categories: {category_filter}. "
        context += f"Total Sales: ${analysis['total_sales']:,.0f}. Sample data: {df_filtered.head(20).to_json(orient='records')[:1500]}..."

        chat_prompt = f"""
        You are an expert sales analyst. Answer: "{user_query}"
        Use this context: {context}
        Be precise, reference actual numbers/regions/categories. If unsure, say 'Not enough data'.
        Suggest follow-up actions if relevant.
        """

        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(chat_prompt)
                ai_answer = response.text

                st.session_state.chat_history.append({
                    "user": user_query,
                    "ai": ai_answer,
                    "data": context[:600] + "..."
                })

            except Exception as e:
                st.error("Chat failed. Check API key.")

    # Display colorful chat history
    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history[-8:]:
        st.markdown(f"<div class='user-bubble'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-bubble'><strong>AI:</strong> {chat['ai']}</div>", unsafe_allow_html=True)
        if show_data:
            st.caption(f"Data used: {chat.get('data', '')}")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Upload a sales dataset (CSV/Excel) to unlock the full power of InsightForge!")
    st.markdown("### 🚀 Try with Sample Data")
    sample_data = {
        'Order Date': pd.date_range('2023-01-01', periods=12, freq='M'),
        'Sales': [22000, 28000, 25000, 32000, 38000, 35000, 40000, 42000, 39000, 45000, 48000, 52000],
        'Profit': [3000, 4200, 3800, 5800, 7200, 6500, 8000, 8500, 7800, 9200, 9800, 11000],
        'Category': ['Technology', 'Furniture', 'Office Supplies'] * 4,
        'Region': ['West', 'East', 'Central', 'South'] * 3,
        'Segment': ['Consumer', 'Corporate', 'Home Office'] * 4,
        'Sub-Category': ['Phones', 'Chairs', 'Binders', 'Machines', 'Tables', 'Storage'] * 2
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)