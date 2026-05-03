import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ─── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    alerts  = pd.read_csv("alerts.csv")
    return results, alerts

results, alerts = load_data()

# ─── Header ────────────────────────────────────────────────
st.title("🛡️ Banking Fraud Detection Dashboard")
st.markdown("Real-time transaction monitoring powered by XGBoost · ROC-AUC: **0.976**")
st.divider()

# ─── Top KPI cards ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total        = len(results)
flagged      = len(alerts)
actual_fraud = int(results['actual_fraud'].sum())
precision    = round(alerts['actual_fraud'].sum() / len(alerts) * 100, 1)

col1.metric("Total Transactions", f"{total:,}")
col2.metric("High Risk Flagged",  f"{flagged}",  delta=f"{round(flagged/total*100,2)}%")
col3.metric("Actual Fraud Found", f"{actual_fraud}")
col4.metric("Alert Precision",    f"{precision}%")

st.divider()

# ─── Row 1: Risk distribution + Fraud probability ──────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Transaction Risk Distribution")
    risk_counts = results['risk_label'].str.replace(
        r'[^\w\s]','', regex=True).str.strip().value_counts()
    colors = {'HIGH RISK':'#e74c3c','MEDIUM RISK':'#f39c12','LOW RISK':'#2ecc71'}
    fig1 = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=colors,
        labels={'x':'Risk Level','y':'Count'},
    )
    fig1.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Fraud Probability Distribution")
    fig2 = px.histogram(
        results, x='fraud_probability',
        nbins=60, color_discrete_sequence=['#3498db']
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ─── Row 2: Risk score comparison + Fraud over time ────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Risk Score: Fraud vs Normal")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=results[results['actual_fraud']==1]['risk_score'],
        name='Actual Fraud', marker_color='#e74c3c', opacity=0.75, nbinsx=30
    ))
    fig3.add_trace(go.Histogram(
        x=results[results['actual_fraud']==0]['risk_score'],
        name='Normal', marker_color='#2ecc71', opacity=0.5, nbinsx=30
    ))
    fig3.update_layout(barmode='overlay', height=350)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Top 10 Fraud Indicators (V-features)")
    features = ['V14','V4','V12','V8','V13','V18','V9','V26','V1','V3']
    scores   = [0.536, 0.049, 0.039, 0.033, 0.022, 0.018, 0.015, 0.013, 0.011, 0.009]
    fig4 = px.bar(
        x=scores, y=features, orientation='h',
        color=scores, color_continuous_scale='Reds',
        labels={'x':'Importance','y':'Feature'}
    )
    fig4.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ─── Live Alerts Table ─────────────────────────────────────
st.subheader("🚨 Live Fraud Alerts")

col_filter1, col_filter2 = st.columns([1,3])
with col_filter1:
    min_score = st.slider("Minimum Risk Score", 0, 100, 75)

filtered = alerts[alerts['risk_score'] >= min_score].copy()
filtered  = filtered.sort_values('risk_score', ascending=False)

st.markdown(f"Showing **{len(filtered)}** flagged transactions")

display_cols = ['risk_score','risk_label','fraud_probability','actual_fraud']
available    = [c for c in display_cols if c in filtered.columns]

display_df = filtered[available].head(50).copy()
display_df['fraud_probability'] = display_df['fraud_probability'].map('{:.1%}'.format)
display_df['risk_score']        = display_df['risk_score'].map('{:.0f}'.format)

st.dataframe(display_df, use_container_width=True, height=400)

st.divider()

# ─── Transaction Explorer ──────────────────────────────────
st.subheader("🔍 Transaction Explorer")
st.markdown("Search any transaction by its index")

txn_id = st.number_input("Transaction Index", min_value=0,
                          max_value=len(results)-1, value=0)
row = results.iloc[txn_id]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Score",        f"{row['risk_score']:.0f}/100")
c2.metric("Fraud Probability", f"{row['fraud_probability']:.1%}")
c3.metric("Risk Label",        str(row['risk_label']).replace('🔴','').replace('🟡','').replace('🟢','').strip())
c4.metric("Actual Fraud",      "YES ⚠️" if row['actual_fraud']==1 else "NO ✅")

st.divider()
st.caption("Built with Python · XGBoost · Streamlit · Plotly | Fraud Detection Project 2026")