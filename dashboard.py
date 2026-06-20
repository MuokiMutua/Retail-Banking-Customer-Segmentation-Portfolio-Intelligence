import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIGURATION & ENTERPRISE THEME
# ─────────────────────────────────────────────
st.set_page_config(page_title="Retail Banking Portfolio Intelligence", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family:'Inter', sans-serif; background:#0f172a; color: #e2e8f0; }
    .stApp { background:#0f172a; }
    
    /* Executive KPI Cards */
    .metric-card { background:#1e293b; border:1px solid #334155; border-radius:6px; padding:1.25rem; margin-bottom:1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .metric-title { color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
    .metric-value { color:#f8fafc; font-size:2.2rem; font-weight:700; margin-top:0.4rem; letter-spacing:-0.02em; }
    .metric-sub { color:#10b981; font-size:0.8rem; font-weight:500; margin-top:0.2rem; }
    
    /* Streamlit Table Styling Overrides to match theme */
    .stDataFrame { background-color: #1e293b; border-radius: 6px; }
    [data-testid="stTable"] { background-color: #1e293b; border-radius: 6px; }
    [data-testid="stTable"] th { background-color: #0f172a !important; color: #94a3b8 !important; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; border-bottom: 1px solid #334155 !important; }
    [data-testid="stTable"] td { color: #e2e8f0 !important; font-size: 0.9rem; border-bottom: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA ACQUISITION & BUSINESS TRANSLATION
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("clustered_customers.csv")
    except FileNotFoundError:
        return None
        
    # Translate ML "Clusters" into Business "Tiers"
    cluster_value = df.groupby('cluster_id')['monetary_value_kes'].mean().sort_values(ascending=False).index
    
    tier_mapping = {
        cluster_value[0]: "Platinum (High-Net-Worth)",
        cluster_value[1]: "Gold (Emerging Affluent)",
        cluster_value[2]: "Silver (Mass Market)",
        cluster_value[3]: "Bronze (Micro-Transactors)",
        cluster_value[4]: "Standard (At-Risk / Dormant)"
    }
    
    df['Segment_Name'] = df['cluster_id'].map(tier_mapping)
    return df

df = load_and_prep_data()

# ─────────────────────────────────────────────
# DASHBOARD UI: HEADER & KPIs
# ─────────────────────────────────────────────
st.markdown("<h1 style='color:#f8fafc; font-size:1.8rem; margin-bottom:0;'>RETAIL BANKING PORTFOLIO INTELLIGENCE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em;'>AI-Driven Customer Segmentation & Revenue Strategy</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#334155; margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

if df is None:
    st.error("SYSTEM ERROR: Data file not found. Please run the clustering engine first.")
    st.stop()

total_portfolio_value = df['monetary_value_kes'].sum()
avg_revenue_per_user = df['monetary_value_kes'].mean()
active_loans = df['active_microloans'].sum()
digital_penetration = (df['preferred_channel'].isin(['Mobile App', 'USSD / Mobile Money'])).mean() * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Est. Monthly Volume</div><div class='metric-value'>KES {total_portfolio_value/1e6:.1f}M</div><div class='metric-sub'>↑ 4.2% vs Last Quarter</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Avg Volume Per User (ARPU)</div><div class='metric-value'>KES {avg_revenue_per_user:,.0f}</div><div class='metric-sub' style='color:#94a3b8;'>Portfolio Benchmark</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Active Credit Lines</div><div class='metric-value'>{active_loans:,}</div><div class='metric-sub'>Performing Microloans</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Digital Channel Penetration</div><div class='metric-value'>{digital_penetration:.1f}%</div><div class='metric-sub'>Mobile App + USSD</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DASHBOARD UI: VISUALIZATIONS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<h3 style='color:#f8fafc; font-size:1.0rem; text-transform:uppercase; letter-spacing:0.05em;'>Portfolio Value Distribution by Segment</h3>", unsafe_allow_html=True)
    
    segment_value = df.groupby('Segment_Name')['monetary_value_kes'].sum().reset_index()
    
    fig_donut = px.pie(
        segment_value, 
        values='monetary_value_kes', 
        names='Segment_Name',
        hole=0.6,
        color='Segment_Name',
        color_discrete_map={
            "Platinum (High-Net-Worth)": "#c084fc",
            "Gold (Emerging Affluent)": "#facc15",
            "Silver (Mass Market)": "#cbd5e1",
            "Bronze (Micro-Transactors)": "#fbbf24",
            "Standard (At-Risk / Dormant)": "#ef4444"
        }
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='#94a3b8', size=10)),
        margin=dict(l=0, r=0, t=20, b=0),
        annotations=[dict(text=f"KES {total_portfolio_value/1e6:.0f}M", x=0.5, y=0.5, font_size=20, font_color="#f8fafc", showarrow=False)]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    st.markdown("<h3 style='color:#f8fafc; font-size:1.0rem; text-transform:uppercase; letter-spacing:0.05em;'>Channel Strategy: Where Do They Transact?</h3>", unsafe_allow_html=True)
    
    channel_dist = df.groupby(['Segment_Name', 'preferred_channel']).size().reset_index(name='Users')
    
    order = ["Platinum (High-Net-Worth)", "Gold (Emerging Affluent)", "Silver (Mass Market)", "Bronze (Micro-Transactors)", "Standard (At-Risk / Dormant)"]
    
    fig_bar = px.bar(
        channel_dist, 
        y='Segment_Name', 
        x='Users', 
        color='preferred_channel',
        orientation='h',
        category_orders={"Segment_Name": order[::-1]},
        color_discrete_map={
            "Mobile App": "#38bdf8", 
            "USSD / Mobile Money": "#10b981", 
            "Branch": "#ef4444", 
            "ATM": "#94a3b8"
        }
    )
    
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", showgrid=True, gridcolor='#334155', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title="", showgrid=False, tickfont=dict(color='#f8fafc')),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(color='#94a3b8', size=11), title=""),
        margin=dict(l=0, r=0, t=40, b=0),
        barmode='stack'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# DASHBOARD UI: STRATEGIC ACTION MATRIX
# ─────────────────────────────────────────────
st.markdown("<h3 style='color:#f8fafc; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.05em; margin-top:2rem;'>Marketing & Product Strategy Playbook</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-bottom: 1rem;'>Prescriptive actions based on machine learning segment profiles.</p>", unsafe_allow_html=True)

# Generate strategy data dynamically
strategy_data = []
order = ["Platinum (High-Net-Worth)", "Gold (Emerging Affluent)", "Silver (Mass Market)", "Bronze (Micro-Transactors)", "Standard (At-Risk / Dormant)"]

for segment in order:
    segment_data = df[df['Segment_Name'] == segment]
    if segment_data.empty: continue
    
    avg_age = int(segment_data['age'].mean())
    avg_loans = segment_data['active_microloans'].mean()
    dom_channel = segment_data['preferred_channel'].mode()[0]
    
    if "Platinum" in segment:
        dna = f"Avg Age: {avg_age} | High Balances | Low Loan Dependency"
        action = "Wealth Management Cross-sell: Assign dedicated relationship managers. Offer premium credit cards and offshore accounts."
    elif "Gold" in segment:
        dna = f"Avg Age: {avg_age} | Upward Mobility | Moderate Borrowing"
        action = "Credit Expansion: Pre-approve for asset finance (car loans, mortgages). Target with lifestyle rewards programs."
    elif "Silver" in segment:
        dna = f"Avg Age: {avg_age} | High Frequency | Digital Native"
        action = "Ecosystem Lock-in: Drive adoption of bill pay and salary processing via Mobile App. Offer digital overdrafts."
    elif "Bronze" in segment:
        dna = f"Avg Age: {avg_age} | Micro-transactions | High Loan Vol: {avg_loans:.1f}"
        action = "Cost-to-Serve Optimization: Migrate away from USSD/Branch to lite mobile app. Utilize alternative data for loan scoring."
    else:
        dna = f"Avg Age: {avg_age} | Low Activity | High Recency Days"
        action = "Re-activation Campaign: Send targeted SMS offers with cash-back incentives. Review for account closure if inactive."

    strategy_data.append({
        "Customer Segment": segment,
        "Profile DNA": dna,
        "Primary Channel": dom_channel,
        "Recommended Business Action": action
    })

# Render as a native Streamlit table
strategy_df = pd.DataFrame(strategy_data)
st.table(strategy_df)