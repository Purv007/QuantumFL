"""
QuantumShieldFL Dashboard — Streamlit Visualization

Multi-page dashboard that visualizes:
1. Training Progress — Accuracy & loss curves
2. QKD Status — QBER, key generation events
3. Security Monitor — Risk scores, adaptive decisions
4. Communication Cost — Bytes, overhead, timing
5. Experiment Comparison — Side-by-side metrics table
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from metrics.db import MetricsDB


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumShieldFL Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .main { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%); }

    /* ── Headings ── */
    h1 {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 700 !important; letter-spacing: -0.5px;
    }
    h2, h3 { color: #c7d2fe !important; font-weight: 600 !important; }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,34,56,0.9) 0%, rgba(22,33,62,0.9) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px; padding: 20px 16px;
        box-shadow: 0 4px 24px rgba(99,102,241,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(12px); transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 8px 32px rgba(99,102,241,0.15);
        transform: translateY(-2px);
    }
    [data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.85rem !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-weight: 500 !important; }

    /* ── Sidebar ── */
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #131a2e 0%, #0c1220 100%);
        border-right: 1px solid rgba(99,102,241,0.15);
    }
    div[data-testid="stSidebarContent"] .stRadio label {
        color: #94a3b8 !important; padding: 6px 12px !important;
        border-radius: 8px; transition: all 0.2s ease;
    }
    div[data-testid="stSidebarContent"] .stRadio label:hover { background: rgba(99,102,241,0.1); color: #c7d2fe !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid rgba(99,102,241,0.15); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 10px 10px 0 0;
        color: #94a3b8; font-weight: 500; padding: 10px 20px;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #c7d2fe; background: rgba(99,102,241,0.08); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(167,139,250,0.1) 100%) !important;
        color: #a78bfa !important; border-bottom: 2px solid #818cf8;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 10px 28px !important;
        font-weight: 600 !important; font-size: 0.9rem !important;
        transition: all 0.3s ease !important; box-shadow: 0 4px 16px rgba(99,102,241,0.3);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 24px rgba(99,102,241,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important; overflow: hidden;
    }

    /* ── Alerts ── */
    .stAlert { border-radius: 12px !important; border: none !important; }
    div[data-testid="stNotification"] { border-radius: 12px !important; }

    /* ── Sliders ── */
    .stSlider > div > div { color: #94a3b8 !important; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: rgba(30,34,56,0.6) !important;
        border-radius: 12px !important; color: #c7d2fe !important;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(99,102,241,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_metrics():
    """Load all experiment metrics from JSONL files OR SQLite database."""
    results_dir = config.RESULTS_DIR
    all_data = {}

    # Try JSONL files first
    if os.path.exists(results_dir):
        for filepath in glob.glob(os.path.join(results_dir, "*_metrics.jsonl")):
            exp_name = os.path.basename(filepath).replace("_metrics.jsonl", "")
            rounds = []
            clients = []
            keys = []
            attacks = []

            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get("type") == "round":
                            rounds.append(entry)
                        elif entry.get("type") == "client":
                            clients.append(entry)
                        elif entry.get("type") == "key_event":
                            keys.append(entry)
                        elif entry.get("type") == "attack_event":
                            attacks.append(entry)
            except Exception:
                continue

            all_data[exp_name] = {
                "rounds": pd.DataFrame(rounds) if rounds else pd.DataFrame(),
                "clients": pd.DataFrame(clients) if clients else pd.DataFrame(),
                "keys": pd.DataFrame(keys) if keys else pd.DataFrame(),
                "attacks": pd.DataFrame(attacks) if attacks else pd.DataFrame(),
            }

    # Fall back to SQLite database if no JSONL data
    if not all_data:
        try:
            db = MetricsDB()
            experiments = db.get_all_experiments()
            cursor = db.conn.cursor()

            for exp in experiments:
                exp_name = exp["name"]
                if exp_name == "playground":
                    continue  # skip playground data for main pages

                # Rounds
                cursor.execute(
                    "SELECT * FROM rounds WHERE experiment_id = ? ORDER BY round_number",
                    (exp["id"],)
                )
                round_rows = [dict(r) for r in cursor.fetchall()]
                # Rename round_number -> round for consistency
                for r in round_rows:
                    r["round"] = r.pop("round_number", r.get("round"))

                # Client rounds
                cursor.execute(
                    "SELECT * FROM client_rounds WHERE experiment_id = ? ORDER BY round_number, client_id",
                    (exp["id"],)
                )
                client_rows = [dict(r) for r in cursor.fetchall()]
                for c in client_rows:
                    c["round"] = c.pop("round_number", c.get("round"))

                # Key events
                cursor.execute(
                    "SELECT * FROM key_events WHERE experiment_id = ? ORDER BY round_number",
                    (exp["id"],)
                )
                key_rows = [dict(r) for r in cursor.fetchall()]
                for k in key_rows:
                    k["round"] = k.pop("round_number", k.get("round"))

                all_data[exp_name] = {
                    "rounds": pd.DataFrame(round_rows) if round_rows else pd.DataFrame(),
                    "clients": pd.DataFrame(client_rows) if client_rows else pd.DataFrame(),
                    "keys": pd.DataFrame(key_rows) if key_rows else pd.DataFrame(),
                    "attacks": pd.DataFrame(),
                }
            db.close()
        except Exception:
            pass

    return all_data

def hex_to_rgba(hex_color, opacity=0.12):
    """Convert hex color to rgba string for Plotly fill."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


@st.cache_data(ttl=30)
def load_db_data():
    """Load comparison data from SQLite."""
    try:
        db = MetricsDB()
        experiments = db.get_all_experiments()
        comparison = db.get_comparison_data()

        all_rounds = {}
        for exp in experiments:
            rounds = db.get_experiment_rounds(exp["id"])
            all_rounds[exp["name"]] = pd.DataFrame(rounds) if rounds else pd.DataFrame()

        db.close()
        return experiments, comparison, all_rounds
    except Exception:
        return [], [], {}


# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────
COLORS = {
    "no_encryption": "#ff6b6b",
    "fixed_qkd": "#4ecdc4",
    "adaptive_qkd": "#667eea",
    "accent": "#764ba2",
    "warning": "#ffa726",
    "danger": "#ef5350",
    "success": "#66bb6a",
    "bg": "#0e1117",
    "card": "#1a1a2e",
}

POLICY_NAMES = {
    "no_encryption": "No Encryption",
    "fixed_qkd": "Fixed QKD",
    "adaptive_qkd": "Adaptive QKD",
}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown("# 🔐")
st.sidebar.title("QuantumShieldFL")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📈 Training Progress", "🔑 QKD Status",
     "🛡️ Security Monitor", "📡 Communication", "⚔️ Attack Analysis",
     "📋 Comparison", "🎮 Interactive Playground"],
    index=7,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.json({
    "Clients": config.NUM_CLIENTS,
    "Rounds": config.NUM_ROUNDS,
    "QKD Qubits": config.QKD_NUM_QUBITS,
    "QBER Threshold": config.QBER_THRESHOLD,
    "Risk Weights": {
        "QBER": config.RISK_WEIGHT_QBER,
        "Divergence": config.RISK_WEIGHT_DIVERGENCE,
        "Anomaly": config.RISK_WEIGHT_ANOMALY,
        "Key Age": config.RISK_WEIGHT_KEY_AGE,
        "Trend": config.RISK_WEIGHT_TREND,
    },
})

# Load data
data = load_metrics()
experiments, comparison, db_rounds = load_db_data()


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def create_plotly_layout(title, xaxis_title="Round", yaxis_title="", height=450):
    """Create a consistent dark-theme Plotly layout."""
    return go.Layout(
        title=dict(text=title, font=dict(size=18, color="#e2e8f0")),
        xaxis=dict(title=xaxis_title, gridcolor="#2d3748", color="#a0aec0"),
        yaxis=dict(title=yaxis_title, gridcolor="#2d3748", color="#a0aec0"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#a0aec0"),
        legend=dict(font=dict(color="#e2e8f0")),
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
    )


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.title("🔐 QuantumShieldFL Dashboard")
    st.markdown("### Adaptive Privacy-Preserving Federated Learning with Quantum Key Distribution")
    
    if not data:
        st.info("💡 **Welcome to QuantumShieldFL!**")
        st.markdown(
            "It looks like you haven't run any full benchmark experiments yet.\n\n"
            "👉 **To explore the project visually right now**, select **🎮 Interactive Playground** from the left sidebar!\n\n"
            "To generate the charts on this page, run the following in your terminal:"
        )
        st.code("python -m orchestrator.experiment_runner --all", language="bash")
        st.stop()
    
    # Summary metrics
    cols = st.columns(4)
    
    for i, (exp_name, exp_data) in enumerate(data.items()):
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            col_idx = i % 4
            with cols[col_idx]:
                display_name = POLICY_NAMES.get(exp_name, exp_name)
                st.metric(
                    f"{display_name} — Final Accuracy",
                    f"{rdf['accuracy'].iloc[-1]:.2%}",
                    delta=f"{rdf['accuracy'].iloc[-1] - rdf['accuracy'].iloc[0]:.2%}",
                )
    
    st.markdown("---")
    
    # Quick accuracy comparison
    fig = go.Figure(layout=create_plotly_layout(
        "Model Accuracy Across Experiments", yaxis_title="Accuracy"
    ))
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            fig.add_trace(go.Scatter(
                x=rdf["round"], y=rdf["accuracy"],
                mode="lines+markers",
                name=POLICY_NAMES.get(exp_name, exp_name),
                line=dict(color=COLORS.get(exp_name, "#888"), width=2),
                marker=dict(size=5),
            ))
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Training Progress":
    st.title("📈 Training Progress")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    # Accuracy
    fig_acc = go.Figure(layout=create_plotly_layout(
        "Global Model Accuracy per Round", yaxis_title="Accuracy"
    ))
    
    fig_loss = go.Figure(layout=create_plotly_layout(
        "Global Model Loss per Round", yaxis_title="Loss"
    ))
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            color = COLORS.get(exp_name, "#888")
            name = POLICY_NAMES.get(exp_name, exp_name)
            
            fig_acc.add_trace(go.Scatter(
                x=rdf["round"], y=rdf["accuracy"],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5),
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=rdf["round"], y=rdf["loss"],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5),
            ))
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_acc, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Per-client training accuracy
    st.markdown("### Per-Client Training Metrics")
    selected_exp = st.selectbox(
        "Select experiment",
        list(data.keys()),
        format_func=lambda x: POLICY_NAMES.get(x, x),
    )
    
    if selected_exp and not data[selected_exp]["clients"].empty:
        cdf = data[selected_exp]["clients"]
        fig_client = px.line(
            cdf, x="round", y="train_accuracy", color="client_id",
            title="Per-Client Training Accuracy",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_client.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#a0aec0"),
        )
        st.plotly_chart(fig_client, use_container_width=True)

elif page == "🔑 QKD Status":
    st.title("🔑 Quantum Key Distribution Status")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    # QBER over rounds
    fig_qber = go.Figure(layout=create_plotly_layout(
        "Quantum Bit Error Rate (QBER) per Round", yaxis_title="QBER"
    ))
    
    # Add threshold line
    fig_qber.add_hline(
        y=config.QBER_THRESHOLD, line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"Threshold ({config.QBER_THRESHOLD})",
    )
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty and exp_name != "no_encryption":
            rdf = exp_data["rounds"]
            if "qber" in rdf.columns:
                fig_qber.add_trace(go.Scatter(
                    x=rdf["round"], y=rdf["qber"],
                    mode="lines+markers",
                    name=POLICY_NAMES.get(exp_name, exp_name),
                    line=dict(color=COLORS.get(exp_name, "#888"), width=2),
                ))
    
    st.plotly_chart(fig_qber, use_container_width=True)
    
    # Key generation events
    st.markdown("### Key Generation Events")
    for exp_name, exp_data in data.items():
        if not exp_data["keys"].empty:
            st.markdown(f"**{POLICY_NAMES.get(exp_name, exp_name)}**")
            kdf = exp_data["keys"]
            
            total_events = len(kdf)
            rej_count = len(kdf[kdf["event_type"] == "rejected"]) if "event_type" in kdf.columns else 0
            gen_count = total_events - rej_count
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Keys Generated", gen_count)
            c2.metric("Keys Rejected", rej_count)
            c3.metric("Success Rate",
                       f"{gen_count/(gen_count+rej_count)*100:.1f}%"
                       if (gen_count+rej_count) > 0 else "N/A")
    
    # Keys generated per round (bar chart)
    fig_keys = go.Figure(layout=create_plotly_layout(
        "Keys Generated per Round", yaxis_title="Keys"
    ))
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty and exp_name != "no_encryption":
            rdf = exp_data["rounds"]
            if "keys_generated" in rdf.columns:
                fig_keys.add_trace(go.Bar(
                    x=rdf["round"], y=rdf["keys_generated"],
                    name=POLICY_NAMES.get(exp_name, exp_name),
                    marker_color=COLORS.get(exp_name, "#888"),
                    opacity=0.8,
                ))
    
    fig_keys.update_layout(barmode="group")
    st.plotly_chart(fig_keys, use_container_width=True)

elif page == "🛡️ Security Monitor":
    st.title("🛡️ Security & Risk Monitor")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    # Risk score over rounds
    fig_risk = go.Figure(layout=create_plotly_layout(
        "Risk Score per Round", yaxis_title="Risk Score"
    ))
    
    # Add threshold bands
    fig_risk.add_hrect(
        y0=0, y1=config.RISK_LOW_THRESHOLD,
        fillcolor=COLORS["success"], opacity=0.1,
        annotation_text="LOW",
    )
    fig_risk.add_hrect(
        y0=config.RISK_LOW_THRESHOLD, y1=config.RISK_HIGH_THRESHOLD,
        fillcolor=COLORS["warning"], opacity=0.1,
        annotation_text="MEDIUM",
    )
    fig_risk.add_hrect(
        y0=config.RISK_HIGH_THRESHOLD, y1=1.0,
        fillcolor=COLORS["danger"], opacity=0.1,
        annotation_text="HIGH",
    )
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            if "risk_score" in rdf.columns:
                fig_risk.add_trace(go.Scatter(
                    x=rdf["round"], y=rdf["risk_score"],
                    mode="lines+markers",
                    name=POLICY_NAMES.get(exp_name, exp_name),
                    line=dict(color=COLORS.get(exp_name, "#888"), width=2.5),
                ))
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Attack timeline
    st.markdown("### Attack Activity Window")
    st.info(f"⚔️ Attacks active during rounds **{config.ATTACK_START_ROUND}** – **{config.ATTACK_END_ROUND}**")
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            if "attack_detected" in rdf.columns:
                detected = rdf[rdf["attack_detected"].astype(bool) == True]
                if not detected.empty:
                    st.success(
                        f"✅ **{POLICY_NAMES.get(exp_name, exp_name)}**: "
                        f"Detected attacks in rounds {detected['round'].tolist()}"
                    )

elif page == "📡 Communication":
    st.title("📡 Communication Overhead")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    # Bytes transmitted per round
    fig_bytes = go.Figure(layout=create_plotly_layout(
        "Bytes Transmitted per Round", yaxis_title="Bytes"
    ))
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            if "bytes_sent" in rdf.columns:
                fig_bytes.add_trace(go.Scatter(
                    x=rdf["round"], y=rdf["bytes_sent"],
                    mode="lines+markers",
                    name=POLICY_NAMES.get(exp_name, exp_name),
                    line=dict(color=COLORS.get(exp_name, "#888"), width=2),
                    fill="tozeroy", fillcolor=hex_to_rgba(COLORS.get(exp_name, "#888888")),
                ))
    
    st.plotly_chart(fig_bytes, use_container_width=True)
    
    # Round timing
    fig_time = go.Figure(layout=create_plotly_layout(
        "Round Duration", yaxis_title="Time (ms)"
    ))
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            if "round_time_ms" in rdf.columns:
                fig_time.add_trace(go.Bar(
                    x=rdf["round"], y=rdf["round_time_ms"],
                    name=POLICY_NAMES.get(exp_name, exp_name),
                    marker_color=COLORS.get(exp_name, "#888"),
                    opacity=0.8,
                ))
    
    fig_time.update_layout(barmode="group")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Cumulative overhead
    st.markdown("### Cumulative Communication Overhead")
    overhead_data = []
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            total_bytes = rdf["bytes_sent"].sum() if "bytes_sent" in rdf.columns else 0
            total_time = rdf["round_time_ms"].sum() if "round_time_ms" in rdf.columns else 0
            total_keys = rdf["keys_generated"].sum() if "keys_generated" in rdf.columns else 0
            overhead_data.append({
                "Experiment": POLICY_NAMES.get(exp_name, exp_name),
                "Total Bytes": f"{total_bytes:,}",
                "Total Time (s)": f"{total_time/1000:.1f}",
                "Total Keys": int(total_keys),
            })
    
    if overhead_data:
        st.dataframe(pd.DataFrame(overhead_data), use_container_width=True)

elif page == "⚔️ Attack Analysis":
    st.title("⚔️ Attack Analysis")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    st.markdown(f"""
    ### Attack Configuration
    - **Eavesdropper (Eve)**: Intercept-and-resend on QKD channel
    - **Byzantine Client**: Client 3 sends noise updates  
    - **Attack Window**: Rounds {config.ATTACK_START_ROUND}–{config.ATTACK_END_ROUND}
    - **Eve Intercept Probability**: {config.EVE_INTERCEPT_PROBABILITY*100:.0f}%
    """)
    
    # QBER spike during attack
    st.markdown("### QBER During Attack Window")
    
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty and exp_name != "no_encryption":
            rdf = exp_data["rounds"]
            if "qber" in rdf.columns:
                attack_rounds = rdf[
                    (rdf["round"] >= config.ATTACK_START_ROUND) & 
                    (rdf["round"] <= config.ATTACK_END_ROUND)
                ]
                normal_rounds = rdf[
                    (rdf["round"] < config.ATTACK_START_ROUND) | 
                    (rdf["round"] > config.ATTACK_END_ROUND)
                ]
                
                avg_attack_qber = attack_rounds["qber"].mean() if not attack_rounds.empty else 0
                avg_normal_qber = normal_rounds["qber"].mean() if not normal_rounds.empty else 0
                
                name = POLICY_NAMES.get(exp_name, exp_name)
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{name} — Normal QBER", f"{avg_normal_qber:.4f}")
                c2.metric(f"{name} — Attack QBER", f"{avg_attack_qber:.4f}",
                          delta=f"+{avg_attack_qber - avg_normal_qber:.4f}",
                          delta_color="inverse")
                c3.metric(f"{name} — Detection",
                          "✅ Yes" if avg_attack_qber > config.QBER_THRESHOLD else "❌ No")

elif page == "📋 Comparison":
    st.title("📋 Experiment Comparison")
    
    if not data:
        st.warning("No data available.")
        st.stop()
    
    # Build comparison table
    comparison_rows = []
    for exp_name, exp_data in data.items():
        if not exp_data["rounds"].empty:
            rdf = exp_data["rounds"]
            
            row = {
                "Experiment": POLICY_NAMES.get(exp_name, exp_name),
                "Policy": exp_name.replace("_", " ").title(),
                "Final Accuracy": f"{rdf.get('accuracy', pd.Series([0])).fillna(0).iloc[-1]:.4f}",
                "Best Accuracy": f"{rdf.get('accuracy', pd.Series([0])).fillna(0).max():.4f}",
                "Final Loss": f"{rdf.get('loss', pd.Series([0])).fillna(0).iloc[-1]:.4f}",
                "Total Keys": int(rdf["keys_generated"].fillna(0).sum()) if "keys_generated" in rdf.columns else 0,
                "Total Bytes": f"{rdf['bytes_sent'].fillna(0).sum():,.0f}" if "bytes_sent" in rdf.columns else "0",
                "Avg Round Time (ms)": f"{rdf['round_time_ms'].fillna(0).mean():.0f}" if "round_time_ms" in rdf.columns else "0",
                "Attack Detection": "✅" if ("attack_detected" in rdf.columns and rdf["attack_detected"].astype(bool).any()) else "❌",
            }
            comparison_rows.append(row)
    
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    
    # Radar chart
    st.markdown("### Multi-Dimensional Comparison")
    
    if len(data) >= 2:
        categories = ["Accuracy", "Security", "Low Overhead", "Attack Detection", "Key Efficiency"]
        
        fig_radar = go.Figure()
        
        for exp_name, exp_data in data.items():
            if not exp_data["rounds"].empty:
                rdf = exp_data["rounds"]
                
                acc = rdf.get('accuracy', pd.Series([0])).fillna(0).iloc[-1]
                security = 1.0 if exp_name != "no_encryption" else 0.0
                overhead = 1.0 - min(1.0, rdf['round_time_ms'].fillna(0).mean() / 5000) if 'round_time_ms' in rdf.columns else 0.5
                detection = 1.0 if ("attack_detected" in rdf.columns and rdf["attack_detected"].astype(bool).any()) else 0.0
                total_keys = rdf['keys_generated'].fillna(0).sum() if 'keys_generated' in rdf.columns else 0
                key_eff = 1.0 - min(1.0, total_keys / (config.NUM_ROUNDS * config.NUM_CLIENTS))
                
                values = [acc, security, overhead, detection, key_eff]
                values.append(values[0])  # Close the polygon
                cats = categories + [categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=cats,
                    fill='toself', name=POLICY_NAMES.get(exp_name, exp_name),
                    line=dict(color=COLORS.get(exp_name, "#888")),
                    fillcolor=COLORS.get(exp_name, "#888") + "30",
                ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0e1117",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2d3748"),
                angularaxis=dict(gridcolor="#2d3748"),
            ),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#a0aec0"),
            showlegend=True,
            height=500,
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

elif page == "🎮 Interactive Playground":
    st.title("🎮 Interactive Playground")
    st.markdown("Live execution & visualization of core QuantumShieldFL modules. All results are saved to the database.")

    # Imports specifically for the playground
    from quantum.bb84 import run_bb84
    from crypto.aes_handler import AESCipher
    from cryptography.exceptions import InvalidTag
    from security.risk_scorer import RiskScorer
    from attacks.model_poisoning import ByzantineAttacker
    from federated.model import get_model, get_model_params
    import torch
    import sqlite3
    from datetime import datetime

    # --- Helper: get or create a playground experiment in the DB ---
    def get_playground_db():
        db = MetricsDB()
        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM experiments WHERE name = 'playground'")
        row = cursor.fetchone()
        if row:
            exp_id = row["id"]
        else:
            exp_id = db.create_experiment(
                name="playground",
                policy="interactive",
                num_clients=5,
                num_rounds=0,
                config_dict={"source": "Interactive Playground"}
            )
        return db, exp_id

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚛️ Qiskit BB84 Simulation",
        "🔒 AES Encryption",
        "🛡️ Risk Scorer",
        "⚔️ Attack Simulator",
        "🗄️ Database Explorer",
    ])

    # --- TAB 1: Qiskit BB84 Simulation ---
    with tab1:
        st.subheader("Simulate Quantum Key Distribution (BB84)")
        col1, col2 = st.columns(2)
        with col1:
            sim_qubits = st.slider("Number of Qubits", min_value=64, max_value=1024, value=256, step=64)
        with col2:
            eve_active = st.checkbox("Enable Eavesdropper (Eve)")
            eve_prob = st.slider("Eve Interception Probability", 0.0, 1.0, 0.5, 0.1, disabled=not eve_active)

        if st.button("Run Quantum Key Distribution", type="primary"):
            with st.spinner("Simulating quantum circuits via Qiskit..."):
                res = run_bb84(num_qubits=sim_qubits, eve_active=eve_active, eve_intercept_prob=eve_prob if eve_active else 0.0)

            st.metric("Quantum Bit Error Rate (QBER)", f"{res['qber']:.4f}")
            if res['eavesdropper_detected']:
                st.error("🚨 Eavesdropper Detected! Key rejected.")
            else:
                st.success("✅ Secure key successfully generated!")
                st.code(f"Key (Hex): {res['shared_key'].hex()}\nLength: {res['key_length']} bits")

            st.info(f"Sifted bits: {res['sifted_bits']} (Matches out of {res['num_qubits']} qubits sent)")

            # Log to database
            try:
                db, exp_id = get_playground_db()
                db.insert_key_event(exp_id, {
                    "round": 0,
                    "client_id": 0,
                    "event_type": "bb84_playground",
                    "qber": res['qber'],
                    "key_length": res['key_length'],
                    "reason": f"Eve={'ON' if eve_active else 'OFF'}, qubits={sim_qubits}, detected={res['eavesdropper_detected']}"
                })
                db.close()
                st.caption("✅ Result saved to database")
            except Exception as e:
                st.caption(f"⚠️ DB log skipped: {e}")

    # --- TAB 2: AES Encryption ---
    with tab2:
        st.subheader("AES-256-GCM Authenticated Encryption")
        message = st.text_area("Secret Message", value="This is a confidential model update from Client 1.")
        tamper = st.checkbox("Simulate Network Tampering (Flip 1 byte of ciphertext)")

        if st.button("Encrypt & Decrypt"):
            key = os.urandom(32)
            cipher = AESCipher(key)

            st.markdown("**1. Encryption**")
            enc = cipher.encrypt(message.encode())
            st.code(f"AES Key (Hex): {key.hex()}\nCiphertext (Hex): {enc['ciphertext'].hex()}\nNonce (Hex): {enc['nonce'].hex()}")

            st.markdown("**2. Transmission & Decryption**")
            ciphertext_to_decrypt = enc['ciphertext']
            if tamper:
                tampered = bytearray(ciphertext_to_decrypt)
                tampered[0] ^= 0xFF
                ciphertext_to_decrypt = bytes(tampered)
                st.warning("⚠️ Ciphertext tampered during transmission!")

            try:
                dec = cipher.decrypt(ciphertext_to_decrypt, enc['nonce'])
                st.success(f"✅ Decrypted Successfully: {dec.decode()}")
            except InvalidTag:
                st.error("🚨 Decryption Failed! Invalid Authentication Tag (Data Tampering Detected).")

    # --- TAB 3: Risk Scorer ---
    with tab3:
        st.subheader("Adaptive Risk Engine")
        st.markdown("Adjust parameters to see how the system calculates risk and assigns policy.")
        rc1, rc2 = st.columns(2)
        with rc1:
            qber_input = st.slider("Current QBER", 0.0, 0.3, 0.02, 0.01)
            age_input = st.slider("Key Age (Rounds)", 0, 10, 1)
        with rc2:
            div_input = st.slider("Model Update Divergence", 0.0, 1.0, 0.05, 0.01)
            anom_norm = st.number_input("Client Update Norm", value=10.0)

        if st.button("Calculate System Risk"):
            scorer = RiskScorer()
            client_norms = {0: 10.0, 1: 10.2, 2: 9.8, 3: anom_norm, 4: 10.1}
            risk = scorer.compute_risk(qber=qber_input, divergences=[div_input], client_norms=client_norms, key_age_rounds=age_input, round_number=2)

            st.metric("Composite Risk Score", f"{risk['risk_score']:.3f}", delta=risk['risk_level'], delta_color="inverse")
            st.info(f"Policy Recommendation: **{risk['recommendation']}**")

            # Show components as a bar chart
            comp = risk['components']
            comp_df = pd.DataFrame({"Signal": list(comp.keys()), "Score": list(comp.values())})
            fig_risk = px.bar(
                comp_df, x="Signal", y="Score",
                color="Score", color_continuous_scale=["#66bb6a", "#ffa726", "#ef5350"],
                range_color=[0, 1], range_y=[0, 1],
                title="Risk Component Breakdown",
            )
            fig_risk.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#a0aec0"),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Log to database
            try:
                db, exp_id = get_playground_db()
                db.insert_round(exp_id, {
                    "round": age_input,
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "qber": qber_input,
                    "risk_score": risk['risk_score'],
                    "risk_level": risk['risk_level'],
                    "keys_generated": 1 if risk['risk_level'] == 'HIGH' else 0,
                    "bytes_sent": 0,
                    "round_time_ms": 0,
                    "eavesdropper_active": qber_input > 0.11,
                    "attack_detected": risk['risk_level'] == 'HIGH',
                })
                db.close()
                st.caption("✅ Result saved to database")
            except Exception as e:
                st.caption(f"⚠️ DB log skipped: {e}")

    # --- TAB 4: Attack Simulator ---
    with tab4:
        st.subheader("Byzantine Model Poisoning Simulator")
        attack_type = st.selectbox("Attack Strategy", ["Gaussian Noise", "Sign-Flip", "Scale"])
        noise_scale = st.slider("Noise/Scale Factor", 1.0, 20.0, 5.0)

        if st.button("Simulate Poisoned Model Update"):
            model = get_model()
            params = get_model_params(model)
            orig_norm = np.linalg.norm(np.concatenate([p.flatten() for p in params]))

            atk_code = "noise" if attack_type == "Gaussian Noise" else "flip" if attack_type == "Sign-Flip" else "scale"
            attacker = ByzantineAttacker(attack_type=atk_code, noise_scale=noise_scale)
            poisoned = attacker.poison_update(params)
            new_norm = np.linalg.norm(np.concatenate([p.flatten() for p in poisoned]))

            # Visual comparison
            norm_df = pd.DataFrame({
                "Model": ["Original (Honest)", "Poisoned (Byzantine)"],
                "L2 Norm": [orig_norm, new_norm],
            })
            fig_atk = px.bar(
                norm_df, x="Model", y="L2 Norm", color="Model",
                color_discrete_map={"Original (Honest)": "#66bb6a", "Poisoned (Byzantine)": "#ef5350"},
                title=f"Model Norm Comparison: {attack_type} Attack (scale={noise_scale})"
            )
            fig_atk.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#a0aec0"), showlegend=False,
            )
            st.plotly_chart(fig_atk, use_container_width=True)

            ratio = new_norm / orig_norm if orig_norm > 0 else 0
            st.markdown(f"**Original Norm:** `{orig_norm:.2f}` | **Poisoned Norm:** `{new_norm:.2f}` | **Ratio:** `{ratio:.1f}x`")

            if ratio > 1.5:
                st.error(f"🚨 Massive divergence detected ({ratio:.1f}x)! The server's anomaly detection would reject this update.")
            elif attack_type == "Sign-Flip":
                st.warning("⚠️ Sign-flipped. The server would detect high cosine distance divergence.")
            else:
                st.success("✅ Update looks normal. The server might accept this if within bounds.")

            # Log to database
            try:
                db, exp_id = get_playground_db()
                db.insert_client_round(exp_id, {
                    "round": 0,
                    "client_id": 99,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "update_norm": float(new_norm),
                    "divergence": float(ratio),
                    "is_byzantine": True,
                    "key_regenerated": False,
                })
                db.close()
                st.caption("✅ Result saved to database")
            except Exception as e:
                st.caption(f"⚠️ DB log skipped: {e}")

    # --- TAB 5: Database Explorer ---
    with tab5:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader("🗄️ Database Explorer")
            st.markdown("View all data stored in the SQLite database from experiments and playground interactions.")
        with col2:
            if st.button("🔄 Refresh Data"):
                st.session_state["db_refreshed"] = True
                st.cache_data.clear()
                st.rerun()
                
        if st.session_state.get("db_refreshed", False):
            st.toast("✅ Database synced with latest records!")
            st.session_state["db_refreshed"] = False
                
        try:
            db = MetricsDB()
            ts = datetime.now().strftime("%Y%m%d%H%M%S")

            # --- Experiments Table ---
            st.markdown("### 📋 Experiments")
            exps = db.get_all_experiments()
            if exps:
                latest_exp_id = exps[0]['id']
                if st.session_state.get('last_seen_exp_id') != latest_exp_id:
                    st.info(f"**Debug Info:** The latest Experiment in the database is ID **{latest_exp_id}**.")
                    st.session_state['last_seen_exp_id'] = latest_exp_id
                exp_df = pd.DataFrame(exps)
                st.dataframe(exp_df, use_container_width=True, key=f"df_exps_{ts}")
            else:
                st.info("No experiments recorded yet. Use the other Playground tabs to generate data, or run:\n`python -m orchestrator.experiment_runner --all`")

            # --- Rounds Table ---
            st.markdown("### 📈 Training Rounds")
            cursor = db.conn.cursor()
            cursor.execute("SELECT r.*, e.name as experiment_name FROM rounds r JOIN experiments e ON r.experiment_id = e.id ORDER BY r.id DESC LIMIT 100")
            rows = [dict(row) for row in cursor.fetchall()]
            if rows:
                latest_round_id = rows[0]['id']
                if st.session_state.get('last_seen_round_id') != latest_round_id:
                    st.info(f"**Debug Info:** The latest Training Round in the database is ID **{latest_round_id}**.")
                    st.session_state['last_seen_round_id'] = latest_round_id
                rounds_df = pd.DataFrame(rows)
                st.dataframe(rounds_df, use_container_width=True, key=f"df_rounds_{ts}")

                # Quick chart of risk scores
                if 'risk_score' in rounds_df.columns and rounds_df['risk_score'].notna().any():
                    fig_db_risk = px.line(
                        rounds_df.sort_values('id'), x='id', y='risk_score',
                        color='experiment_name', title="Risk Score History",
                        markers=True,
                    )
                    fig_db_risk.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#a0aec0"),
                    )
                    st.plotly_chart(fig_db_risk, use_container_width=True)
            else:
                st.info("No round data yet. Use the Risk Scorer tab to generate entries.")

            # --- Key Events Table ---
            st.markdown("### 🔑 QKD Key Events")
            cursor.execute("SELECT ke.*, e.name as experiment_name FROM key_events ke JOIN experiments e ON ke.experiment_id = e.id ORDER BY ke.id DESC LIMIT 50")
            key_rows = [dict(row) for row in cursor.fetchall()]
            if key_rows:
                latest_key_id = key_rows[0]['id']
                if st.session_state.get('last_seen_key_id') != latest_key_id:
                    st.info(f"**Debug Info:** The latest Key Event in the database is ID **{latest_key_id}**.")
                    st.session_state['last_seen_key_id'] = latest_key_id
                keys_df = pd.DataFrame(key_rows)
                st.dataframe(keys_df, use_container_width=True, key=f"df_keys_{ts}")

                # QBER scatter
                if 'qber' in keys_df.columns and keys_df['qber'].notna().any():
                    fig_qber = px.scatter(
                        keys_df, x='id', y='qber', color='reason',
                        title="QBER from Playground BB84 Sessions",
                        size_max=10,
                    )
                    fig_qber.add_hline(y=config.QBER_THRESHOLD, line_dash="dash",
                                       line_color="#ef5350",
                                       annotation_text=f"Threshold ({config.QBER_THRESHOLD})")
                    fig_qber.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#a0aec0"),
                    )
                    st.plotly_chart(fig_qber, use_container_width=True)
            else:
                st.info("No key events yet. Use the BB84 Simulation tab to generate entries.")

            # --- Client Rounds / Attack Logs ---
            st.markdown("### ⚔️ Client & Attack Logs")
            cursor.execute("SELECT cr.*, e.name as experiment_name FROM client_rounds cr JOIN experiments e ON cr.experiment_id = e.id ORDER BY cr.id DESC LIMIT 50")
            client_rows = [dict(row) for row in cursor.fetchall()]
            if client_rows:
                latest_client_id = client_rows[0]['id']
                if st.session_state.get('last_seen_client_id') != latest_client_id:
                    st.info(f"**Debug Info:** The latest Client/Attack Log in the database is ID **{latest_client_id}**.")
                    st.session_state['last_seen_client_id'] = latest_client_id
                client_df = pd.DataFrame(client_rows)
                st.dataframe(client_df, use_container_width=True, key=f"df_client_{ts}")
            else:
                st.info("No client/attack data yet. Use the Attack Simulator tab to generate entries.")

            # --- Summary Stats ---
            st.markdown("### 📊 Database Summary")
            cursor.execute("SELECT COUNT(*) as c FROM experiments")
            n_exp = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM rounds")
            n_rnd = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM key_events")
            n_key = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM client_rounds")
            n_cli = cursor.fetchone()["c"]

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Experiments", n_exp)
            sc2.metric("Round Records", n_rnd)
            sc3.metric("Key Events", n_key)
            sc4.metric("Client Records", n_cli)

            db.close()

        except Exception as e:
            st.error(f"Database error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**QuantumShieldFL** v1.0  \n"
    "Adaptive Privacy-Preserving FL  \n"
    "with Quantum Key Distribution"
)
