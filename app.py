'''
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bosch Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

    .stApp {
        background: radial-gradient(ellipse at top left, #0d1b2a 0%, #050a14 40%, #0a0f1e 100%);
        background-attachment: fixed;
    }

    .main .block-container { padding-top: 1rem; }

    .title-text {
        font-family: 'Orbitron', monospace;
        font-size: 2.8em;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b35, #f7c59f, #ff6b35);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: shine 3s linear infinite;
        letter-spacing: 2px;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .subtitle-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2em;
        color: #64ffda;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(22,33,62,0.9));
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(100,255,218,0.2);
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .danger-card {
        background: linear-gradient(135deg, rgba(61,21,21,0.95), rgba(92,26,26,0.95));
        border-radius: 16px;
        padding: 22px;
        border: 2px solid #ff4444;
        text-align: center;
        box-shadow: 0 0 30px rgba(255,68,68,0.4);
        animation: pulse-red 2s infinite;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 30px rgba(255,68,68,0.4); }
        50% { box-shadow: 0 0 50px rgba(255,68,68,0.8); }
    }

    .safe-card {
        background: linear-gradient(135deg, rgba(13,51,32,0.95), rgba(26,92,53,0.95));
        border-radius: 16px;
        padding: 22px;
        border: 2px solid #00ff88;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,255,136,0.3);
    }

    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.1em;
        font-weight: 700;
        color: #ff6b35;
        border-bottom: 2px solid #ff6b35;
        padding-bottom: 8px;
        margin-bottom: 15px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .cost-box {
        background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(22,33,62,0.9));
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #ff6b35;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }

    .cost-box-green {
        background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(22,33,62,0.9));
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }

    .machine-card {
        background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(22,33,62,0.9));
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(100,255,218,0.15);
        margin: 8px 0;
        text-align: center;
    }

    .stSidebar {
        background: linear-gradient(180deg, #0d1b2a, #050a14) !important;
        border-right: 1px solid rgba(100,255,218,0.1);
    }

    .nav-tab {
        background: rgba(13,27,42,0.8);
        border-radius: 10px;
        padding: 8px 15px;
        border: 1px solid rgba(100,255,218,0.2);
        color: #64ffda;
        cursor: pointer;
        text-align: center;
    }

    div[data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'Orbitron', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
features = pickle.load(open(os.path.join(BASE_DIR, 'features.pkl'), 'rb'))

FEATURE_NAMES = [
    'Type', 'Air temperature _K', 'Process temperature _K',
    'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min'
]

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="title-text">⚙ BOSCH PREDICTIVE MAINTENANCE AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">[ Industry 4.0 • Real-time Intelligence • Smart Manufacturing ]</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Navigation ───────────────────────────────────────────────
tabs = st.tabs(["🔍 Single Machine", "🏭 Multi-Machine Dashboard", "📈 Trend Analysis", "📄 Report"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Single Machine Analysis
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    with st.sidebar:
        st.markdown("### ⚙️ Sensor Input Panel")
        st.markdown("---")
        machine_type = st.selectbox("🏭 Machine Type", ["L", "M", "H"])
        type_encoded = {"L": 0, "M": 1, "H": 2}[machine_type]
        st.markdown("**🌡️ Temperature**")
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
        st.markdown("**⚡ Mechanical**")
        rpm = st.slider("Rotational Speed (RPM)", 1168, 2886, 1500)
        torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
        st.markdown("**🔩 Tool**")
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)
        st.markdown("---")
        predict_btn = st.button("🔍 Analyze Machine Health", use_container_width=True)
        st.markdown("---")
        st.markdown("**📊 Normal Ranges**")
        st.caption("Air Temp: 295-305 K")
        st.caption("Process Temp: 305-315 K")
        st.caption("RPM: 1168-2886")
        st.caption("Torque: 3.8-76.6 Nm")
        st.caption("Tool Wear: 0-253 min")

        # Email alert
        st.markdown("---")
        st.markdown("**📧 Email Alerts**")
        alert_email = st.text_input("Your Email", placeholder="you@email.com")
        email_btn = st.button("Enable Alerts", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame(
            [[type_encoded, air_temp, process_temp, rpm, torque, tool_wear]],
            columns=FEATURE_NAMES
        )

        with st.spinner("🔄 Running AI analysis..."):
            time.sleep(1)

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        # Store in session state for PDF
        st.session_state['last_prediction'] = {
            'prob': prob, 'pred': pred,
            'air_temp': air_temp, 'process_temp': process_temp,
            'rpm': rpm, 'torque': torque, 'tool_wear': tool_wear,
            'machine_type': machine_type, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # ── Risk Cards ───────────────────────────────────────
        st.markdown('<p class="section-header">📊 Risk Assessment</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="danger-card">
                    <h2>⚠️ FAILURE RISK</h2>
                    <h1 style="color:#ff4444;font-size:3em;font-family:'Orbitron'">{prob*100:.1f}%</h1>
                    <p style="color:#ff9999">Immediate Action Required</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    <h2>✅ HEALTHY</h2>
                    <h1 style="color:#00ff88;font-size:3em;font-family:'Orbitron'">{prob*100:.1f}%</h1>
                    <p style="color:#99ffcc">Operating Normally</p>
                </div>""", unsafe_allow_html=True)

        with col2:
            wear_pct = tool_wear / 253 * 100
            wear_color = "#ff4444" if wear_pct > 80 else "#ffaa00" if wear_pct > 60 else "#00ff88"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#64ffda">🔩 Tool Wear</h3>
                <h1 style="color:{wear_color};font-size:2.5em;font-family:'Orbitron'">{wear_pct:.0f}%</h1>
                <p style="color:#8892b0">of max lifespan</p>
            </div>""", unsafe_allow_html=True)

        with col3:
            temp_diff = process_temp - air_temp
            temp_color = "#ff4444" if temp_diff > 11 else "#00ff88"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#64ffda">🌡️ Temp Delta</h3>
                <h1 style="color:{temp_color};font-size:2.5em;font-family:'Orbitron'">{temp_diff:.1f}K</h1>
                <p style="color:#8892b0">process vs air</p>
            </div>""", unsafe_allow_html=True)

        # ── Gauge Chart ──────────────────────────────────────
        st.markdown("---")
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={'text': "Failure Risk Score", 'font': {'color': 'white', 'size': 18}},
                number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#ff4444" if prob > 0.5 else "#ffaa00" if prob > 0.3 else "#00ff88"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,136,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(255,170,0,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(255,68,68,0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=280,
                margin=dict(t=50, b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            # Radar chart of sensor health
            categories = ['Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear']
            values = [
                (air_temp - 295) / 10 * 100,
                (process_temp - 305) / 10 * 100,
                (rpm - 1168) / (2886 - 1168) * 100,
                (torque - 3.8) / (76.6 - 3.8) * 100,
                tool_wear / 253 * 100
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(255,107,53,0.2)',
                line=dict(color='#ff6b35', width=2),
                name='Current'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='white'),
                    angularaxis=dict(color='white'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title=dict(text='Sensor Health Radar', font=dict(color='white')),
                height=280,
                margin=dict(t=50, b=0),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Failure Type Analysis ────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">🔬 Failure Type Analysis</p>', unsafe_allow_html=True)

        failure_types = {
            'Tool Wear Failure': min(100, tool_wear / 253 * 100 * 1.2),
            'Heat Dissipation': min(100, max(0, (temp_diff - 8.6) / 3 * 100)),
            'Power Failure': min(100, max(0, (torque * rpm / 9550 - 3.5) / 6 * 100)),
            'Overstrain Failure': min(100, max(0, (torque * tool_wear - 11000) / 3000 * 100)),
            'Random Failure': 3.0
        }

        colors = ['#ff4444' if v > 50 else '#ffaa00' if v > 25 else '#00ff88'
                  for v in failure_types.values()]

        fig_bar = go.Figure(go.Bar(
            x=list(failure_types.values()),
            y=list(failure_types.keys()),
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
            text=[f'{v:.1f}%' for v in failure_types.values()],
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.5)',
            font={'color': 'white'},
            xaxis=dict(range=[0, 115], color='white', title='Risk %'),
            yaxis=dict(color='white'),
            title=dict(text='Failure Type Risk Breakdown', font=dict(color='white', size=14)),
            height=300,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Cost Estimator ───────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">💰 Maintenance Cost Estimator</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        unplanned_downtime = int(prob * 8 * 500)
        repair_cost = int(prob * 15000)
        production_loss = int(prob * 25000)
        total_failure_cost = unplanned_downtime + repair_cost + production_loss
        preventive_cost = 2500
        parts_cost = int(tool_wear / 253 * 3000)
        total_preventive = preventive_cost + parts_cost
        savings = total_failure_cost - total_preventive

        with col1:
            st.markdown("**🔴 Cost if IGNORED:**")
            st.markdown(f"""
            <div class="cost-box">
                <p>⏱️ Downtime Cost: <b style="color:#ff6b35">₹{unplanned_downtime:,}</b></p>
                <p>🔧 Emergency Repair: <b style="color:#ff6b35">₹{repair_cost:,}</b></p>
                <p>📦 Production Loss: <b style="color:#ff6b35">₹{production_loss:,}</b></p>
                <hr style="border-color:#3a3f52">
                <p style="font-size:1.2em">💸 Total: <b style="color:#ff4444;font-size:1.4em">₹{total_failure_cost:,}</b></p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("**🟢 Preventive Cost:**")
            st.markdown(f"""
            <div class="cost-box-green">
                <p>👨‍🔧 Scheduled Service: <b style="color:#00ff88">₹{preventive_cost:,}</b></p>
                <p>🔩 Parts Replacement: <b style="color:#00ff88">₹{parts_cost:,}</b></p>
                <p>&nbsp;</p>
                <hr style="border-color:#3a3f52">
                <p style="font-size:1.2em">✅ Total: <b style="color:#00ff88;font-size:1.4em">₹{total_preventive:,}</b></p>
                <p>💡 Savings: <b style="color:#ffaa00;font-size:1.3em">₹{savings:,}</b></p>
            </div>""", unsafe_allow_html=True)

        # ── SHAP ────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">🔍 AI Explainability</p>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP values..."):
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)
        fig_shap, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        fig_s = plt.gcf()
        fig_s.patch.set_facecolor('#0d1b2a')
        for ax in fig_s.axes:
            ax.set_facecolor('#0d1b2a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for text in ax.texts:
                text.set_color('white')
        st.pyplot(fig_s)
        plt.close()

        # ── Recommendations ──────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">📋 Recommended Actions</p>', unsafe_allow_html=True)
        if prob > 0.7:
            st.error("🚨 IMMEDIATE ACTION REQUIRED")
            st.markdown("- 🛑 Stop machine immediately\n- 📞 Contact maintenance team\n- 🔩 Replace tool now\n- 📝 Log incident")
        elif prob > 0.4:
            st.warning("⚠️ SCHEDULE MAINTENANCE SOON")
            st.markdown("- 📅 Maintenance within 24hrs\n- 👁️ Monitor sensors closely\n- 🔩 Plan tool replacement\n- 📊 Increase monitoring")
        else:
            st.success("✅ MACHINE IS HEALTHY")
            st.markdown("- ✅ Continue normal operation\n- 📅 Next scheduled maintenance on time\n- 📊 All sensors normal")

        # Email alert
        if email_btn and alert_email and pred == 1:
            st.warning(f"⚠️ Alert would be sent to {alert_email} (configure SMTP to enable)")

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
            <h1 style="font-size:4em">⚙️</h1>
            <h2 style="color:#64ffda;font-family:'Orbitron'">SYSTEM READY</h2>
            <p style="color:#8892b0;font-size:1.2em">Adjust sensor values in the sidebar<br>and click <span style="color:#ff6b35">Analyze Machine Health</span></p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Multi-Machine Dashboard
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header">🏭 Multi-Machine Production Floor</p>', unsafe_allow_html=True)

    # Simulate 6 machines
    np.random.seed(42)
    machines = []
    for i in range(1, 7):
        at = round(random.uniform(296, 304), 1)
        pt = round(random.uniform(306, 314), 1)
        r = random.randint(1200, 2800)
        t = round(random.uniform(5, 75), 1)
        tw = random.randint(10, 250)
        te = random.randint(0, 2)
        inp = pd.DataFrame([[te, at, pt, r, t, tw]], columns=FEATURE_NAMES)
        p = model.predict_proba(inp)[0][1]
        machines.append({
            'id': f'M-{i:03d}', 'prob': p,
            'air_temp': at, 'process_temp': pt,
            'rpm': r, 'torque': t, 'tool_wear': tw,
            'status': '🔴 CRITICAL' if p > 0.7 else '🟡 WARNING' if p > 0.4 else '🟢 HEALTHY'
        })

    # Summary metrics
    critical = sum(1 for m in machines if m['prob'] > 0.7)
    warning = sum(1 for m in machines if 0.4 < m['prob'] <= 0.7)
    healthy = sum(1 for m in machines if m['prob'] <= 0.4)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", len(machines))
    c2.metric("🔴 Critical", critical, delta=f"{critical} need attention")
    c3.metric("🟡 Warning", warning)
    c4.metric("🟢 Healthy", healthy)

    st.markdown("---")

    # Machine cards
    cols = st.columns(3)
    for i, m in enumerate(machines):
        with cols[i % 3]:
            color = "#ff4444" if m['prob'] > 0.7 else "#ffaa00" if m['prob'] > 0.4 else "#00ff88"
            st.markdown(f"""
            <div class="machine-card" style="border:1px solid {color}; box-shadow: 0 0 15px {color}33">
                <h3 style="color:{color};font-family:'Orbitron'">⚙ {m['id']}</h3>
                <h2 style="color:{color}">{m['prob']*100:.1f}%</h2>
                <p style="color:#8892b0">{m['status']}</p>
                <small style="color:#64ffda">RPM: {m['rpm']} | Torque: {m['torque']}Nm</small><br>
                <small style="color:#64ffda">Tool Wear: {m['tool_wear']}min | ΔT: {m['process_temp']-m['air_temp']:.1f}K</small>
            </div>""", unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")

    # Fleet overview chart
    fig_fleet = go.Figure()
    colors_fleet = ['#ff4444' if m['prob'] > 0.7 else '#ffaa00' if m['prob'] > 0.4 else '#00ff88'
                    for m in machines]
    fig_fleet.add_trace(go.Bar(
        x=[m['id'] for m in machines],
        y=[m['prob'] * 100 for m in machines],
        marker=dict(color=colors_fleet, line=dict(color='rgba(0,0,0,0)')),
        text=[f"{m['prob']*100:.1f}%" for m in machines],
        textposition='outside',
        textfont=dict(color='white')
    ))
    fig_fleet.add_hline(y=70, line_dash="dash", line_color="#ff4444",
                        annotation_text="Critical threshold (70%)")
    fig_fleet.add_hline(y=40, line_dash="dash", line_color="#ffaa00",
                        annotation_text="Warning threshold (40%)")
    fig_fleet.update_layout(
        title=dict(text='Fleet Risk Overview', font=dict(color='white', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.5)',
        font={'color': 'white'},
        xaxis=dict(color='white'),
        yaxis=dict(color='white', title='Failure Risk %', range=[0, 110]),
        height=350
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Trend Analysis
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">📈 Historical Sensor Trends</p>', unsafe_allow_html=True)

    # Generate 24hr simulated data
    hours = 24
    timestamps = [datetime.now() - timedelta(hours=hours-i) for i in range(hours)]

    np.random.seed(123)
    air_temps = [300 + np.random.normal(0, 1.5) for _ in range(hours)]
    process_temps = [310 + np.random.normal(0, 1.5) for _ in range(hours)]
    rpms = [1500 + np.random.normal(0, 150) for _ in range(hours)]
    torques = [40 + np.random.normal(0, 8) for _ in range(hours)]
    tool_wears = list(range(80, 80 + hours))

    # Simulate increasing risk
    risks = []
    for i in range(hours):
        inp = pd.DataFrame([[1, air_temps[i], process_temps[i],
                              rpms[i], torques[i], tool_wears[i]]],
                           columns=FEATURE_NAMES)
        risks.append(model.predict_proba(inp)[0][1] * 100)

    # Risk trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=timestamps, y=risks,
        mode='lines+markers',
        name='Failure Risk %',
        line=dict(color='#ff6b35', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,107,53,0.1)',
        marker=dict(size=6, color='#ff6b35')
    ))
    fig_trend.add_hline(y=50, line_dash="dash", line_color="#ff4444",
                        annotation_text="Risk threshold")
    fig_trend.update_layout(
        title=dict(text='24-Hour Risk Trend', font=dict(color='white', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.5)',
        font={'color': 'white'},
        xaxis=dict(color='white', title='Time'),
        yaxis=dict(color='white', title='Failure Risk %'),
        height=300
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Sensor trends
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=timestamps, y=air_temps,
                                       name='Air Temp', line=dict(color='#64ffda')))
        fig_temp.add_trace(go.Scatter(x=timestamps, y=process_temps,
                                       name='Process Temp', line=dict(color='#ff6b35')))
        fig_temp.update_layout(
            title=dict(text='Temperature Trends', font=dict(color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.5)',
            font={'color': 'white'}, height=250,
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        fig_mech = go.Figure()
        fig_mech.add_trace(go.Scatter(x=timestamps, y=torques,
                                       name='Torque (Nm)', line=dict(color='#ff6b35')))
        fig_mech.add_trace(go.Scatter(x=timestamps,
                                       y=[r/50 for r in rpms],
                                       name='RPM/50', line=dict(color='#bd93f9')))
        fig_mech.update_layout(
            title=dict(text='Mechanical Trends', font=dict(color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.5)',
            font={'color': 'white'}, height=250,
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_mech, use_container_width=True)

    # Tool wear trend
    fig_tool = go.Figure()
    fig_tool.add_trace(go.Scatter(
        x=timestamps, y=tool_wears,
        mode='lines+markers',
        name='Tool Wear (min)',
        line=dict(color='#ffaa00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,170,0,0.1)'
    ))
    fig_tool.add_hline(y=200, line_dash="dash", line_color="#ff4444",
                       annotation_text="Replace threshold (200 min)")
    fig_tool.update_layout(
        title=dict(text='Tool Wear Progression', font=dict(color='white', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.5)',
        font={'color': 'white'},
        xaxis=dict(color='white'), yaxis=dict(color='white'),
        height=250
    )
    st.plotly_chart(fig_tool, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PDF Report
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">📄 Generate Analysis Report</p>', unsafe_allow_html=True)

    if 'last_prediction' in st.session_state:
        p = st.session_state['last_prediction']
        st.success("✅ Prediction data available — ready to generate report!")

        st.markdown(f"""
        **Last Analysis:** {p['timestamp']}
        | **Machine Type:** {p['machine_type']}
        | **Risk Score:** {p['prob']*100:.1f}%
        | **Status:** {'⚠️ FAILURE RISK' if p['pred'] == 1 else '✅ HEALTHY'}
        """)

        if st.button("📥 Generate & Download PDF Report", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_fill_color(13, 27, 42)
            pdf.rect(0, 0, 210, 297, 'F')

            pdf.set_font("Helvetica", "B", 24)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 20, "BOSCH PREDICTIVE MAINTENANCE", ln=True, align='C')

            pdf.set_font("Helvetica", size=12)
            pdf.set_text_color(100, 255, 218)
            pdf.cell(0, 10, "Machine Health Analysis Report", ln=True, align='C')
            pdf.cell(0, 8, f"Generated: {p['timestamp']}", ln=True, align='C')

            pdf.set_draw_color(255, 107, 53)
            pdf.line(10, 45, 200, 45)
            pdf.ln(10)

            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "MACHINE PARAMETERS", ln=True)

            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(255, 255, 255)
            params = [
                ("Machine Type", p['machine_type']),
                ("Air Temperature", f"{p['air_temp']} K"),
                ("Process Temperature", f"{p['process_temp']} K"),
                ("Rotational Speed", f"{p['rpm']} RPM"),
                ("Torque", f"{p['torque']} Nm"),
                ("Tool Wear", f"{p['tool_wear']} min"),
            ]
            for label, value in params:
                pdf.cell(80, 8, label + ":", ln=False)
                pdf.set_text_color(100, 255, 218)
                pdf.cell(0, 8, value, ln=True)
                pdf.set_text_color(255, 255, 255)

            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "RISK ASSESSMENT", ln=True)

            pdf.set_font("Helvetica", "B", 20)
            risk_color = (255, 68, 68) if p['pred'] == 1 else (0, 255, 136)
            pdf.set_text_color(*risk_color)
            status = "⚠ FAILURE RISK DETECTED" if p['pred'] == 1 else "✓ MACHINE HEALTHY"
            pdf.cell(0, 12, status, ln=True)
            pdf.set_font("Helvetica", size=14)
            pdf.cell(0, 10, f"Failure Probability: {p['prob']*100:.1f}%", ln=True)

            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "RECOMMENDATION", ln=True)
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(255, 255, 255)
            if p['prob'] > 0.7:
                rec = "IMMEDIATE ACTION REQUIRED: Stop machine and contact maintenance team."
            elif p['prob'] > 0.4:
                rec = "Schedule maintenance within 24 hours. Monitor sensors closely."
            else:
                rec = "Continue normal operation. Next scheduled maintenance on time."
            pdf.multi_cell(0, 8, rec)

            pdf.ln(5)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, "Generated by Bosch Predictive Maintenance AI System", align='C')

            report_path = os.path.join(BASE_DIR, 'maintenance_report.pdf')
            pdf.output(report_path)

            with open(report_path, 'rb') as f:
                st.download_button(
                    "📥 Download PDF",
                    f.read(),
                    file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            st.success("✅ Report generated successfully!")
    else:
        st.info("👈 Go to **Single Machine** tab first and run a prediction, then come back here to download the report.")
        '''

#UPDATED CODE 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import time
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GUARDIAN — Smart Maintenance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Background ── */
.stApp {
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(255,107,53,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(100,255,218,0.05) 0%, transparent 50%),
        linear-gradient(135deg, #020818 0%, #050d1a 50%, #020818 100%);
    background-attachment: fixed;
}

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        linear-gradient(rgba(100,255,218,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(100,255,218,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── Fonts ── */
* { font-family: 'Rajdhani', sans-serif; }

/* ── Header ── */
.guardian-header {
    text-align: center;
    padding: 20px 0 10px;
    position: relative;
}

.guardian-logo {
    font-family: 'Orbitron', monospace;
    font-size: 3.2em;
    font-weight: 900;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 40%, #ff6b35 60%, #64ffda 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    letter-spacing: 4px;
    text-shadow: none;
    margin-bottom: 5px;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 300% center; }
}

.guardian-tagline {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85em;
    color: #64ffda;
    letter-spacing: 6px;
    text-transform: uppercase;
    opacity: 0.8;
}

.guardian-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff6b35, #64ffda, #ff6b35, transparent);
    margin: 15px 0;
    animation: glow-line 3s ease-in-out infinite;
}

@keyframes glow-line {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; box-shadow: 0 0 10px #ff6b35; }
}

/* ── Cards ── */
.g-card {
    background: linear-gradient(135deg, rgba(5,13,26,0.95), rgba(13,27,42,0.95));
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(100,255,218,0.12);
    backdrop-filter: blur(20px);
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.g-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, #64ffda, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.g-card:hover::before { opacity: 1; }
.g-card:hover {
    border-color: rgba(100,255,218,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(100,255,218,0.05);
}

.g-card-danger {
    background: linear-gradient(135deg, rgba(20,5,5,0.97), rgba(40,10,10,0.97));
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,68,68,0.4);
    animation: danger-pulse 2s ease-in-out infinite;
    text-align: center;
}

@keyframes danger-pulse {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(255,68,68,0.2), inset 0 0 20px rgba(255,68,68,0.05); 
        border-color: rgba(255,68,68,0.4);
    }
    50% { 
        box-shadow: 0 0 40px rgba(255,68,68,0.5), inset 0 0 30px rgba(255,68,68,0.1); 
        border-color: rgba(255,68,68,0.8);
    }
}

.g-card-safe {
    background: linear-gradient(135deg, rgba(5,20,12,0.97), rgba(10,40,22,0.97));
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(0,255,136,0.3);
    box-shadow: 0 0 25px rgba(0,255,136,0.1);
    text-align: center;
}

/* ── Section headers ── */
.g-section {
    font-family: 'Orbitron', monospace;
    font-size: 0.9em;
    font-weight: 700;
    color: #ff6b35;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 8px 0;
    margin: 20px 0 15px;
    border-bottom: 1px solid rgba(255,107,53,0.3);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Metrics ── */
.g-metric {
    text-align: center;
    padding: 8px;
}

.g-metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2em;
    font-weight: 700;
    line-height: 1;
}

.g-metric-label {
    font-size: 0.8em;
    color: #8892b0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75em;
    letter-spacing: 2px;
}

.status-online {
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.4);
    color: #00ff88;
}

.status-critical {
    background: rgba(255,68,68,0.1);
    border: 1px solid rgba(255,68,68,0.4);
    color: #ff4444;
}

/* ── Cost boxes ── */
.cost-danger {
    background: linear-gradient(135deg, rgba(5,13,26,0.9), rgba(13,27,42,0.9));
    border-left: 3px solid #ff4444;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}

.cost-safe {
    background: linear-gradient(135deg, rgba(5,13,26,0.9), rgba(13,27,42,0.9));
    border-left: 3px solid #00ff88;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}

/* ── Machine cards ── */
.machine-tile {
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    transition: all 0.3s ease;
    margin: 6px 0;
}
.machine-tile:hover {
    transform: translateY(-3px);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020818, #050d1a) !important;
    border-right: 1px solid rgba(100,255,218,0.08) !important;
}

section[data-testid="stSidebar"] * {
    color: #c8d6e5 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(5,13,26,0.8);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(100,255,218,0.1);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    color: #8892b0 !important;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(100,255,218,0.1)) !important;
    color: #ff6b35 !important;
    border: 1px solid rgba(255,107,53,0.3) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(100,255,218,0.1));
    border: 1px solid rgba(255,107,53,0.4);
    color: #ff6b35 !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 10px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(255,107,53,0.3), rgba(100,255,218,0.2));
    border-color: #ff6b35;
    box-shadow: 0 0 20px rgba(255,107,53,0.3);
    transform: translateY(-1px);
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: #ff6b35 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020818; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(#ff6b35, #64ffda); 
    border-radius: 3px; 
}

/* ── Live indicator ── */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: blink 1.5s ease-in-out infinite;
    margin-right: 6px;
}

@keyframes blink {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #00ff88; }
    50% { opacity: 0.3; box-shadow: none; }
}

/* ── Main content padding ── */
.main .block-container {
    padding: 1rem 2rem 2rem;
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
features = pickle.load(open(os.path.join(BASE_DIR, 'features.pkl'), 'rb'))

FEATURE_NAMES = [
    'Type', 'Air temperature _K', 'Process temperature _K',
    'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min'
]

# ── Header ───────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 6, 1])

with col_logo:
    if os.path.exists(os.path.join(BASE_DIR, 'logo.png')):
        st.image(os.path.join(BASE_DIR, 'logo.png'), width=80)

with col_title:
    st.markdown("""
    <div class="guardian-header">
        <div class="guardian-logo">🛡️ GUARDIAN</div>
        <div class="guardian-tagline">Smart Industrial Maintenance Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="text-align:right;padding-top:20px">
        <span class="live-dot"></span>
        <span style="color:#00ff88;font-family:'Share Tech Mono';font-size:0.8em">LIVE</span><br>
        <span style="color:#8892b0;font-size:0.7em;font-family:'Share Tech Mono'">SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="guardian-divider"></div>', unsafe_allow_html=True)

# ── Navigation tabs ───────────────────────────────────────────
tabs = st.tabs(["🔍 Single Machine", "🏭 Fleet Dashboard", "📈 Trend Analysis", "📄 Report"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Single Machine
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0">
            <div style="font-family:'Orbitron';color:#ff6b35;font-size:1em;letter-spacing:2px">⚙ SENSOR PANEL</div>
            <div style="color:#8892b0;font-size:0.7em;letter-spacing:1px">REAL-TIME INPUT</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        machine_type = st.selectbox("🏭 Machine Grade", ["L — Low", "M — Medium", "H — High"])
        type_encoded = {"L — Low": 0, "M — Medium": 1, "H — High": 2}[machine_type]

        st.markdown("**🌡️ Temperature Sensors**")
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)

        st.markdown("**⚡ Mechanical Sensors**")
        rpm = st.slider("Rotational Speed (RPM)", 1168, 2886, 1500)
        torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)

        st.markdown("**🔩 Tool Condition**")
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)

        st.markdown("---")
        predict_btn = st.button("⚡ ANALYZE MACHINE HEALTH", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Share Tech Mono';font-size:0.7em;color:#8892b0;line-height:2">
        NORMAL RANGES<br>
        ─────────────<br>
        Air: 295–305 K<br>
        Process: 305–315 K<br>
        RPM: 1168–2886<br>
        Torque: 3.8–76.6 Nm<br>
        Wear: 0–253 min
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**📧 Alert Configuration**")
        alert_email = st.text_input("Notification Email", placeholder="engineer@company.com")

    if predict_btn:
        input_data = pd.DataFrame(
            [[type_encoded, air_temp, process_temp, rpm, torque, tool_wear]],
            columns=FEATURE_NAMES
        )

        with st.spinner("⚡ Running AI diagnostics..."):
            time.sleep(0.8)

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.session_state['last_prediction'] = {
            'prob': prob, 'pred': pred,
            'air_temp': air_temp, 'process_temp': process_temp,
            'rpm': rpm, 'torque': torque, 'tool_wear': tool_wear,
            'machine_type': machine_type.replace("—", "-").replace("–", "-"),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # ── Risk cards ───────────────────────────────────────
        st.markdown('<div class="g-section">📊 RISK ASSESSMENT</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="g-card-danger">
                    <div style="font-family:'Share Tech Mono';color:#ff4444;font-size:0.7em;letter-spacing:3px">⚠ FAILURE DETECTED</div>
                    <div style="font-family:'Orbitron';color:#ff4444;font-size:3.5em;font-weight:900;line-height:1.1">{prob*100:.1f}<span style="font-size:0.4em">%</span></div>
                    <div style="color:#ff9999;font-size:0.85em;margin-top:8px">FAILURE PROBABILITY</div>
                    <div style="margin-top:12px"><span class="status-badge status-critical">● CRITICAL</span></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="g-card-safe">
                    <div style="font-family:'Share Tech Mono';color:#00ff88;font-size:0.7em;letter-spacing:3px">✓ SYSTEM NOMINAL</div>
                    <div style="font-family:'Orbitron';color:#00ff88;font-size:3.5em;font-weight:900;line-height:1.1">{prob*100:.1f}<span style="font-size:0.4em">%</span></div>
                    <div style="color:#99ffcc;font-size:0.85em;margin-top:8px">FAILURE PROBABILITY</div>
                    <div style="margin-top:12px"><span class="status-badge status-online">● HEALTHY</span></div>
                </div>""", unsafe_allow_html=True)

        with col2:
            wear_pct = tool_wear / 253 * 100
            wear_color = "#ff4444" if wear_pct > 80 else "#ffaa00" if wear_pct > 60 else "#00ff88"
            st.markdown(f"""
            <div class="g-card" style="text-align:center">
                <div style="font-family:'Share Tech Mono';color:#8892b0;font-size:0.7em;letter-spacing:2px">TOOL CONDITION</div>
                <div style="font-family:'Orbitron';color:{wear_color};font-size:3em;font-weight:700;line-height:1.2">{wear_pct:.0f}<span style="font-size:0.4em">%</span></div>
                <div style="color:#8892b0;font-size:0.8em">LIFESPAN CONSUMED</div>
                <div style="margin-top:10px;background:rgba(255,255,255,0.05);border-radius:4px;height:6px">
                    <div style="width:{wear_pct}%;background:{wear_color};height:6px;border-radius:4px;transition:width 1s"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col3:
            temp_diff = process_temp - air_temp
            temp_color = "#ff4444" if temp_diff > 11 else "#ffaa00" if temp_diff > 10 else "#00ff88"
            st.markdown(f"""
            <div class="g-card" style="text-align:center">
                <div style="font-family:'Share Tech Mono';color:#8892b0;font-size:0.7em;letter-spacing:2px">THERMAL DELTA</div>
                <div style="font-family:'Orbitron';color:{temp_color};font-size:3em;font-weight:700;line-height:1.2">{temp_diff:.1f}<span style="font-size:0.4em">K</span></div>
                <div style="color:#8892b0;font-size:0.8em">PROCESS vs AIR TEMP</div>
                <div style="margin-top:10px;color:{temp_color};font-size:0.75em">
                    {'⚠ ABOVE THRESHOLD' if temp_diff > 11 else '✓ WITHIN RANGE'}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauge + Radar ────────────────────────────────────
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "FAILURE RISK SCORE",
                       'font': {'color': '#8892b0', 'size': 14, 'family': 'Rajdhani'}},
                number={'suffix': "%", 'font': {'color': 'white', 'size': 40,
                                                 'family': 'Orbitron'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#8892b0",
                              'tickfont': {'color': '#8892b0'}},
                    'bar': {'color': "#ff4444" if prob > 0.5 else "#ffaa00" if prob > 0.3 else "#00ff88",
                             'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'bordercolor': "rgba(100,255,218,0.1)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,136,0.08)'},
                        {'range': [30, 60], 'color': 'rgba(255,170,0,0.08)'},
                        {'range': [60, 100], 'color': 'rgba(255,68,68,0.08)'}
                    ],
                    'threshold': {
                        'line': {'color': "rgba(255,255,255,0.3)", 'width': 2},
                        'thickness': 0.75, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(5,13,26,0.8)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Rajdhani'},
                height=280,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            categories = ['Air Temp', 'Process Temp', 'Speed', 'Torque', 'Tool Wear']
            values = [
                (air_temp - 295) / 10 * 100,
                (process_temp - 305) / 10 * 100,
                (rpm - 1168) / (2886 - 1168) * 100,
                (torque - 3.8) / (76.6 - 3.8) * 100,
                tool_wear / 253 * 100
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(255,107,53,0.1)',
                line=dict(color='#ff6b35', width=2),
                marker=dict(color='#ff6b35', size=6)
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[50] * 6,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(100,255,218,0.03)',
                line=dict(color='rgba(100,255,218,0.2)', width=1, dash='dot'),
                name='Normal'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    color='#8892b0', gridcolor='rgba(255,255,255,0.05)'),
                    angularaxis=dict(color='#8892b0',
                                     gridcolor='rgba(255,255,255,0.05)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(5,13,26,0.8)',
                font={'color': '#8892b0', 'family': 'Rajdhani'},
                title=dict(text='SENSOR HEALTH RADAR',
                           font=dict(color='#8892b0', size=12, family='Orbitron')),
                height=280,
                margin=dict(t=60, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Failure type chart ───────────────────────────────
        st.markdown('<div class="g-section">🔬 FAILURE TYPE ANALYSIS</div>', unsafe_allow_html=True)
        failure_types = {
            'Tool Wear': min(100, tool_wear / 253 * 100 * 1.2),
            'Heat Dissipation': min(100, max(0, (temp_diff - 8.6) / 3 * 100)),
            'Power Failure': min(100, max(0, (torque * rpm / 9550 - 3.5) / 6 * 100)),
            'Overstrain': min(100, max(0, (torque * tool_wear - 11000) / 3000 * 100)),
            'Random': 3.0
        }
        colors = ['#ff4444' if v > 50 else '#ffaa00' if v > 25 else '#00ff88'
                  for v in failure_types.values()]
        fig_bar = go.Figure(go.Bar(
            x=list(failure_types.values()),
            y=list(failure_types.keys()),
            orientation='h',
            marker=dict(color=colors,
                        line=dict(color='rgba(0,0,0,0)'),
                        opacity=0.85),
            text=[f'{v:.1f}%' for v in failure_types.values()],
            textposition='outside',
            textfont=dict(color='white', family='Share Tech Mono', size=11)
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(5,13,26,0.8)',
            plot_bgcolor='rgba(13,27,42,0.3)',
            font={'color': '#8892b0', 'family': 'Rajdhani'},
            xaxis=dict(range=[0, 120], color='#8892b0',
                       gridcolor='rgba(255,255,255,0.04)', title='Risk %'),
            yaxis=dict(color='white', tickfont=dict(size=12)),
            height=260,
            margin=dict(t=20, b=30, l=10, r=60)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Cost estimator ───────────────────────────────────
        st.markdown('<div class="g-section">💰 MAINTENANCE COST ANALYSIS</div>', unsafe_allow_html=True)
        unplanned = int(prob * 8 * 500)
        repair = int(prob * 15000)
        loss = int(prob * 25000)
        total_fail = unplanned + repair + loss
        prev_service = 2500
        prev_parts = int(tool_wear / 253 * 3000)
        total_prev = prev_service + prev_parts
        savings = total_fail - total_prev

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="cost-danger">
                <div style="font-family:'Share Tech Mono';color:#ff4444;font-size:0.75em;letter-spacing:2px;margin-bottom:12px">⚠ COST IF IGNORED</div>
                <div style="display:flex;justify-content:space-between;margin:6px 0"><span style="color:#8892b0">Downtime Cost</span><span style="color:#ff6b35;font-family:'Share Tech Mono'">₹{unplanned:,}</span></div>
                <div style="display:flex;justify-content:space-between;margin:6px 0"><span style="color:#8892b0">Emergency Repair</span><span style="color:#ff6b35;font-family:'Share Tech Mono'">₹{repair:,}</span></div>
                <div style="display:flex;justify-content:space-between;margin:6px 0"><span style="color:#8892b0">Production Loss</span><span style="color:#ff6b35;font-family:'Share Tech Mono'">₹{loss:,}</span></div>
                <div style="border-top:1px solid rgba(255,68,68,0.2);margin-top:10px;padding-top:10px;display:flex;justify-content:space-between">
                    <span style="color:white;font-weight:700">TOTAL RISK COST</span>
                    <span style="color:#ff4444;font-family:'Orbitron';font-size:1.2em">₹{total_fail:,}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="cost-safe">
                <div style="font-family:'Share Tech Mono';color:#00ff88;font-size:0.75em;letter-spacing:2px;margin-bottom:12px">✓ PREVENTIVE COST</div>
                <div style="display:flex;justify-content:space-between;margin:6px 0"><span style="color:#8892b0">Scheduled Service</span><span style="color:#00ff88;font-family:'Share Tech Mono'">₹{prev_service:,}</span></div>
                <div style="display:flex;justify-content:space-between;margin:6px 0"><span style="color:#8892b0">Parts Replacement</span><span style="color:#00ff88;font-family:'Share Tech Mono'">₹{prev_parts:,}</span></div>
                <div style="margin:6px 0;height:24px"></div>
                <div style="border-top:1px solid rgba(0,255,136,0.2);margin-top:10px;padding-top:10px">
                    <div style="display:flex;justify-content:space-between">
                        <span style="color:white;font-weight:700">TOTAL PREVENTIVE</span>
                        <span style="color:#00ff88;font-family:'Orbitron';font-size:1.2em">₹{total_prev:,}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:8px">
                        <span style="color:#ffaa00;font-weight:700">💡 POTENTIAL SAVINGS</span>
                        <span style="color:#ffaa00;font-family:'Orbitron';font-size:1.2em">₹{savings:,}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── SHAP ─────────────────────────────────────────────
        st.markdown('<div class="g-section">🔍 AI EXPLAINABILITY</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP values..."):
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)
        plt.style.use('default')
        fig_shap, _ = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        fig_s = plt.gcf()
        fig_s.patch.set_facecolor('#0d1b2a')
        for ax in fig_s.axes:
            ax.set_facecolor('#0d1b2a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for text in ax.texts:
                text.set_color('white')
        st.pyplot(fig_s)
        plt.close()
        


        # ── Recommendations ──────────────────────────────────
        st.markdown('<div class="g-section">📋 RECOMMENDED ACTIONS</div>', unsafe_allow_html=True)
        if prob > 0.7:
            st.error("🚨 CRITICAL — IMMEDIATE ACTION REQUIRED")
            st.markdown("- 🛑 **Stop machine immediately**\n- 📞 Contact maintenance team now\n- 🔩 Replace tool — critical wear level\n- 📝 Log incident in maintenance system")
        elif prob > 0.4:
            st.warning("⚠️ WARNING — SCHEDULE MAINTENANCE SOON")
            st.markdown("- 📅 Schedule maintenance within 24 hours\n- 👁️ Monitor all sensors closely\n- 🔩 Plan tool replacement at next service\n- 📊 Increase monitoring frequency")
        else:
            st.success("✅ NOMINAL — CONTINUE NORMAL OPERATION")
            st.markdown("- ✅ All systems within normal parameters\n- 📅 Next scheduled maintenance on time\n- 📊 Continue standard monitoring\n- 🔍 No immediate action required")

    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px">
            <div style="font-size:5em;margin-bottom:20px">🛡️</div>
            <div style="font-family:'Orbitron';color:#64ffda;font-size:1.8em;letter-spacing:4px;margin-bottom:10px">GUARDIAN READY</div>
            <div style="color:#8892b0;font-size:1.1em;line-height:1.8">
                Configure sensor parameters in the sidebar<br>
                and click <span style="color:#ff6b35;font-weight:700">ANALYZE MACHINE HEALTH</span><br>
                to begin diagnostics
            </div>
            <div style="margin-top:30px;font-family:'Share Tech Mono';color:#3a4a5a;font-size:0.8em;letter-spacing:3px">
                [ AWAITING INPUT ]
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Fleet Dashboard
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="g-section">🏭 PRODUCTION FLOOR MONITOR</div>', unsafe_allow_html=True)

    random.seed(42)
    machines = []
    for i in range(1, 7):
        at = round(random.uniform(296, 304), 1)
        pt = round(random.uniform(306, 314), 1)
        r = random.randint(1200, 2800)
        t = round(random.uniform(5, 75), 1)
        tw = random.randint(10, 250)
        te = random.randint(0, 2)
        inp = pd.DataFrame([[te, at, pt, r, t, tw]], columns=FEATURE_NAMES)
        p = model.predict_proba(inp)[0][1]
        machines.append({
            'id': f'UNIT-{i:03d}', 'prob': p,
            'air_temp': at, 'process_temp': pt,
            'rpm': r, 'torque': t, 'tool_wear': tw,
            'status': 'CRITICAL' if p > 0.7 else 'WARNING' if p > 0.4 else 'NOMINAL'
        })

    critical = sum(1 for m in machines if m['prob'] > 0.7)
    warning = sum(1 for m in machines if 0.4 < m['prob'] <= 0.7)
    healthy = sum(1 for m in machines if m['prob'] <= 0.4)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚙ Total Units", len(machines))
    c2.metric("🔴 Critical", critical)
    c3.metric("🟡 Warning", warning)
    c4.metric("🟢 Nominal", healthy)

    st.markdown("---")
    cols = st.columns(3)
    for i, m in enumerate(machines):
        with cols[i % 3]:
            color = "#ff4444" if m['prob'] > 0.7 else "#ffaa00" if m['prob'] > 0.4 else "#00ff88"
            status_icon = "🔴" if m['prob'] > 0.7 else "🟡" if m['prob'] > 0.4 else "🟢"
            st.markdown(f"""
            <div class="machine-tile" style="background:linear-gradient(135deg,rgba(5,13,26,0.95),rgba(13,27,42,0.95));border:1px solid {color}33;box-shadow:0 0 20px {color}1a">
                <div style="font-family:'Orbitron';color:{color};font-size:1.1em;letter-spacing:2px">{status_icon} {m['id']}</div>
                <div style="font-family:'Orbitron';color:{color};font-size:2.5em;font-weight:700;margin:8px 0">{m['prob']*100:.1f}%</div>
                <div style="font-family:'Share Tech Mono';color:#8892b0;font-size:0.7em;letter-spacing:2px">{m['status']}</div>
                <div style="margin-top:10px;font-size:0.78em;color:#64ffda;font-family:'Share Tech Mono'">
                    RPM {m['rpm']} · {m['torque']}Nm<br>
                    WEAR {m['tool_wear']}min · ΔT {m['process_temp']-m['air_temp']:.1f}K
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")
    colors_fleet = ['#ff4444' if m['prob'] > 0.7 else '#ffaa00' if m['prob'] > 0.4 else '#00ff88'
                    for m in machines]
    fig_fleet = go.Figure()
    fig_fleet.add_trace(go.Bar(
        x=[m['id'] for m in machines],
        y=[m['prob'] * 100 for m in machines],
        marker=dict(color=colors_fleet, opacity=0.85,
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f"{m['prob']*100:.1f}%" for m in machines],
        textposition='outside',
        textfont=dict(color='white', family='Share Tech Mono')
    ))
    fig_fleet.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.5)",
                        annotation_text="Critical", annotation_font_color="#ff4444")
    fig_fleet.add_hline(y=40, line_dash="dash", line_color="rgba(255,170,0,0.5)",
                        annotation_text="Warning", annotation_font_color="#ffaa00")
    fig_fleet.update_layout(
        title=dict(text='FLEET RISK OVERVIEW',
                   font=dict(color='#8892b0', size=13, family='Orbitron')),
        paper_bgcolor='rgba(5,13,26,0.8)',
        plot_bgcolor='rgba(13,27,42,0.3)',
        font={'color': '#8892b0', 'family': 'Rajdhani'},
        xaxis=dict(color='#8892b0', gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(color='#8892b0', title='Risk %', range=[0, 110],
                   gridcolor='rgba(255,255,255,0.04)'),
        height=320,
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Trend Analysis
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="g-section">📈 24-HOUR SENSOR TRENDS</div>', unsafe_allow_html=True)

    hours = 24
    timestamps = [datetime.now() - timedelta(hours=hours-i) for i in range(hours)]
    np.random.seed(123)
    air_temps = [300 + np.random.normal(0, 1.5) for _ in range(hours)]
    process_temps = [310 + np.random.normal(0, 1.5) for _ in range(hours)]
    rpms = [1500 + np.random.normal(0, 150) for _ in range(hours)]
    torques = [40 + np.random.normal(0, 8) for _ in range(hours)]
    tool_wears = list(range(80, 80 + hours))

    risks = []
    for i in range(hours):
        inp = pd.DataFrame([[1, air_temps[i], process_temps[i],
                              rpms[i], torques[i], tool_wears[i]]],
                           columns=FEATURE_NAMES)
        risks.append(model.predict_proba(inp)[0][1] * 100)

    def make_plot(y_data, title, color, y_title, threshold=None, threshold_label=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_data,
            mode='lines',
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)'
        ))
        if threshold:
            fig.add_hline(y=threshold, line_dash="dash",
                          line_color="rgba(255,68,68,0.4)",
                          annotation_text=threshold_label,
                          annotation_font_color="#ff4444")
        fig.update_layout(
            title=dict(text=title, font=dict(color='#8892b0', size=11, family='Orbitron')),
            paper_bgcolor='rgba(5,13,26,0.8)',
            plot_bgcolor='rgba(13,27,42,0.3)',
            font={'color': '#8892b0'},
            xaxis=dict(color='#8892b0', gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(color='#8892b0', title=y_title,
                       gridcolor='rgba(255,255,255,0.03)'),
            height=220,
            margin=dict(t=40, b=20, l=10, r=10),
            showlegend=False
        )
        return fig

    st.plotly_chart(make_plot(risks, 'FAILURE RISK TREND', '#ff6b35', 'Risk %',
                               threshold=50, threshold_label='Alert threshold'),
                    use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=timestamps, y=air_temps,
                                    name='Air', line=dict(color='#64ffda', width=2)))
        fig_t.add_trace(go.Scatter(x=timestamps, y=process_temps,
                                    name='Process', line=dict(color='#ff6b35', width=2)))
        fig_t.update_layout(
            title=dict(text='TEMPERATURE', font=dict(color='#8892b0', size=11, family='Orbitron')),
            paper_bgcolor='rgba(5,13,26,0.8)',
            plot_bgcolor='rgba(13,27,42,0.3)',
            font={'color': '#8892b0'}, height=220,
            legend=dict(font=dict(color='#8892b0')),
            margin=dict(t=40, b=20),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)')
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.plotly_chart(make_plot(torques, 'TORQUE', '#bd93f9', 'Nm'),
                        use_container_width=True)

    st.plotly_chart(make_plot(tool_wears, 'TOOL WEAR PROGRESSION', '#ffaa00', 'Minutes',
                               threshold=200, threshold_label='Replace threshold'),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — Report
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="g-section">📄 ANALYSIS REPORT</div>', unsafe_allow_html=True)

    if 'last_prediction' in st.session_state:
        p = st.session_state['last_prediction']
        st.markdown(f"""
        <div class="g-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="font-family:'Orbitron';color:#ff6b35;font-size:0.9em;letter-spacing:2px">LAST ANALYSIS</div>
                    <div style="color:#8892b0;font-family:'Share Tech Mono';font-size:0.8em;margin-top:4px">{p['timestamp']}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-family:'Orbitron';color:{'#ff4444' if p['pred']==1 else '#00ff88'};font-size:1.5em">{p['prob']*100:.1f}%</div>
                    <div style="color:#8892b0;font-size:0.8em">{'⚠ FAILURE RISK' if p['pred']==1 else '✓ HEALTHY'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("📥 GENERATE PDF REPORT", use_container_width=True):
            def c(text):
                return str(text).encode('ascii', 'ignore').decode('ascii').strip()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_fill_color(2, 8, 24)
            pdf.rect(0, 0, 210, 297, 'F')
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 20, "GUARDIAN", ln=True, align='C')
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(100, 255, 218)
            pdf.cell(0, 8, "Smart Industrial Maintenance Intelligence", ln=True, align='C')
            pdf.cell(0, 6, f"Report Generated: {p['timestamp']}", ln=True, align='C')
            pdf.set_draw_color(255, 107, 53)
            pdf.line(10, 42, 200, 42)
            pdf.ln(8)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "MACHINE PARAMETERS", ln=True)
            pdf.set_font("Helvetica", size=10)
            params = [
                ("Machine Grade", p['machine_type']),
                ("Air Temperature", f"{p['air_temp']} K"),
                ("Process Temperature", f"{p['process_temp']} K"),
                ("Rotational Speed", f"{p['rpm']} RPM"),
                ("Torque", f"{p['torque']} Nm"),
                ("Tool Wear", f"{p['tool_wear']} min"),
            ]
            for label, value in params:
                value = str(value).encode("ascii", "ignore").decode("ascii").strip()
                pdf.set_text_color(136, 146, 176)
                pdf.cell(80, 7, label + ":", ln=False)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 7, value, ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "RISK ASSESSMENT", ln=True)
            pdf.set_font("Helvetica", "B", 18)
            r, g, b = (255, 68, 68) if p['pred'] == 1 else (0, 255, 136)
            pdf.set_text_color(r, g, b)
            pdf.cell(0, 12, f"{'FAILURE RISK DETECTED' if p['pred']==1 else 'SYSTEM NOMINAL'}", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 8, f"Failure Probability: {p['prob']*100:.1f}%", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "RECOMMENDATION", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(255, 255, 255)
            if p['prob'] > 0.7:
                rec = "IMMEDIATE ACTION: Stop machine and contact maintenance team now."
            elif p['prob'] > 0.4:
                rec = "Schedule maintenance within 24 hours. Monitor sensors closely."
            else:
                rec = "Continue normal operation. All parameters within acceptable range."
            pdf.multi_cell(0, 7, c(rec))
            pdf.ln(5)

            # Cost Analysis
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "COST ANALYSIS", ln=True)

            unplanned = int(p['prob'] * 8 * 500)
            repair = int(p['prob'] * 15000)
            loss = int(p['prob'] * 25000)
            total_fail = unplanned + repair + loss
            prev_service = 2500
            prev_parts = int(p['tool_wear'] / 253 * 3000)
            total_prev = prev_service + prev_parts
            savings = total_fail - total_prev

            cost_items = [
                ("Failure Cost (if ignored)", f"Rs {total_fail:,}"),
                ("Preventive Cost", f"Rs {total_prev:,}"),
                ("Potential Savings", f"Rs {savings:,}"),
            ]
            pdf.set_font("Helvetica", size=10)
            for label, value in cost_items:
                pdf.set_text_color(136, 146, 176)
                pdf.cell(80, 7, c(label) + ":", ln=False)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 7, c(value), ln=True)

            # Failure Type Analysis
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(255, 107, 53)
            pdf.cell(0, 10, "FAILURE TYPE ANALYSIS", ln=True)

            temp_diff = p['process_temp'] - p['air_temp']
            failure_types = {
                "Tool Wear": min(100, p['tool_wear'] / 253 * 100 * 1.2),
                "Heat Dissipation": min(100, max(0, (temp_diff - 8.6) / 3 * 100)),
                "Power Failure": min(100, max(0, (p['torque'] * p['rpm'] / 9550 - 3.5) / 6 * 100)),
                "Overstrain": min(100, max(0, (p['torque'] * p['tool_wear'] - 11000) / 3000 * 100)),
                "Random": 3.0
            }
            pdf.set_font("Helvetica", size=10)
            for fname, fval in failure_types.items():
                color = (255, 68, 68) if fval > 50 else (255, 170, 0) if fval > 25 else (0, 255, 136)
                pdf.set_text_color(136, 146, 176)
                pdf.cell(80, 7, c(fname) + ":", ln=False)
                pdf.set_text_color(*color)
                pdf.cell(0, 7, f"{fval:.1f}%", ln=True)

            pdf.ln(10)
            # Footer
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(60, 70, 90)
            pdf.cell(0, 6, "GUARDIAN - Smart Industrial Maintenance Intelligence System", align='C')

            report_path = os.path.join(BASE_DIR, 'guardian_report.pdf')
            pdf.output(report_path)
            with open(report_path, 'rb') as f:
                st.download_button(
                    "📥 Download Report",
                    f.read(),
                    file_name=f"GUARDIAN_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            st.success("✅ Report generated!")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#8892b0">
            <div style="font-size:3em">📄</div>
            <div style="font-family:'Orbitron';margin:10px 0;color:#3a4a5a">NO DATA YET</div>
            <div>Run an analysis in the Single Machine tab first</div>
        </div>
        """, unsafe_allow_html=True)