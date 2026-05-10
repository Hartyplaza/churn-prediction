import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY   = "#4f9fff"
SECONDARY = "#a78bfa"
SUCCESS   = "#34d399"
WARNING   = "#fbbf24"
DANGER    = "#f87171"
BG_CARD   = "#1e2235"
BG_PAGE   = "#141824"
BORDER    = "#2d3a5c"
TEXT_PRI  = "#e8eaf6"
TEXT_MUT  = "#94a3b8"
TEXT_HEAD = "#ffffff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI} !important;
}}
.stApp {{ background-color: {BG_PAGE} !important; }}
section[data-testid="stSidebar"] {{
    background-color: #0f1320 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
.hero-banner {{
    background: linear-gradient(135deg, #1a1f36 0%, #0d1117 60%, #1a2744 100%);
    border: 1px solid {BORDER}; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}}
.hero-title {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}}
.hero-subtitle {{ color: {TEXT_MUT}; font-size: 0.95rem; margin-top: 0.4rem; }}
.metric-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}}
.metric-label {{ color: {TEXT_MUT}; font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
.metric-value {{ font-size: 1.8rem; font-weight: 700; color: {TEXT_PRI}; }}
.metric-accent {{ font-size: 1.8rem; font-weight: 700; color: {PRIMARY}; }}
.section-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}}
.section-title {{
    color: {PRIMARY}; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}
.nav-header {{ color: {TEXT_MUT}; font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em; padding: 0.5rem 0 0.3rem; }}
.result-high   {{ background: #2a1010; border: 1px solid {DANGER};  border-radius: 16px; padding: 2rem; text-align: center; }}
.result-medium {{ background: #2a1f00; border: 1px solid {WARNING}; border-radius: 16px; padding: 2rem; text-align: center; }}
.result-low    {{ background: #0d2016; border: 1px solid {SUCCESS}; border-radius: 16px; padding: 2rem; text-align: center; }}
.result-prob   {{ font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }}
.insight-box {{
    background: #1a2340; border-left: 3px solid {PRIMARY};
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin-top: 0.8rem; color: {TEXT_PRI}; font-size: 0.85rem; line-height: 1.6;
}}
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-weight: 600 !important;
    font-size: 1rem !important; width: 100% !important;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {{ color: {TEXT_MUT} !important; font-size: 0.8rem !important; }}
div[data-testid="stRadio"] label {{ color: {TEXT_PRI} !important; font-size: 0.88rem !important; }}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_pipeline.pkl')
    return joblib.load(model_path)

def engineer_features(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                                labels=['new','developing','established','loyal'])
    service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    df['num_addons'] = df[service_cols].apply(lambda row: (row=='Yes').sum(), axis=1)
    df['risky_payment'] = ((df['PaperlessBilling']=='Yes') &
                           (df['PaymentMethod']=='Electronic check')).astype(int)
    df['no_support'] = ((df['OnlineSecurity']=='No') &
                        (df['TechSupport']=='No')).astype(int)
    df['is_monthly'] = (df['Contract']=='Month-to-month').astype(int)
    df.drop(columns=['TotalCharges'], inplace=True)
    return df

def pcfg(title="", h=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRI, family="Inter"), height=h,
        margin=dict(t=45 if title else 20, b=30, l=10, r=10),
        title=dict(text=title, font=dict(color=TEXT_HEAD, size=14)) if title else None,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRI)),
    )

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0'>ChurnGuard AI</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.8rem;margin-top:0'>Customer Retention Intelligence</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p class="nav-header">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", options=[
        "Project Overview", "Dashboard & Plots",
        "Prediction", "Engineered Features", "Model Metrics",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown(f'<p class="nav-header">Model Info</p>', unsafe_allow_html=True)
    status_color = SUCCESS if model_loaded else DANGER
    status_text  = "Loaded" if model_loaded else "Not Found"
    for k,v in [("Algorithm","XGBoost + SMOTE"),("ROC-AUC","0.843"),
                ("Dataset","Telco Churn"),("Records","7,043"),
                ("Features","25"),("Model Status", status_text)]:
        color = status_color if k=="Model Status" else TEXT_PRI
        st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.8rem;margin:0.25rem 0'>"
                    f"<span style='color:{TEXT_PRI};font-weight:500'>{k}:</span> "
                    f"<span style='color:{color}'>{v}</span></p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p class="nav-header">Risk Thresholds</p>', unsafe_allow_html=True)
    for label,color,threshold in [("High Risk",DANGER,">= 70%"),
                                   ("Medium Risk",WARNING,"40 – 69%"),
                                   ("Low Risk",SUCCESS,"< 40%")]:
        st.markdown(f"<p style='color:{color};font-size:0.82rem;margin:0.2rem 0;font-weight:500'>"
                    f"{label}: <span style='font-weight:400'>{threshold}</span></p>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">ChurnGuard AI</p>
        <p class="hero-subtitle">An end-to-end machine learning system for predicting and preventing customer churn in the telecom industry.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [(c1,"Dataset Size","7,043"),(c2,"Features","25"),
                           (c3,"Churn Rate","26.5%"),(c4,"Best ROC-AUC","0.843")]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-accent">{val}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        steps_html = "".join([f"""
        <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px'>
            <div style='min-width:32px;height:32px;border-radius:8px;background:{PRIMARY}22;
                 border:1px solid {PRIMARY}55;display:flex;align-items:center;justify-content:center;
                 color:{PRIMARY};font-size:0.7rem;font-weight:700;flex-shrink:0'>{n}</div>
            <div><p style='color:{TEXT_PRI};font-size:0.85rem;font-weight:600;margin:0'>{t}</p>
                 <p style='color:{TEXT_MUT};font-size:0.78rem;margin:0'>{d}</p></div>
        </div>""" for n,t,d in [
            ("01","Data Collection","Telco Customer Churn dataset — 7,043 records"),
            ("02","EDA","Distributions, correlations, class imbalance"),
            ("03","Feature Engineering","6 new features, encoding, scaling"),
            ("04","Modeling","Logistic Regression, XGBoost, LightGBM"),
            ("05","Explainability","SHAP values for global and local explanations"),
            ("06","Deployment","FastAPI REST API + Streamlit Cloud"),
        ]])
        st.markdown(f'<div class="section-card"><p class="section-title">Project Pipeline</p>{steps_html}</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="section-card"><p class="section-title">Problem Statement</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8">Customer churn costs the telecom industry billions annually. '
                    f'Acquiring a new customer costs <strong style="color:{PRIMARY}">5–7x more</strong> than retaining one.</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8;margin-top:0.8rem">This system:</p>'
                    f'<ul style="color:{TEXT_PRI};font-size:0.9rem;line-height:2.2">'
                    f'<li>Identifies at-risk customers before they leave</li>'
                    f'<li>Explains why using SHAP feature importance</li>'
                    f'<li>Provides actionable retention recommendations</li>'
                    f'<li>Delivers real-time predictions via REST API</li>'
                    f'</ul></div>', unsafe_allow_html=True)
    techs = ["Python 3.10","pandas","NumPy","scikit-learn","XGBoost",
             "LightGBM","SHAP","MLflow","FastAPI","Streamlit","Docker","SMOTE","Plotly"]
    badges = " ".join([f"<span style='background:{PRIMARY}22;border:1px solid {PRIMARY}44;border-radius:20px;"
                       f"padding:4px 14px;color:{PRIMARY};font-size:0.8rem;margin:3px;display:inline-block'>{t}</span>"
                       for t in techs])
    st.markdown(f'<div class="section-card"><p class="section-title">Tech Stack</p>{badges}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard & Plots":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Dashboard & Plots</p>
        <p class="hero-subtitle">Key insights from exploratory data analysis on the Telco Churn dataset.</p>
    </div>""", unsafe_allow_html=True)

    r1c1,r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure(go.Pie(labels=["No Churn","Churn"],values=[73.5,26.5],hole=0.55,
            marker=dict(colors=[PRIMARY,DANGER],line=dict(color=BG_PAGE,width=2)),
            textfont=dict(color=TEXT_PRI,size=13)))
        fig.add_annotation(text="26.5%<br>Churn",x=0.5,y=0.5,showarrow=False,
                           font=dict(size=14,color=TEXT_PRI))
        fig.update_layout(**pcfg("Target Variable Distribution"))
        st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        fig = go.Figure(go.Bar(x=["Month-to-month","One year","Two year"],y=[42.7,11.3,2.8],
            marker_color=[DANGER,WARNING,SUCCESS],
            text=["42.7%","11.3%","2.8%"],textposition="outside",textfont=dict(color=TEXT_PRI)))
        cfg = pcfg("Churn Rate by Contract Type"); cfg["yaxis"]["title"]="Churn Rate (%)"
        fig.update_layout(**cfg); st.plotly_chart(fig, use_container_width=True)

    r2c1,r2c2 = st.columns(2)
    with r2c1:
        vals=[47.7,23.5,15.2,6.6]
        fig = go.Figure(go.Bar(x=["0–12 months","12–24 months","24–48 months","48–72 months"],y=vals,
            marker=dict(color=vals,colorscale=[[0,SUCCESS],[0.5,WARNING],[1,DANGER]]),
            text=[f"{v}%" for v in vals],textposition="outside",textfont=dict(color=TEXT_PRI)))
        cfg = pcfg("Churn Rate by Tenure Group"); cfg["yaxis"]["title"]="Churn Rate (%)"
        fig.update_layout(**cfg); st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        fig = go.Figure(go.Bar(x=["Fiber optic","DSL","No Internet"],y=[41.9,19.0,7.4],
            marker_color=[DANGER,WARNING,SUCCESS],
            text=["41.9%","19.0%","7.4%"],textposition="outside",textfont=dict(color=TEXT_PRI)))
        cfg = pcfg("Churn Rate by Internet Service"); cfg["yaxis"]["title"]="Churn Rate (%)"
        fig.update_layout(**cfg); st.plotly_chart(fig, use_container_width=True)

    r3c1,r3c2 = st.columns(2)
    with r3c1:
        vals=[45.3,19.1,16.7,15.2]
        fig = go.Figure(go.Bar(x=["Electronic check","Mailed check","Bank transfer","Credit card"],y=vals,
            marker=dict(color=vals,colorscale=[[0,SUCCESS],[0.5,WARNING],[1,DANGER]]),
            text=[f"{v}%" for v in vals],textposition="outside",textfont=dict(color=TEXT_PRI)))
        cfg = pcfg("Churn Rate by Payment Method"); cfg["yaxis"]["title"]="Churn Rate (%)"
        fig.update_layout(**cfg); st.plotly_chart(fig, use_container_width=True)
    with r3c2:
        fig = go.Figure(go.Bar(x=["No Security & No Support","Has One Service","Has Both"],y=[51.2,28.4,14.7],
            marker_color=[DANGER,WARNING,SUCCESS],
            text=["51.2%","28.4%","14.7%"],textposition="outside",textfont=dict(color=TEXT_PRI)))
        cfg = pcfg("Churn Rate by Support Services"); cfg["yaxis"]["title"]="Churn Rate (%)"
        fig.update_layout(**cfg); st.plotly_chart(fig, use_container_width=True)

    feats=["is_monthly","tenure","charges_per_tenure","MonthlyCharges",
           "no_support","risky_payment","num_addons","SeniorCitizen"]
    corrs=[0.405,-0.352,0.198,0.193,0.271,0.302,-0.164,0.151]
    fig = go.Figure(go.Bar(y=feats,x=corrs,orientation='h',
        marker_color=[DANGER if c>0 else PRIMARY for c in corrs],
        text=[f"{c:+.3f}" for c in corrs],textposition="outside",textfont=dict(color=TEXT_PRI)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=BG_CARD,
                      font=dict(color=TEXT_PRI),height=320,margin=dict(t=10,b=20,l=10,r=60),
                      title=dict(text="Feature Correlation with Churn",font=dict(color=TEXT_HEAD,size=14)),
                      xaxis=dict(gridcolor=BORDER,zerolinecolor=PRIMARY,
                                 title="Correlation with Churn",color=TEXT_MUT),
                      yaxis=dict(gridcolor=BORDER,color=TEXT_MUT))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Churn Prediction</p>
        <p class="hero-subtitle">Enter customer details to get a real-time churn risk assessment powered by XGBoost.</p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model not loaded: {model_error}. Make sure models/best_model_pipeline.pkl exists.")
    else:
        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="section-card"><p class="section-title">Demographics</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender",["Male","Female"])
            senior = st.selectbox("Senior Citizen",[0,1],format_func=lambda x:"Yes" if x==1 else "No")
            partner = st.selectbox("Partner",["Yes","No"])
            dependents = st.selectbox("Dependents",["Yes","No"])
            tenure = st.slider("Tenure (months)",0,72,12)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="section-card"><p class="section-title">Services</p>', unsafe_allow_html=True)
            phone_service = st.selectbox("Phone Service",["Yes","No"])
            multiple_lines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
            internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
            online_security = st.selectbox("Online Security",["Yes","No","No internet service"])
            online_backup = st.selectbox("Online Backup",["Yes","No","No internet service"])
            device_protection = st.selectbox("Device Protection",["Yes","No","No internet service"])
            tech_support = st.selectbox("Tech Support",["Yes","No","No internet service"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="section-card"><p class="section-title">Billing & Contract</p>', unsafe_allow_html=True)
            streaming_tv = st.selectbox("Streaming TV",["Yes","No","No internet service"])
            streaming_movies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])
            contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
            paperless = st.selectbox("Paperless Billing",["Yes","No"])
            payment = st.selectbox("Payment Method",["Electronic check","Mailed check",
                                   "Bank transfer (automatic)","Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges ($)",0.0,200.0,70.0,step=0.5)
            total = st.number_input("Total Charges ($)",0.0,10000.0,840.0,step=10.0)
            st.markdown('</div>', unsafe_allow_html=True)

        _,col_btn,_ = st.columns([1,2,1])
        with col_btn:
            predict_clicked = st.button("Analyze Churn Risk")

        if predict_clicked:
            input_data = {
                "gender":gender,"SeniorCitizen":senior,"Partner":partner,
                "Dependents":dependents,"tenure":tenure,"PhoneService":phone_service,
                "MultipleLines":multiple_lines,"InternetService":internet,
                "OnlineSecurity":online_security,"OnlineBackup":online_backup,
                "DeviceProtection":device_protection,"TechSupport":tech_support,
                "StreamingTV":streaming_tv,"StreamingMovies":streaming_movies,
                "Contract":contract,"PaperlessBilling":paperless,
                "PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total,
            }
            try:
                input_df = pd.DataFrame([input_data])
                input_df = engineer_features(input_df)
                prob = model.predict_proba(input_df)[0][1]
                risk = "High" if prob>=0.7 else "Medium" if prob>=0.4 else "Low"
                risk_color = DANGER if risk=="High" else WARNING if risk=="Medium" else SUCCESS
                verdict = {"High":"Likely to Churn","Medium":"At Risk","Low":"Likely to Stay"}[risk]
                risk_class = {"High":"result-high","Medium":"result-medium","Low":"result-low"}[risk]

                st.markdown("---")
                st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.1em'>Risk Assessment</p>",
                            unsafe_allow_html=True)
                res1,res2 = st.columns(2)
                with res1:
                    insights = {
                        "High": "High churn risk detected. Consider immediate retention action — offer a contract upgrade or personalized discount.",
                        "Medium": "Moderate risk detected. Monitor closely and consider proactive outreach with a loyalty offer.",
                        "Low": "Customer appears stable. Continue delivering value to maintain long-term satisfaction."
                    }
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <p style='color:{TEXT_MUT};font-size:0.78rem;margin:0;text-transform:uppercase;letter-spacing:0.08em'>Churn Risk Level</p>
                        <p class="result-prob" style="color:{risk_color}">{risk}</p>
                        <p style="color:{TEXT_PRI};font-size:1rem;font-weight:600;margin:0">{verdict}</p>
                        <p style="color:{TEXT_MUT};font-size:0.9rem;margin-top:0.5rem">
                            Probability: <strong style="color:{risk_color}">{prob:.1%}</strong>
                        </p>
                    </div>
                    <div class="insight-box">{insights[risk]}</div>
                    """, unsafe_allow_html=True)
                with res2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",value=prob*100,
                        number={"suffix":"%","font":{"size":36,"color":TEXT_PRI}},
                        gauge={"axis":{"range":[0,100],"tickcolor":TEXT_MUT,"tickfont":{"color":TEXT_MUT}},
                               "bar":{"color":risk_color,"thickness":0.25},
                               "bgcolor":BG_CARD,"bordercolor":BORDER,
                               "steps":[{"range":[0,40],"color":"#0d2016"},
                                        {"range":[40,70],"color":"#2a1f00"},
                                        {"range":[70,100],"color":"#2a1010"}],
                               "threshold":{"line":{"color":risk_color,"width":3},"thickness":0.8,"value":prob*100}}
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":TEXT_PRI},
                                      height=280,margin=dict(t=20,b=10,l=30,r=30))
                    st.plotly_chart(fig, use_container_width=True)

                # Risk factors
                st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem'>Key Risk Factors</p>",
                            unsafe_allow_html=True)
                factors=[]
                if contract=="Month-to-month": factors.append(("Contract Type","Month-to-month contracts have 3x higher churn than annual.",DANGER))
                if tenure<12: factors.append(("New Customer",f"Only {tenure} months — new customers churn significantly more.",DANGER))
                if internet=="Fiber optic" and monthly>70: factors.append(("High Charges",f"${monthly}/mo on Fiber optic is above average spend.",WARNING))
                if online_security=="No" and tech_support=="No": factors.append(("No Support Services","No security or tech support increases churn risk.",WARNING))
                if payment=="Electronic check": factors.append(("Payment Method","Electronic check correlates with higher churn rates.",WARNING))
                if tenure>36: factors.append(("Loyal Customer",f"{tenure} months — long-term loyalty reduces churn.",SUCCESS))
                if contract in ["One year","Two year"]: factors.append(("Long-term Contract","Contract commitment substantially reduces churn probability.",SUCCESS))

                f1,f2=st.columns(2)
                for i,(title,desc,color) in enumerate(factors):
                    card=(f"<div style='background:{BG_CARD};border:1px solid {color}44;"
                          f"border-left:3px solid {color};border-radius:8px;"
                          f"padding:0.8rem 1rem;margin-bottom:0.6rem'>"
                          f"<p style='color:{color};font-size:0.82rem;font-weight:600;margin:0'>{title}</p>"
                          f"<p style='color:{TEXT_PRI};font-size:0.8rem;margin:0.2rem 0 0'>{desc}</p></div>")
                    (f1 if i%2==0 else f2).markdown(card,unsafe_allow_html=True)

                # Summary metrics
                st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem'>Customer Summary</p>",
                            unsafe_allow_html=True)
                m1,m2,m3,m4=st.columns(4)
                svc=sum([phone_service=="Yes",online_security=="Yes",online_backup=="Yes",
                         device_protection=="Yes",tech_support=="Yes",
                         streaming_tv=="Yes",streaming_movies=="Yes"])
                for col,label,val in [(m1,"Tenure",f"{tenure} mo"),(m2,"Monthly Charges",f"${monthly:.0f}"),
                                      (m3,"Services Used",f"{svc} / 7"),(m4,"Est. Lifetime Value",f"${monthly*tenure:,.0f}")]:
                    col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                                 f'<p class="metric-value">{val}</p></div>',unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ENGINEERED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Engineered Features":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Engineered Features</p>
        <p class="hero-subtitle">6 new features created from domain knowledge and EDA insights to improve model performance.</p>
    </div>""", unsafe_allow_html=True)

    for name,formula,explanation,feat_type,importance in [
        ("charges_per_tenure","TotalCharges / (tenure + 1)",
         "Captures monthly spend efficiency. Customers with high charges relative to tenure signal price sensitivity — a key churn driver.","Numeric","High"),
        ("tenure_group","pd.cut(tenure, bins=[0, 12, 24, 48, 72])",
         "Bins tenure into 4 lifecycle stages: new, developing, established, loyal. New customers under 12 months have dramatically higher churn rates.","Categorical","High"),
        ("num_addons","Count of add-on services == 'Yes'",
         "Counts how many of 6 add-on services the customer subscribes to. More services create higher switching costs and reduce churn.","Numeric","Medium"),
        ("risky_payment","(PaperlessBilling == 'Yes') & (PaymentMethod == 'Electronic check')",
         "Binary flag for the highest-churn payment combination. Customers with this combination churn at over 45%.","Binary","High"),
        ("no_support","(OnlineSecurity == 'No') & (TechSupport == 'No')",
         "Flags customers with neither security nor tech support. These customers churn at 51% — the highest of any segment.","Binary","High"),
        ("is_monthly","(Contract == 'Month-to-month').astype(int)",
         "Binary flag for month-to-month contracts. These customers churn at 42.7% vs 2.8% for two-year contracts.","Binary","Very High"),
    ]:
        imp_color = DANGER if importance in ["High","Very High"] else WARNING if importance=="Medium" else SUCCESS
        st.markdown(f"""
        <div style='background:{BG_CARD};border:1px solid {BORDER};border-left:4px solid {PRIMARY};
             border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;
                 flex-wrap:wrap;gap:8px;margin-bottom:0.8rem'>
                <p style='color:{TEXT_PRI};font-size:1rem;font-weight:700;margin:0;font-family:monospace'>{name}</p>
                <div style='display:flex;gap:8px;flex-wrap:wrap'>
                    <span style='background:{PRIMARY}22;border:1px solid {PRIMARY}44;border-radius:20px;
                         padding:3px 10px;color:{PRIMARY};font-size:0.72rem'>{feat_type}</span>
                    <span style='background:{imp_color}22;border:1px solid {imp_color}44;border-radius:20px;
                         padding:3px 10px;color:{imp_color};font-size:0.72rem'>Importance: {importance}</span>
                </div>
            </div>
            <div style='background:#1a2035;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.6rem;
                 font-family:monospace;font-size:0.8rem;color:{SECONDARY}'>{formula}</div>
            <p style='color:{TEXT_PRI};font-size:0.85rem;margin:0;line-height:1.6'>{explanation}</p>
        </div>""", unsafe_allow_html=True)

    feat_names=["is_monthly","no_support","risky_payment","charges_per_tenure","num_addons","tenure_group"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="High Risk Group",x=feat_names,y=[42.7,51.2,45.3,38.1,12.4,47.7],marker_color=DANGER))
    fig.add_trace(go.Bar(name="Low Risk Group",x=feat_names,y=[2.8,14.7,18.2,18.9,38.6,6.6],marker_color=SUCCESS))
    layout=pcfg("Feature Impact on Churn Rate",h=320); layout["barmode"]="group"; layout["yaxis"]["title"]="Churn Rate (%)"
    fig.update_layout(**layout); st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Model Metrics</p>
        <p class="hero-subtitle">Performance comparison across all trained models with full evaluation metrics.</p>
    </div>""", unsafe_allow_html=True)

    m1,m2,m3,m4=st.columns(4)
    for col,label,val in [(m1,"Best ROC-AUC","0.843"),(m2,"Best F1","0.623"),
                          (m3,"Best Recall","0.783"),(m4,"Models Trained","4")]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-accent">{val}</p></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    df=pd.DataFrame({
        "Model":["Logistic Regression","XGBoost","LightGBM","XGBoost (Tuned)"],
        "ROC-AUC":[0.8426,0.8381,0.8370,0.8429],
        "F1 Score":[0.6227,0.5923,0.5884,0.6129],
        "Precision":[0.5168,0.5884,0.5807,0.5640],
        "Recall":[0.7834,0.5963,0.5963,0.6711],
    })
    model_colors=[PRIMARY,SECONDARY,SUCCESS,WARNING]

    col1,col2=st.columns(2)
    with col1:
        categories=["ROC-AUC","F1 Score","Precision","Recall","ROC-AUC"]
        fig=go.Figure()
        for i,(_,row) in enumerate(df.iterrows()):
            vals=[row["ROC-AUC"],row["F1 Score"],row["Precision"],row["Recall"],row["ROC-AUC"]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=categories,fill='toself',
                name=row["Model"],line=dict(color=model_colors[i]),opacity=0.65))
        fig.update_layout(
            polar=dict(bgcolor=BG_CARD,
                radialaxis=dict(visible=True,range=[0.4,1],color=TEXT_MUT,gridcolor=BORDER),
                angularaxis=dict(color=TEXT_MUT,gridcolor=BORDER)),
            paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_PRI,family="Inter"),
            height=360,legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_PRI)),
            margin=dict(t=30,b=30),
            title=dict(text="Model Comparison — Radar",font=dict(color=TEXT_HEAD,size=14)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_cols=["ROC-AUC","F1 Score","Precision","Recall"]
        fig=go.Figure()
        for i,(_,row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(name=row["Model"],x=metric_cols,
                y=[row[m] for m in metric_cols],marker_color=model_colors[i]))
        layout=pcfg("Model Metrics Comparison",h=360); layout["barmode"]="group"; layout["yaxis"]["range"]=[0,1]
        fig.update_layout(**layout); st.plotly_chart(fig, use_container_width=True)

    shap_feats=["is_monthly","tenure","charges_per_tenure","MonthlyCharges",
                "no_support","risky_payment","Contract_Two year",
                "num_addons","InternetService_Fiber optic","SeniorCitizen"]
    shap_vals=[0.412,0.387,0.241,0.198,0.187,0.165,0.143,0.121,0.108,0.089]
    fig=go.Figure(go.Bar(y=shap_feats,x=shap_vals,orientation='h',
        marker=dict(color=shap_vals,colorscale=[[0,PRIMARY],[0.5,SECONDARY],[1,DANGER]]),
        text=[f"{v:.3f}" for v in shap_vals],textposition="outside",textfont=dict(color=TEXT_PRI)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=BG_CARD,
                      font=dict(color=TEXT_PRI,family="Inter"),height=340,
                      margin=dict(t=30,b=10,l=10,r=60),
                      title=dict(text="SHAP Feature Importance — Top 10",font=dict(color=TEXT_HEAD,size=14)),
                      xaxis=dict(gridcolor=BORDER,title="Mean |SHAP Value|",color=TEXT_MUT),
                      yaxis=dict(gridcolor=BORDER,color=TEXT_MUT))
    st.plotly_chart(fig, use_container_width=True)

    fig=go.Figure(go.Heatmap(z=[[874,178],[148,209]],x=["No Churn","Churn"],y=["No Churn","Churn"],
        colorscale=[[0,BG_CARD],[1,PRIMARY]],
        text=[["874","178"],["148","209"]],texttemplate="%{text}",textfont={"size":22,"color":TEXT_HEAD}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=BG_CARD,
                      font=dict(color=TEXT_PRI,family="Inter"),height=300,
                      title=dict(text="Confusion Matrix — XGBoost (Tuned)",font=dict(color=TEXT_HEAD,size=14)),
                      xaxis=dict(title="Predicted",color=TEXT_MUT),
                      yaxis=dict(title="Actual",color=TEXT_MUT),
                      margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)
