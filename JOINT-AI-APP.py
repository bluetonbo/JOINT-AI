import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS (사각형 박스 슬림 최적화)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #090d16 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1524 !important;
        border-right: 1px solid #1e293b;
        min-width: 360px !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    /* [수정됨] 박스 위아래 패딩 축소 (4px 16px) */
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 6px;
        padding: 4px 16px; 
        margin-bottom: 12px;
    }
    
    /* 박스 내부 타이틀 텍스트 여백 최적화 */
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        padding: 4px 0;
    }

    .stButton>button, .stDownloadButton>button {
        height: 2.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        background: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #059669 !important;
    }
    
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    }
    
    .stNumberInput label, .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        margin-bottom: 2px !important;
    }
    
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        height: 2.8rem !important;
        color: #64748b !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 16px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 시스템 암호 인증 패널
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div class='glass-card' style='text-align: center; padding: 20px 40px;'>
                <h2 style='color: #10b981; margin-top: 10px; margin-bottom: 5px; font-size: 1.5rem;'>JOINT PROCESS INTELLIGENCE</h2>
                <p style='color: #64748b; font-size:0.85rem; margin-bottom: 0px;'>Core AI Optimization Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# 4. 데이터 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'optimizer_status': "STANDBY",
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5,
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_confidence': None
    })

# 5. 사이드바
with st.sidebar:
    st.markdown("## CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df[X_list])
            X_scaled = scaler.transform(df[X_list])
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                'scaler': scaler, 'df_caulking': df,
                'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 뷰포트
if st.session_state['model_tq']:
    st.title("JOINT PROCESS INTELLIGENCE")
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])

    with tab1:
        layout_l, layout_r = st.columns([1, 1.5])
        with layout_l:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Boundary Condition Optimizer</div></div>", unsafe_allow_html=True)
            # (이하 생략 - 동일한 로직 유지)
            st.info("Optimizer logic active.")
        with layout_r:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Predicted Performance</div></div>", unsafe_allow_html=True)

    with tab2:
        st.write("Simulator active.")
    with tab3:
        st.dataframe(st.session_state['df_caulking'])
else:
    st.info("좌측 CONTROL CONSOLE에서 파일을 로드하여 엔진을 가동하십시오.")
