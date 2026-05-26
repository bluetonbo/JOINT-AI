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

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS
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
    
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 8px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 15px;
    }

    .stButton>button, .stDownloadButton>button {
        height: 3rem !important;
        font-size: 0.95rem !important;
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
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
    }

    /* 숫자 입력창 라벨 스타일 정의 */
    .stNumberInput label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin-bottom: 4px !important;
    }
    
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        height: 3.2rem !important;
        color: #64748b !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    
    .stAlert {
        background-color: #141f36 !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
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
            <div class='glass-card' style='text-align: center; padding: 40px; margin-bottom: 25px;'>
                <h2 style='color: #10b981; margin-top: 0px; margin-bottom: 5px; font-size: 1.8rem;'>JOINT PROCESS INTELLIGENCE</h2>
                <p style='color: #64748b; font-size:0.9rem; margin-bottom: 0px;'>Core Optimization Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        pwd = st.text_input("Enter Password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. System access denied.")
    st.stop()

# 4. 세션 데이터 구조 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'data_bounds': {  
            'Caulking_Distance': (4.0, 7.0),
            'Stud_Center': (1.5, 3.5),
            'Aging_Status': (0, 1)
        },
        'optimizer_status': "STANDBY",
        'target_tq_range': (35.0, 37.0),
        'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_executed_vars': None, 'sim_confidence': None
    })

# 5. 사이드바 - 제어반
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    
    with st.expander("Master Data Stream", expanded=True):
        u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            def load_data(f):
                return pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
            
            df_master = load_data(u_input)
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            
            scaler = MinMaxScaler().fit(df_comb[X_list])
            X_scaled = scaler.transform(df_comb[X_list])
            
            model_tq = LinearRegression().fit(X_scaled, df_comb['Torque'])
            model_ed = LinearRegression().fit(X_scaled, df_comb['Endurance'])
            
            st.session_state.update({
                'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb,
                'optimizer_status': "ENGINE READY",
                'data_bounds': {
                    'Caulking_Distance': (float(df_comb['Caulking_Distance'].min()), float(df_comb['Caulking_Distance'].max())),
                    'Stud_Center': (float(df_comb['Stud_Center'].min()), float(df_comb['Stud_Center'].max())),
                    'Aging_Status': (0, 1)
                }
            })
            st.rerun()
        else:
            st.error("Please upload a data log file.")

# 6. 메인 뷰포트 영역
if st.session_state['model_tq']:
    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown("<h1 style='margin-bottom:0px; font-size:1.8rem;'>JOINT PROCESS INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; margin-bottom:25px; font-size:0.9rem;'>Inverse Optimization & Process Simulation Terminal</p>", unsafe_allow_html=True)
    with h_right:
        st.markdown(f"""
            <div style='display:flex; gap:10px; justify-content:flex-end; align-items:center; height:100%; padding-bottom:20px;'>
                <span style='background:#1e293b; color:#38bdf8; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #334155;'>CORE: ACTIVE</span>
                <span style='background:#1e293b; color:#10b981; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #334155;'>LOGS: {len(st.session_state['df_caulking'])} Rows</span>
                <span style='background:#022c22; color:#34d399; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #065f46;'>{st.session_state['optimizer_status']}</span>
            </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])

    # ------------------ TAB 1: 품질 타겟 추적 ------------------
    with tab1:
        layout_l, layout_r = st.columns([1.1, 1.4], gap="large")
        
        with layout_l:
            st.markdown("""
                <div class='glass-card'>
                    <div class='glass-card-title'>Boundary Condition Optimizer</div>
            """, unsafe_allow_html=True)
            
            bound_mode = st.radio(
                "Safety Bound Limit Mode",
                options=["Auto Mode", "Manual Expert Tuning"],
                index=0, horizontal=True
            )
            
            db = st.session_state['data_bounds']
            st.markdown("<br>", unsafe_allow_html=True)
            
            if "Auto Mode" in bound_mode:
                st.markdown(f"""
                    <div style='background:#0f172a; padding:15px; border-radius:6px; border:1px solid #1e293b; font-size:0.85rem;'>
                        <span style='color:#38bdf8; font-weight:600;'>[Auto-Bound Enabled]</span><br>
                        • Caulking Distance: {db['Caulking_Distance'][0]:.2f} ~ {db['Caulking_Distance'][1]:.2f} mm<br>
                        • Stud Center: {db['Stud_Center'][0]:.2f} ~ {db['Stud_Center'][1]:.2f} mm
                    </div>
                """, unsafe_allow_html=True)
                chosen_bounds = {
                    'Caulking_Distance': db['Caulking_Distance'],
                    'Stud_Center': db['Stud_Center']
                }
            else:
                # [수정] 경계 설정 레이어 수치형 직접 타이핑 키인(st.number_input) 전환
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    m_cd_min = st.number_input("CD Min Boundary (mm)", value=float(round(db['Caulking_Distance'][0], 2)), step=0.05, format="%.2f")
                    m_sc_min = st.number_input("SC Min Boundary (mm)", value=float(round(db['Stud_Center'][0], 2)), step=0.05, format="%.2f")
                with b_col2:
                    m_cd_max = st.number_input("CD Max Boundary (mm)", value=float(round(db['Caulking_Distance'][1], 2)), step=0.05, format="%.2f")
                    m_sc_max = st.number_input("SC Max Boundary (mm)", value=float(round(db['Stud_Center'][1], 2)), step=0.05, format="%.2f")
                
                chosen_bounds = {
                    'Caulking_Distance': (m_cd_min, m_cd_max),
                    'Stud_Center': (m_sc_min, m_sc_max)
                }
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
                <div class='glass-card'>
                    <div class='glass-card-title'>Target Quality KPIs Range</div>
            """, unsafe_allow_html=True)
            
            # [수정] Target 품질 목표치 설정 수치형 직접 타이핑 키인(st.number_input) 전환
            k_col1, k_col2 = st.columns(2)
            with k_col1:
                t_tq_min = st.number_input("Target Torque Min (Nm)", value=st.session_state['target_tq_range'][0], step=0.1, format="%.1f")
                t_ed_min = st.number_input("Target Endurance Min (Cyc)", value=int(st.session_state['target_ed_range'][0]), step=1000)
            with k_col2:
                t_tq_max = st.number_input("Target Torque Max (Nm)", value=st.session_state['target_tq_range'][1], step=0.1, format="%.1f")
                t_ed_max = st.number_input("Target Endurance Max (Cyc)", value=int(st.session_state['target_ed_range'][1]), step=1000)
            
            st.session_state['target_tq_range'] = (t_tq_min, t_tq_max)
            st.session_state['target_ed_range'] = (float(t_ed_min), float(t_ed_max))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("RUN INVERSE INFERENCE SEARCH", type="secondary", use_container_width=True):
                X_vars = st.session_state['process_vars']
                
                def target_loss_function(x_input):
                    df_query = pd.DataFrame([x_input], columns=X_vars)
                    scaled_query = st.session_state['scaler'].transform(df_query)
                    pred_tq = st.session_state['model_tq'].predict(scaled_query)[0]
                    pred_ed = st.session_state['model_ed'].predict(scaled_query)[0]
                    
                    tq_min, tq_max = st.session_state['target_tq_range']
                    ed_min, ed_max = st.session_state['target_ed_range']
                    
                    tq_loss = 0.0
                    if pred_tq < tq_min: tq_loss = (tq_min - pred_tq) ** 2
                    elif pred_tq > tq_max: tq_loss = (pred_tq - tq_max) ** 2
                    
                    ed_loss = 0.0
                    if pred_ed < ed_min: ed_loss = ((ed_min - pred_ed) / 1000.0) ** 2
                    elif pred_ed > ed_max: ed_loss = ((pred_ed - ed_max) / 1000.0) ** 2
                    
                    return tq_loss + ed_loss

                best_score = float('inf')
                best_res = None
                bounds_list = [chosen_bounds['Caulking_Distance'], chosen_bounds['Stud_Center'], (0, 1)]
                
                for ag_option in [0, 1]:
                    init_guess = [(bounds_list[0][0] + bounds_list[0][1]) / 2.0, (bounds_list[1][0] + bounds_list[1][1]) / 2.0, ag_option]
                    current_bounds = [(bounds_list[0][0], bounds_list[0][1]), (bounds_list[1][0], bounds_list[1][1]), (ag_option, ag_option)]
                    res = minimize(target_loss_function, init_guess, method='SLSQP', bounds=current_bounds)
                    if res.fun < best_score:
                        best_score = res.fun
                        best_res = res
                
                if best_res:
                    opt_x = best_res.x
                    df_opt_x = pd.DataFrame([opt_x], columns=X_vars)
                    scaled_opt = st.session_state['scaler'].transform(df_opt_x)
                    
                    pred_tq = st.session_state['model_tq'].predict(scaled_opt)[0]
                    pred_ed = st.session_state['model_ed'].predict(scaled_opt)[0]
                    
                    st.session_state['opt_result_x'] = opt_x
                    st.session_state['opt_pred_tq'] = pred_tq
                    st.session_state['opt_pred_ed'] = pred_ed
                    
                    tq_min, tq_max = st.session_state['target_tq_range']
                    ed_min, ed_max = st.session_state['target_ed_range']
                    
                    tq_mid = (tq_min + tq_max) / 2.0
                    tq_half_range = (tq_max - tq_min) / 2.0 if (tq_max - tq_min) > 0 else 1.0
                    tq_dev = abs(pred_tq - tq_mid) / tq_half_range
                    tq_conf = max(0.0, 100.0 - (tq_dev * 50.0)) if tq_min <= pred_tq <= tq_max else max(0.0, 50.0 - (min(abs(pred_tq-tq_min), abs(pred_tq-tq_max)) * 50.0))
                    
                    ed_mid = (ed_min + ed_max) / 2.0
                    ed_half_range = (ed_max - ed_min) / 2.0 if (ed_max - ed_min) > 0 else 1.0
                    ed_dev = abs(pred_ed - ed_mid) / ed_half_range
                    ed_conf = max(0.0, 100.0 - (ed_dev * 50.0)) if ed_min <= pred_ed <= ed_max else max(0.0, 50.0 - (min(abs(pred_ed-ed_min), abs(pred_ed-ed_max)) / 1000.0 * 50.0))
                    
                    st.session_state['confidence_score'] = round((tq_conf + ed_conf) / 2.0, 1)
                    st.session_state['optimizer_status'] = "SUCCESS" if st.session_state['confidence_score'] >= 80.0 else "APPROXIMATED"
                st.rerun()

        with layout_r:
            if st.session_state['opt_result_x'] is not None:
                opt_x = st.session_state['opt_result_x']
                aging_text = "Aged" if round(opt_x[2]) == 1 else "Unaged"
                
                st.markdown("""
                    <div class='glass-card'>
                        <div class='glass-card-title' style='color:#3b82f6;'>Predicted Performance & Confidence</div>
                """, unsafe_allow_html=True)
                
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #3b82f6; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Predicted Torque</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{st.session_state['opt_pred_tq']:.2f}<span style='font-size:0.85rem; color:#64748b;'> Nm</span></h3></div>", unsafe_allow_html=True)
                with r_col2:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #3b82f6; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Predicted Endurance</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{st.session_state['opt_pred_ed']:,.0f}<span style='font-size:0.85rem; color:#64748b;'> Cyc</span></h3></div>", unsafe_allow_html=True)
                with r_col3:
                    conf_color = "#10b981" if st.session_state['confidence_score'] >= 80.0 else "#ef4444"
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid {conf_color}; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Target Confidence</span><h3 style='color:{conf_color}; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{st.session_state['confidence_score']:.1f}<span style='font-size:0.85rem; color:#64748b;'> %</span></h3></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""
                    <div class='glass-card'>
                        <div class='glass-card-title' style='color:#10b981;'>Recommended Process Specifications</div>
                """, unsafe_allow_html=True)
                
                c_cols = st.columns(3)
                with c_cols[0]:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #10b981; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Caulking Distance</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{opt_x[0]:.2f}<span style='font-size:0.85rem; color:#64748b;'> mm</span></h3></div>", unsafe_allow_html=True)
                with c_cols[1]:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #10b981; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Stud Center</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{opt_x[1]:.2f}<span style='font-size:0.85rem; color:#64748b;'> mm</span></h3></div>", unsafe_allow_html=True)
                with c_cols[2]:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #10b981; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Aging Status</span><h3 style='color:#ffffff; font-size:1.3rem; margin:6px 0; font-family:Inter; font-weight:600;'>{aging_text}</h3></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                df_excel_data = pd.DataFrame({
                    "KPI Parameters": ["Recommended Caulking Distance (mm)", "Recommended Stud Center (mm)", "Recommended Aging Status", 
                                     "Expected Torque Value (Nm)", "Expected Endurance Life (Cycles)", "Optimization Confidence Score (%)"],
                    "AI Optimized Specification": [f"{opt_x[0]:.2f}", f"{opt_x[1]:.2f}", aging_text, f"{st.session_state['opt_pred_tq']:.2f}", f"{st.session_state['opt_pred_ed']:,.0f}", f"{st.session_state['confidence_score']:.1f}"]
                })
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_excel_data.to_excel(writer, index=False, sheet_name='Optimization_Report')
                
                st.download_button(
                    label="DOWNLOAD OPTIMIZATION REPORT (.XLSX)",
                    data=output.getvalue(),
                    file_name="Process_Optimization_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.markdown("<div style='text-align:center; padding:100px 0; color:#475569; font-size:0.9rem;'>수치 지정 후 하단의 검색 버튼을 클릭하면 최적화 결과가 정렬됩니다.</div>", unsafe_allow_html=True)

    # ------------------ TAB 2: 실시간 시뮬레이터 ------------------
    with tab2:
        sim_l, sim_r = st.columns([1.1, 1.4], gap="large")
        
        with sim_l:
            st.markdown("""
                <div class='glass-card'>
                    <div class='glass-card-title'>Real-time Parameter Input Panel</div>
            """, unsafe_allow_html=True)
            
            sim_cb = st.session_state['data_bounds']
            
            # [수정] What-if 시뮬레이터 변수 입력 방식도 드래그에서 키보드 숫자 키인(st.number_input)으로 변경
            sim_cd = st.number_input(
                "Live Field Caulking Distance (mm)",
                min_value=0.0, max_value=15.0,
                value=float(round((sim_cb['Caulking_Distance'][0] + sim_cb['Caulking_Distance'][1])/2, 2)), 
                step=0.01, format="%.2f"
            )
            sim_sc = st.number_input(
                "Live Field Stud Center (mm)",
                min_value=0.0, max_value=10.0,
                value=float(round((sim_cb['Stud_Center'][0] + sim_cb['Stud_Center'][1])/2, 2)), 
                step=0.01, format="%.2f"
            )
            sim_ag_label = st.radio(
                "Live Field Aging Processing Status",
                options=["Unaged (Status: 0)", "Aged (Status: 1)"], index=0
            )
            sim_ag = 1 if "Aged" in sim_ag_label else 0
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("EXECUTE PREDICTIVE SIMULATION", type="secondary", use_container_width=True):
                X_vars = st.session_state['process_vars']
                df_sim_query = pd.DataFrame([[sim_cd, sim_sc, sim_ag]], columns=X_vars)
                scaled_sim_query = st.session_state['scaler'].transform(df_sim_query)
                
                pred_tq = st.session_state['model_tq'].predict(scaled_sim_query)[0]
                pred_ed = st.session_state['model_ed'].predict(scaled_sim_query)[0]
                
                cd_min, cd_max = sim_cb['Caulking_Distance']
                sc_min, sc_max = sim_cb['Stud_Center']
                
                def calculate_bound_score(val, v_min, v_max):
                    if v_min <= val <= v_max: return 100.0
                    v_range = (v_max - v_min) if (v_max - v_min) > 0 else 1.0
                    return max(0.0, 100.0 - (min(abs(val - v_min), abs(val - v_max)) / v_range * 200.0))
                    
                cd_score = calculate_bound_score(sim_cd, cd_min, cd_max)
                sc_score = calculate_bound_score(sim_sc, sc_min, sc_max)
                
                st.session_state['sim_pred_tq'] = pred_tq
                st.session_state['sim_pred_ed'] = pred_ed
                st.session_state['sim_confidence'] = round((cd_score + sc_score) / 2.0, 1)
                st.session_state['sim_executed_vars'] = [sim_cd, sim_sc, sim_ag]
                st.rerun()

        with sim_r:
            if st.session_state['sim_pred_tq'] is not None:
                st.markdown("""
                    <div class='glass-card'>
                        <div class='glass-card-title' style='color:#38bdf8;'>AI Forward Simulation Outputs</div>
                """, unsafe_allow_html=True)
                
                s_res1, s_res2, s_res3 = st.columns(3)
                with s_res1:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #38bdf8; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>AI Est. Torque</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{st.session_state['sim_pred_tq']:.2f}<span style='font-size:0.85rem; color:#64748b;'> Nm</span></h3></div>", unsafe_allow_html=True)
                with s_res2:
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid #38bdf8; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>AI Est. Endurance</span><h3 style='color:#ffffff; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{st.session_state['sim_pred_ed']:,.0f}<span style='font-size:0.85rem; color:#64748b;'> Cyc</span></h3></div>", unsafe_allow_html=True)
                with s_res3:
                    s_conf = st.session_state['sim_confidence']
                    s_conf_color = "#10b981" if s_conf >= 80.0 else ("#f59e0b" if s_conf >= 50.0 else "#ef4444")
                    st.markdown(f"<div style='border-radius:4px; border-left:3px solid {s_conf_color}; padding:12px; background:#0f172a;'><span style='color:#64748b; font-size:0.8rem; font-weight:600;'>Safe Range Index</span><h3 style='color:{s_conf_color}; font-size:1.6rem; margin:2px 0; font-family:JetBrains Mono;'>{s_conf:.1f}<span style='font-size:0.85rem; color:#64748b;'> %</span></h3></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                ev = st.session_state['sim_executed_vars']
                df_sim_excel = pd.DataFrame({
                    "Simulation Log Parameters": ["Input Caulking Distance (mm)", "Input Stud Center (mm)", "Input Aging Configuration", "AI Synthesized Torque (Nm)", "AI Synthesized Endurance (Cycles)", "Safe Range Proximity Index (%)"],
                    "Value Config Log": [f"{ev[0]:.2f}", f"{ev[1]:.2f}", "Aged" if ev[2] == 1 else "Unaged", f"{st.session_state['sim_pred_tq']:.2f}", f"{st.session_state['sim_pred_ed']:,.0f}", f"{st.session_state['sim_confidence']:.1f}"]
                })
                sim_output = io.BytesIO()
                with pd.ExcelWriter(sim_output, engine='openpyxl') as writer:
                    df_sim_excel.to_excel(writer, index=False, sheet_name='Simulation_Report')
                
                st.download_button(
                    label="DOWNLOAD SIMULATION REPORT (.XLSX)",
                    data=sim_output.getvalue(),
                    file_name="Simulation_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.markdown("<div style='text-align:center; padding:100px 0; color:#475569; font-size:0.9rem;'>변수를 입력한 후 시뮬레이션 실행 버튼을 클릭하면 결과 카드가 정렬됩니다.</div>", unsafe_allow_html=True)

    # ------------------ TAB 3: 공정 로그 데이터레이크 ------------------
    with tab3:
        st.markdown("""
            <div class='glass-card'>
                <div class='glass-card-title'>Central Data Repository Log</div>
        """, unsafe_allow_html=True)
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("CORE ENGINE INACTIVE: 좌측 CONTROL CONSOLE에서 로그 파일을 로드한 후 가동해 주십시오.")
