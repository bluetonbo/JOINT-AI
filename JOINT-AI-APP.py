import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정: 태블릿 및 데스크탑 확장 레이아웃
st.set_page_config(
    layout="wide", 
    page_title="Optimization Conditions of Joint",
    page_icon="---"
)

# 2. 프로페셔널 하이엔드 엔지니어링 UI/UX 스타일 (Deep Navy & Emerald Green)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700;900&display=swap');
    
    /* 전체 배경 및 폰트 - 신뢰감을 주는 딥 네이비 */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 사이드바 스타일 및 경계선 */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
        min-width: 340px !important;
    }

    /* 프로페셔널 사각 플랫 버튼 스타일 (에메랄드 그린 테마) */
    .stButton>button, .stDownloadButton>button {
        height: 3.5rem !important;
        font-size: 1.05rem !important;
        border-radius: 6px !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        font-weight: 700;
        border: 1px solid #10b981 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #10b981 !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }

    /* 슬라이더 제어 영역 최적화 */
    .stSlider label {
        color: #10b981 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px !important;
    }
    
    div[data-baseweb="slider"] {
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }

    /* 데이터 대시보드 메트릭 카드 스타일 */
    [data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 18px !important;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }

    /* 파일 업로더 라벨 및 버튼 */
    div[data-testid="stFileUploader"] label {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #374151 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563;
        height: 3rem !important;
    }

    /* 탭(Tab) 메뉴 디자인 */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        height: 3.2rem !important;
        color: #9ca3af !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #10b981 !important;
        border-bottom-color: #10b981 !important;
    }

    .stDataFrame {
        border: 1px solid #374151;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 보안 인증 (Admin1234)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([0.5, 2, 0.5])
    with center:
        st.markdown("<br><br><h2 style='text-align: center; color: #10b981;'>CAULKING AI SYSTEM ACCESS</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Enter System Password", type="password")
        if st.button("Connect System"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
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
        'optimizer_status': "Wait",
        'target_tq_range': (35.0, 37.0),
        'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_executed_vars': None, 'sim_confidence': None
    })

# 5. 사이드바 - JOINT-INPUT 데이터 패널 관리
with st.sidebar:
    st.markdown("<h3 style='color: #10b981; font-family: JetBrains Mono;'>DATA MANAGEMENT</h3>", unsafe_allow_html=True)
    
    with st.expander("공정 데이터 로드", expanded=True):
        u_input = st.file_uploader("JOINT-INPUT 데이터 업로드 (CSV, XLSX)", type=['csv','xlsx'])
    
    if st.button("AI 엔진 및 공정 최적화 실행", type="primary"):
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
                'optimizer_status': "Engine Ready",
                'data_bounds': {
                    'Caulking_Distance': (float(df_comb['Caulking_Distance'].min()), float(df_comb['Caulking_Distance'].max())),
                    'Stud_Center': (float(df_comb['Stud_Center'].min()), float(df_comb['Stud_Center'].max())),
                    'Aging_Status': (0, 1)
                }
            })
            st.rerun()
        else:
            st.error("JOINT-INPUT 파일을 업로드해 주세요.")

# 6. 메인 통합 관제 대시보드
if st.session_state['model_tq']:
    st.markdown("<h1>Optimization Conditions <span style='color:#10b981; font-family:JetBrains Mono;'>of Joint</span></h1>", unsafe_allow_html=True)
    
    # 상단 모니터링 메트릭 영역
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("AI LEARNING ENGINE", "ACTIVE")
    with m2: st.metric("HISTORICAL DATA LOGS", f"{len(st.session_state['df_caulking'])} Rows")
    with m3: st.metric("FINDER SYSTEM STATUS", st.session_state['optimizer_status'])

    tab1, tab2, tab3 = st.tabs(["품질 타겟 추적 솔루션", "현장 변수 실시간 시뮬레이터", "코킹 공정 원천 로그"])

    # ------------------ TAB 1: 품질 타겟 추적 솔루션 ------------------
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>공정 탐색 알고리즘 경계 조건 설정</h4>", unsafe_allow_html=True)
            
            bound_mode = st.radio(
                "안전 한계선 설정 모드 선택",
                options=["자동 모드 (업로드 데이터 기준)", "수동 모드 (엔지니어 직접 지정)"],
                index=0,
                horizontal=True
            )
            
            db = st.session_state['data_bounds']
            
            if "자동 모드" in bound_mode:
                st.info("선택 모드: [자동] 업로드된 마스터 데이터의 실제 공정 범위를 안전 한계선으로 고정 적용합니다.")
                chosen_bounds = {
                    'Caulking_Distance': db['Caulking_Distance'],
                    'Stud_Center': db['Stud_Center']
                }
                st.markdown(f"""
                * **지정된 코킹 거리(CD) 영역:** {chosen_bounds['Caulking_Distance'][0]:.2f} mm ~ {chosen_bounds['Caulking_Distance'][1]:.2f} mm
                * **지정된 스터드 센터(SC) 영역:** {chosen_bounds['Stud_Center'][0]:.2f} mm ~ {chosen_bounds['Stud_Center'][1]:.2f} mm
                """)
            else:
                st.warning("선택 모드: [수동] 설비 보호 및 공정 목적에 맞게 안전 한계치 한계 영역을 직접 튜닝합니다.")
                
                manual_cd = st.slider(
                    "수동 코킹 거리 한계 범위 지정 (CD, mm)",
                    min_value=0.0, max_value=15.0,
                    value=(float(round(db['Caulking_Distance'][0], 2)), float(round(db['Caulking_Distance'][1], 2))),
                    step=0.05
                )
                manual_sc = st.slider(
                    "수동 스터드 센터 한계 범위 지정 (SC, mm)",
                    min_value=0.0, max_value=10.0,
                    value=(float(round(db['Stud_Center'][0], 2)), float(round(db['Stud_Center'][1], 2))),
                    step=0.05
                )
                chosen_bounds = {
                    'Caulking_Distance': manual_cd,
                    'Stud_Center': manual_sc
                }

        with col2:
            st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>요구 품질 목표 범위 지정 (Target Quality)</h4>", unsafe_allow_html=True)
            
            st.session_state['target_tq_range'] = st.slider(
                "목표 토크 범위 지정 (Torque, Nm)",
                min_value=20.0, max_value=50.0,
                value=st.session_state['target_tq_range'], step=0.1
            )
            
            st.session_state['target_ed_range'] = st.slider(
                "목표 내구 수명 범위 지정 (Endurance, Cycles)",
                min_value=50000, max_value=200000,
                value=(int(st.session_state['target_ed_range'][0]), int(st.session_state['target_ed_range'][1])), step=1000
            )

        st.markdown("<br><hr style='border: 0.5px solid #1f2937;'><br>", unsafe_allow_html=True)
        
        if st.button("품질 만족 공정 변수(CD, SC, AG) 역산 추적 실행", type="primary", use_container_width=True):
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
            
            bounds_list = [
                chosen_bounds['Caulking_Distance'],
                chosen_bounds['Stud_Center'],
                (0, 1)
            ]
            
            for ag_option in [0, 1]:
                init_guess = [
                    (bounds_list[0][0] + bounds_list[0][1]) / 2.0,
                    (bounds_list[1][0] + bounds_list[1][1]) / 2.0,
                    ag_option
                ]
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
                st.session_state['optimizer_status'] = "Optimization Success" if st.session_state['confidence_score'] >= 80.0 else "Approximated"
            st.rerun()

        if st.session_state['opt_result_x'] is not None:
            opt_x = st.session_state['opt_result_x']
            aging_text = "Aged (에이징 적용)" if round(opt_x[2]) == 1 else "Unaged (미적용)"
            
            st.markdown("<h3 style='color:#ffffff; margin-top: 25px;'>AI 분석 기반 권장 공정 사양</h3>", unsafe_allow_html=True)
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 코킹 거리 (Caulking_Distance)</span><h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{opt_x[0]:.2f} <span style='font-size:1.2rem; color:#10b981;'>mm</span></h2></div>", unsafe_allow_html=True)
            with c_col2:
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 스터드 센터 (Stud_Center)</span><h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{opt_x[1]:.2f} <span style='font-size:1.2rem; color:#10b981;'>mm</span></h2></div>", unsafe_allow_html=True)
            with c_col3:
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 에이징 여부 (Aging_Status)</span><h2 style='color: #ffffff; font-size: 1.8rem; margin: 12px 0; font-family: Inter; font-weight:700;'>{aging_text}</h2></div>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='color:#ffffff; margin-top: 25px;'>해당 조건 세팅 시 예상 품질 인자 및 신뢰도</h3>", unsafe_allow_html=True)
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>해당 조건 세팅 시 예상 토크 값</span><h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['opt_pred_tq']:.2f} <span style='font-size:1.2rem; color:#3b82f6;'>Nm</span></h2></div>", unsafe_allow_html=True)
            with r_col2:
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>해당 조건 세팅 시 예상 내구 수명</span><h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['opt_pred_ed']:,.0f} <span style='font-size:1.2rem; color:#3b82f6;'>Cycles</span></h2></div>", unsafe_allow_html=True)
            with r_col3:
                conf_color = "#10b981" if st.session_state['confidence_score'] >= 80.0 else "#ef4444"
                st.markdown(f"<div style='border-radius: 6px; border-left: 6px solid {conf_color}; padding: 20px; background: #1f2937;'><span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>품질 조건 만족 범위 신뢰도</span><h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['confidence_score']:.1f} <span style='font-size:1.2rem; color:{conf_color};'>%</span></h2></div>", unsafe_allow_html=True)

            # 다운로드 버튼 생략... (이전 코드와 동일)

    # ------------------ TAB 2: 현장 변수 실시간 시뮬레이터 ------------------
    with tab2:
        st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>현장 실시간 공정 조건 입력 패널</h4>", unsafe_allow_html=True)
        st.info("현재 현장 설비에 셋팅된 공정 변수값을 슬라이더로 조절한 후, 아래 [현장 입력 조건 기반 품질 예측 연산 실행] 버튼을 눌러 계산하세요.")
        
        sim_cb = st.session_state['data_bounds']
        
        sim_col1, sim_col2 = st.columns([1, 1], gap="large")
        with sim_col1:
            sim_cd = st.slider(
                "현장 코킹 거리 입력 (Caulking_Distance, mm)",
                min_value=0.0, max_value=15.0,  # 시뮬레이터는 자유로운 범위를 탐색할 수 있도록 유연하게 지정
                value=float(round((sim_cb['Caulking_Distance'][0] + sim_cb['Caulking_Distance'][1])/2, 2)),
                step=0.01
            )
            sim_sc = st.slider(
                "현장 스터드 센터 입력 (Stud_Center, mm)",
                min_value=0.0, max_value=10.0,
                value=float(round((sim_cb['Stud_Center'][0] + sim_cb['Stud_Center'][1])/2, 2)),
                step=0.01
            )
        
        with sim_col2:
            sim_ag_label = st.radio(
                "현장 에이징 처리 상태 여부 선택 (Aging_Status)",
                options=["Unaged (미적용 - 0)", "Aged (에이징 적용 - 1)"],
                index=0
            )
            sim_ag = 1 if "Aged (에이징 적용" in sim_ag_label else 0
            
            st.markdown("<div style='margin-top: 25px; padding: 15px; background-color: #111827; border-radius: 6px; border: 1px solid #1f2937;'><span style='color:#10b981; font-weight:700;'>AI 예측 시뮬레이션 산출 근거:</span><br>엔지니어가 수동 입력한 조건이 실제 과거 공정 이력 데이터 영역(안전선) 내에 위치할수록 예측 신뢰도가 상승하며, 경험 영역 외곽으로 갈수록 수치가 하락합니다.</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("현장 입력 조건 기반 품질 예측 연산 실행", type="primary", use_container_width=True):
            X_vars = st.session_state['process_vars']
            df_sim_query = pd.DataFrame([[sim_cd, sim_sc, sim_ag]], columns=X_vars)
            scaled_sim_query = st.session_state['scaler'].transform(df_sim_query)
            
            # 1. 품질 인자 예측
            pred_tq = st.session_state['model_tq'].predict(scaled_sim_query)[0]
            pred_ed = st.session_state['model_ed'].predict(scaled_sim_query)[0]
            
            # [추가] 2. 예측 안전성 신뢰도 연산 (입력 변수가 과거 마스터 데이터 경험 영역 내에 들어오는지 평가)
            cd_min, cd_max = sim_cb['Caulking_Distance']
            sc_min, sc_max = sim_cb['Stud_Center']
            
            # 각 인자별 데이터 범위 이탈률 기준 점수 계산
            def calculate_bound_score(val, v_min, v_max):
                if v_min <= val <= v_max: return 100.0
                v_range = (v_max - v_min) if (v_max - v_min) > 0 else 1.0
                distance = min(abs(val - v_min), abs(val - v_max))
                return max(0.0, 100.0 - (distance / v_range * 200.0)) # 경험 범위를 크게 벗어날수록 감점
                
            cd_score = calculate_bound_score(sim_cd, cd_min, cd_max)
            sc_score = calculate_bound_score(sim_sc, sc_min, sc_max)
            
            st.session_state['sim_pred_tq'] = pred_tq
            st.session_state['sim_pred_ed'] = pred_ed
            st.session_state['sim_confidence'] = round((cd_score + sc_score) / 2.0, 1)
            st.session_state['sim_executed_vars'] = [sim_cd, sim_sc, sim_ag]
            st.rerun()

        if st.session_state['sim_pred_tq'] is not None:
            st.markdown("<h3 style='color:#ffffff; margin-top: 30px;'>입력된 조건에 대한 AI 품질 예측 결과 및 예측 안전성</h3>", unsafe_allow_html=True)
            s_res1, s_res2, s_res3 = st.columns(3) # [변경] 예측 신뢰도 표현을 위해 3열 구조로 확장
            with s_res1:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 25px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.95rem; font-weight:600;'>AI 예측 토크 출력값</span>
                        <h2 style='color: #ffffff; font-size: 3rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['sim_pred_tq']:.2f} <span style='font-size:1.4rem; color:#3b82f6;'>Nm</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with s_res2:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 25px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.95rem; font-weight:600;'>AI 예측 내구 수명 출력값</span>
                        <h2 style='color: #ffffff; font-size: 3rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['sim_pred_ed']:,.0f} <span style='font-size:1.4rem; color:#3b82f6;'>Cycles</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with s_res3:
                # 신뢰도가 낮으면 옐로우/레드로 시각 경고 부여
                s_conf = st.session_state['sim_confidence']
                s_conf_color = "#10b981" if s_conf >= 80.0 else ("#f59e0b" if s_conf >= 50.0 else "#ef4444")
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid {s_conf_color}; padding: 25px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.95rem; font-weight:600;'>예측 모델 데이터 신뢰도 (경험 범위)</span>
                        <h2 style='color: #ffffff; font-size: 3rem; margin: 5px 0; font-family: JetBrains Mono;'>{s_conf:.1f} <span style='font-size:1.4rem; color:{s_conf_color};'>%</span></h2>
                    </div>
                """, unsafe_allow_html=True)

            # 다운로드 버튼 생략... (이전 코드와 동일)

    # ------------------ TAB 3: 코킹 공정 원천 로그 ------------------
    with tab3:
        st.markdown("#### 원천 학습 데이터 통합 로그")
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
else:
    st.info("활성화를 위해 왼쪽 사이드바 패널에서 'JOINT-INPUT' 원천 데이터 파일을 로드한 후 실행 버튼을 클릭하십시오.")
