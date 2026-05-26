import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정: 태블릿 및 데스크탑 확장 레이아웃
st.set_page_config(
    layout="wide", 
    page_title="Caulking Parameter Target Finder",
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
    .stButton>button {
        height: 3.5rem !important;
        font-size: 1.05rem !important;
        border-radius: 6px !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white !important;
        font-weight: 700;
        border: 1px solid #10b981;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
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
        'global_bounds': {
            'Caulking_Distance': (4.0, 7.0),
            'Stud_Center': (1.5, 3.5),
            'Aging_Status': (0, 1)
        },
        'optimizer_status': "Wait",
        'target_tq_range': (35.0, 37.0),
        'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None
    })

# 5. 사이드바 - JOINT-INPUT 데이터 패널 관리
with st.sidebar:
    st.markdown("<h3 style='color: #10b981; font-family: JetBrains Mono;'>DATA MANAGEMENT</h3>", unsafe_allow_html=True)
    with st.expander("공정 데이터 자산 로드", expanded=True):
        u_input = st.file_uploader("JOINT-INPUT 데이터 업로드 (CSV, XLSX)", type=['csv','xlsx'])
    
    if st.button("AI 엔진 및 모델 최적화 가동", type="primary"):
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
                'global_bounds': {
                    'Caulking_Distance': (float(df_comb['Caulking_Distance'].min() * 0.9), float(df_comb['Caulking_Distance'].max() * 1.1)),
                    'Stud_Center': (float(df_comb['Stud_Center'].min() * 0.9), float(df_comb['Stud_Center'].max() * 1.1)),
                    'Aging_Status': (0, 1)
                }
            })
            st.rerun()
        else:
            st.error("JOINT-INPUT 파일을 업로드해 주세요.")

# 6. 메인 통합 관제 대시보드
if st.session_state['model_tq']:
    st.markdown("<h1>Caulking Target <span style='color:#10b981; font-family:JetBrains Mono;'>Condition Finder</span></h1>", unsafe_allow_html=True)
    
    # 상단 모니터링 메트릭 영역
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("AI LEARNING ENGINE", "ACTIVE")
    with m2: st.metric("HISTORICAL DATA LOGS", f"{len(st.session_state['df_caulking'])} Rows")
    with m3: st.metric("FINDER SYSTEM STATUS", st.session_state['optimizer_status'])

    tab1, tab2 = st.tabs(["품질 타겟 추적 솔루션", "코킹 공정 원천 로그"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>공정 탐색 알고리즘 경계 조건</h4>", unsafe_allow_html=True)
            st.info("AI 엔진이 설비 고장을 방지하기 위해 데이터 분석 기반 안전 한계선 내에서 공정 변수(CD, SC)의 최적 조합을 탐색합니다.")
            
            cb = st.session_state['global_bounds']
            st.markdown(f"""
            * **코킹 거리(CD) 탐색 영역:** {cb['Caulking_Distance'][0]:.2f} mm ~ {cb['Caulking_Distance'][1]:.2f} mm
            * **스터드 센터(SC) 탐색 영역:** {cb['Stud_Center'][0]:.2f} mm ~ {cb['Stud_Center'][1]:.2f} mm
            """)

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
        
        # 역산 실행 프로세스 부
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
            bounds_list = [st.session_state['global_bounds'].get(v, (0.0, 10.0)) for v in X_vars]
            
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

        # 결과 리포트 대시보드 출력부
        if st.session_state['opt_result_x'] is not None:
            opt_x = st.session_state['opt_result_x']
            aging_text = "Aged (에이징 적용)" if round(opt_x[2]) == 1 else "Unaged (미적용)"
            
            # 상단 영역: [수정] 표 형태를 제거하고 예상 토크값과 동일한 3열 카드 모양으로 교체
            st.markdown("<h3 style='color:#ffffff; margin-top: 25px;'>AI 분석 기반 권장 공정 사양</h3>", unsafe_allow_html=True)
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 코킹 거리 (Caulking_Distance)</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{opt_x[0]:.2f} <span style='font-size:1.2rem; color:#10b981;'>mm</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with c_col2:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 스터드 센터 (Stud_Center)</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{opt_x[1]:.2f} <span style='font-size:1.2rem; color:#10b981;'>mm</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with c_col3:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>추천 에이징 여부 (Aging_Status)</span>
                        <h2 style='color: #ffffff; font-size: 1.8rem; margin: 12px 0; font-family: Inter; font-weight:700;'>{aging_text}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # 하단 영역: 품질 변수 만족 스펙 및 신뢰도 모니터링 카드 (3열 배치)
            st.markdown("<h3 style='color:#ffffff; margin-top: 25px;'>해당 조건 세팅 시 예상 품질 인자 및 신뢰도</h3>", unsafe_allow_html=True)
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>해당 조건 세팅 시 예상 토크 값</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['opt_pred_tq']:.2f} <span style='font-size:1.2rem; color:#3b82f6;'>Nm</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with r_col2:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>해당 조건 세팅 시 예상 내구 수명</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['opt_pred_ed']:,.0f} <span style='font-size:1.2rem; color:#3b82f6;'>Cycles</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            with r_col3:
                conf_color = "#10b981" if st.session_state['confidence_score'] >= 80.0 else "#ef4444"
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid {conf_color}; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>품질 조건 만족 범위 신뢰도</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{st.session_state['confidence_score']:.1f} <span style='font-size:1.2rem; color:{conf_color};'>%</span></h2>
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 원천 학습 데이터 통합 로그")
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
else:
    st.info("대시보드 활성화를 위해 왼쪽 사이드바 패널에서 'JOINT-INPUT' 원천 데이터 파일을 로드한 후 가동 버튼을 클릭하십시오.")
