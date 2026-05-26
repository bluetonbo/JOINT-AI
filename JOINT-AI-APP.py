import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import time

# 1. 페이지 설정: 태블릿 및 데스크탑 확장 레이아웃
st.set_page_config(
    layout="wide", 
    page_title="Caulking Process AI Optimization",
    page_icon="⚙️"
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
        min-width: 320px !important;
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.95rem !important;
        font-weight: 600;
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

    /* 데이터프레임 테두리 정돈 */
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

# 4. Caulking 공정 전용 세션 데이터 구조 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'target_vars': ['Torque', 'Endurance'],
        'global_bounds': {
            'Caulking_Distance': (0.0, 15.0),
            'Stud_Center': (0.0, 10.0),
            'Aging_Status': (0, 1) # 0: Unaged, 1: Aged
        },
        'expert_constraints': {},
        'current_inputs': {'Caulking_Distance': 5.0, 'Stud_Center': 2.0, 'Aging_Status': 0},
        'ver': 0, 
        'expert_reliability': 0.5,
        'convergence_status': "Wait",
        'last_res_tq': None, 'last_res_ed': None, 'last_opt_df': None,
        'last_diag_tq': None, 'last_diag_ed': None
    })

# 5. 사이드바 - 데이터 관리 영역
with st.sidebar:
    st.markdown("<h3 style='color: #10b981; font-family: JetBrains Mono;'>DATA MANAGEMENT</h3>", unsafe_allow_html=True)
    with st.expander("공정 데이터 자산 로드", expanded=True):
        u1 = st.file_uploader("최적 가이드라인 표준 데이터", type=['csv','xlsx'])
        u2 = st.file_uploader("생산 이력 및 평가 데이터", type=['csv','xlsx'])
    
    if st.button("AI 엔진 및 모델 최적화 실행", type="primary"):
        if u1 and u2:
            def load_data(f):
                if f is None: return None
                return pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
            
            df_guide, df_hist = load_data(u1), load_data(u2)
            df_comb = df_hist.dropna(subset=['Torque', 'Endurance'])
            
            X_list = st.session_state['process_vars']
            
            # 독립변수 스케일러 정의 및 다중 모델 학습 (Torque, Endurance 각각 선형/다항 회귀 최적화 모델 탑재)
            scaler = MinMaxScaler().fit(df_comb[X_list])
            X_scaled = scaler.transform(df_comb[X_list])
            
            model_tq = LinearRegression().fit(X_scaled, df_comb['Torque'])
            model_ed = LinearRegression().fit(X_scaled, df_comb['Endurance'])
            
            st.session_state.update({
                'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb,
                'convergence_status': "Engine Ready"
            })
            
            # 가이드라인 기준 초기 조건 세팅
            init_row = df_guide.iloc[0].to_dict()
            for v in X_list:
                base = float(init_row.get(v, st.session_state['current_inputs'][v]))
                st.session_state['current_inputs'][v] = base
                # 자동 바운더리 설정 (Aging을 제외한 연속 변수 적용)
                if v != 'Aging_Status':
                    st.session_state['global_bounds'][v] = (round(base * 0.5, 2), round(base * 1.5, 2))
                    
            st.rerun()

# 6. 메인 통합 관제 화면
if st.session_state['model_tq']:
    st.markdown("<h1>Caulking Process <span style='color:#10b981; font-family:JetBrains Mono;'>AI Optimizer</span></h1>", unsafe_allow_html=True)
    
    # 4열 모니터링 메트릭 배치
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("AI CORE STATUS", "Operational")
    with m2: st.metric("CONTROL VARIABLES", f"{len(st.session_state['process_vars'])} EA")
    with m3: st.metric("EXPERT WEIGHT", f"{st.session_state['expert_reliability']}")
    with m4: st.metric("OPTIMIZER STATE", st.session_state['convergence_status'])

    tab1, tab2 = st.tabs(["공정 최적화 시뮬레이션", "코킹 품질 데이터 통합 로그"])

    with tab1:
        col1, col2 = st.columns([1.2, 1], gap="large")
        
        with col1:
            st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>실시간 공정 인자 제어 (X)</h4>", unsafe_allow_html=True)
            inner_cols = st.columns(2)
            
            # Caulking Distance 제어 슬라이더
            with inner_cols[0]:
                v = 'Caulking_Distance'
                b_min, b_max = st.session_state['global_bounds'][v]
                st.session_state['current_inputs'][v] = st.slider(
                    "📐 Caulking Distance (CD)", float(b_min), float(b_max), 
                    float(st.session_state['current_inputs'].get(v, b_min)), step=0.05
                )
            
            # Stud Center 제어 슬라이더
            with inner_cols[1]:
                v = 'Stud_Center'
                b_min, b_max = st.session_state['global_bounds'][v]
                st.session_state['current_inputs'][v] = st.slider(
                    "📍 Stud Center (SC)", float(b_min), float(b_max), 
                    float(st.session_state['current_inputs'].get(v, b_min)), step=0.05
                )
            
            # Aging 유무 라디오 버튼 (터치 최적화 크기)
            st.markdown("<br>", unsafe_allow_html=True)
            st.session_state['current_inputs']['Aging_Status'] = st.radio(
                "⏳ Aging 유무 (AG)", 
                options=[0, 1], 
                format_func=lambda x: "Aged (에이징 적용)" if x == 1 else "Unaged (미적용)",
                index=int(st.session_state['current_inputs']['Aging_Status'])
            )

        with col2:
            st.markdown("<h4 style='color:#e5e7eb; margin-bottom:15px;'>엔지니어 제약 조건 및 목표 (Target)</h4>", unsafe_allow_html=True)
            
            # 엔지니어가 수동으로 고정하고 싶은 타겟 지정
            selected = st.multiselect("수동 관리 타겟 인자 지정", st.session_state['process_vars'], default=list(st.session_state['expert_constraints'].keys()))
            for v in selected:
                st.session_state['expert_constraints'].setdefault(v, {'limit': st.session_state['current_inputs'].get(v, 0.0)})
                st.session_state['expert_constraints'][v]['limit'] = st.number_input(f"Target Value: {v}", value=float(st.session_state['expert_constraints'][v]['limit']))
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.session_state['expert_reliability'] = st.select_slider("도메인 지식 (Expert) 반영 가중치", options=[0.0, 0.3, 0.5, 0.7, 1.0], value=st.session_state['expert_reliability'])

        st.markdown("<br><hr style='border: 0.5px solid #1f2937;'><br>", unsafe_allow_html=True)
        
        # 하단 제어 버튼 영역
        act1, act2 = st.columns(2)
        
        # 다중 목적 평가 함수 정의 (Torque와 Endurance의 예측값을 복합 목적 함수로 변환)
        # 기본 스케일 상에서 두 값이 클수록 좋으므로 음수화하여 최소화 문제로 변환
        def evaluate_process(x_input):
            X_vars = st.session_state['process_vars']
            df_query = pd.DataFrame([x_input], columns=X_vars)
            X_scaled = st.session_state['scaler'].transform(df_query)
            
            pred_tq = st.session_state['model_tq'].predict(X_scaled)[0]
            pred_ed = st.session_state['model_ed'].predict(X_scaled)[0]
            
            # Objective: Maximize Torque & Endurance -> Minimize (-Torque - Endurance)
            # 엔지니어 패널티 제약조건 산출
            penalty = sum(abs(x_input[X_vars.index(v)] - c['limit']) / (c['limit'] or 1.0) for v, c in st.session_state['expert_constraints'].items())
            
            # Torque와 Endurance의 중요도를 동등(1:1) 비율로 목적함수 구성
            return (-pred_tq - pred_ed) + (penalty * st.session_state['expert_reliability'] * 100)

        with act1:
            if st.button("현행 조건 품질 정밀 진단"):
                current_vals = [float(st.session_state['current_inputs'].get(v, 0.0)) for v in st.session_state['process_vars']]
                
                df_q = pd.DataFrame([current_vals], columns=st.session_state['process_vars'])
                scaled_q = st.session_state['scaler'].transform(df_q)
                
                st.session_state['last_diag_tq'] = st.session_state['model_tq'].predict(scaled_q)[0]
                st.session_state['last_diag_ed'] = st.session_state['model_ed'].predict(scaled_q)[0]
                
                st.session_state['last_opt_df'] = None
                st.session_state['convergence_status'] = "Diagnosed"
                st.rerun()

        with act2:
            if st.button("AI 기반 최적 공정 솔루션 도출", type="primary"):
                X_vars = st.session_state['process_vars']
                init_guess = [float(st.session_state['current_inputs'].get(v, 0.0)) for v in X_vars]
                bounds_list = [st.session_state['global_bounds'].get(v, (0.0, 10.0)) for v in X_vars]
                
                res = minimize(evaluate_process, init_guess, method='SLSQP', bounds=bounds_list)
                
                if res.success:
                    st.session_state['convergence_status'] = "Optimization Success"
                    
                    # 최적화 결과값으로부터 도출된 변수 매핑
                    opt_x = res.x
                    df_opt_x = pd.DataFrame([opt_x], columns=X_vars)
                    scaled_opt = st.session_state['scaler'].transform(df_opt_x)
                    
                    st.session_state['last_res_tq'] = st.session_state['model_tq'].predict(scaled_opt)[0]
                    st.session_state['last_res_ed'] = st.session_state['model_ed'].predict(scaled_opt)[0]
                    
                    # 엔지니어 표기 데이터프레임 빌드
                    st.session_state['last_opt_df'] = pd.DataFrame([{
                        'Caulking_Distance (CD)': round(opt_x[0], 2),
                        'Stud_Center (SC)': round(opt_x[1], 2),
                        'Aging_Status (AG)': "Aged (적용)" if round(opt_x[2]) == 1 else "Unaged (미적용)"
                    }])
                    st.session_state['last_diag_tq'] = None
                    st.session_state['last_diag_ed'] = None
                else:
                    st.session_state['convergence_status'] = "Optimization Failed"
                st.rerun()

        # 결과 품질 대시보드 리포트 영역
        tq_val = st.session_state['last_diag_tq'] or st.session_state['last_res_tq']
        ed_val = st.session_state['last_diag_ed'] or st.session_state['last_res_ed']
        
        if tq_val is not None and ed_val is not None:
            st.markdown("<h4 style='color:#ffffff; margin-top: 25px;'>AI 공정 예측 및 분석 리포트</h4>", unsafe_allow_html=True)
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #10b981; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>예측 토크 변동값 (Torque)</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{tq_val:.2f} <span style='font-size:1.2rem; color:#10b981;'>Nm</span></h2>
                    </div>
                """, unsafe_allow_html=True)
                
            with r_col2:
                st.markdown(f"""
                    <div style='border-radius: 6px; border-left: 6px solid #3b82f6; padding: 20px; background: #1f2937;'>
                        <span style='color: #9ca3af; font-size: 0.9rem; font-weight:600;'>예측 내구 수명 (Endurance)</span>
                        <h2 style='color: #ffffff; font-size: 2.5rem; margin: 5px 0; font-family: JetBrains Mono;'>{ed_val:.1f} <span style='font-size:1.2rem; color:#3b82f6;'>Cycles</span></h2>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.session_state['last_opt_df'] is not None:
                st.markdown("<h5 style='margin-top:25px; color:#10b981;'>💡 [AI 권장] 품질 극대화를 위한 최적 Caulking 사양</h5>", unsafe_allow_html=True)
                st.dataframe(st.session_state['last_opt_df'], use_container_width=True)

    with tab2:
        st.markdown("#### Caulking 공정 통합 데이터 분석 로그")
        if not st.session_state['df_caulking'].empty:
            st.dataframe(st.session_state['df_caulking'], use_container_width=True)
        else:
            st.info("로그 데이터를 불러오지 못했습니다. 사이드바에서 원천 데이터를 로드하십시오.")
else:
    st.info("시스템 제어를 위해 왼쪽 사이드바 패널에서 실시간 생산 데이터 자산 및 가이드라인 가동 데이터를 업로드해주십시오.")
