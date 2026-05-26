import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import time

# 1. 페이지 설정: 태블릿에서도 넓게 보이도록 설정
st.set_page_config(
    layout="wide", 
    page_title="Weld Line Ai Solution",
    page_icon="---"
)

# 2. 태블릿 전용 UI/UX 스타일 (CSS 최적화)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* 전체 배경 및 폰트 */
    .stApp {
        background-color: #0e0e10 !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 사이드바 너비 조정 및 경계선 */
    [data-testid="stSidebar"] {
        background-color: #16161a !important;
        border-right: 1px solid #2d2d30;
        min-width: 320px !important;
    }

    /* 태블릿 터치를 고려한 버튼/입력창 높이 확대 */
    .stButton>button {
        height: 3.5rem !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #007acc 0%, #005a9e 100%);
        color: white !important;
        font-weight: 700;
        border: none;
    }

    /* 슬라이더 터치 영역 확대 및 라벨 가시성 */
    .stSlider label {
        color: #4fc1ff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important; /* 글자 크기 확대 */
        margin-bottom: 10px !important;
    }
    
    /* 슬라이더 트랙 높이 조정 */
    div[data-baseweb="slider"] {
        padding-top: 15px !important;
        padding-bottom: 15px !important;
    }

    /* 지표(Metrics) 카드 스타일 */
    [data-testid="stMetric"] {
        background-color: #1c1c21;
        border: 1px solid #3a3a3a;
        padding: 15px !important;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-transform: none !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4fc1ff !important;
    }

    /* 파일 업로더 라벨 및 버튼 */
    div[data-testid="stFileUploader"] label {
        color: #4fc1ff !important;
        font-weight: 600 !important;
        padding-bottom: 10px;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #444446 !important;
        color: #ffffff !important;
        height: 3rem !important;
    }

    /* 탭(Tab) 메뉴 크기 확대 */
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        height: 3rem !important;
    }

    /* 모바일/태블릿용 테이블 스크롤 최적화 */
    .stDataFrame {
        border: 1px solid #3a3a3a;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 보안 인증 (Admin1234)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([0.5, 2, 0.5])
    with center:
        st.markdown("<br><br><h2 style='text-align: center;'>Ai System Access</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("Connect System"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
    st.stop()

# 4. 세션 초기화
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None, 'scaler': None, 'df_weld': pd.DataFrame(),
        'ui_display_vars': [], 'global_process_vars': [],
        'global_bounds': {}, 'expert_constraints': {},
        'current_inputs': {}, 'ver': 0, 
        'expert_reliability': 0.5,
        'convergence_status': "Wait",
        'last_res_val': None, 'last_opt_df': None, 'last_diag_val': None
    })

# 5. 사이드바
with st.sidebar:
    st.markdown("<h3 style='color: #4fc1ff;'>Data Input</h3>", unsafe_allow_html=True)
    with st.expander("데이터 자산 관리", expanded=True):
        u1 = st.file_uploader("최적 조건 데이터", type=['csv','xlsx'])
        u2 = st.file_uploader("기존 누적 데이터", type=['csv','xlsx'])
        u3 = st.file_uploader("해석 데이터", type=['csv','xlsx'])
    
    if st.button("데이터 분석 및 솔루션 실행", type="primary"):
        if u1 and (u2 or u3):
            def load_data(f):
                if f is None: return None
                return pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
            df_i, df_v, df_r = load_data(u1), load_data(u2), load_data(u3)
            df_comb = pd.concat([df for df in [df_v, df_r] if df is not None]).dropna(subset=['Y_Weld'])
            vars_list = [c for c in df_comb.columns if c != 'Y_Weld']
            df_comb['Y_Weld'] = np.where(df_comb['Y_Weld'] >= 0.5, 1, 0)
            scaler = MinMaxScaler().fit(df_comb[vars_list])
            model = LogisticRegression().fit(scaler.transform(df_comb[vars_list]), df_comb['Y_Weld'])
            st.session_state.update({
                'model': model, 'scaler': scaler, 'df_weld': df_comb,
                'global_process_vars': vars_list, 'ui_display_vars': [c for c in df_i.columns if c != 'Y_Weld'],
                'convergence_status': "Ready"
            })
            init_row = df_i.iloc[0].to_dict()
            for v in vars_list:
                base = float(init_row.get(v, 0))
                st.session_state['global_bounds'][v] = (0, int(base * 2) if base > 0 else 100)
                st.session_state['current_inputs'][v] = int(base)
            st.rerun()

# 6. 메인 화면
if st.session_state['model']:
    st.markdown("<h1>Weld Line Ai <span style='color:#4fc1ff;'>Solution</span></h1>", unsafe_allow_html=True)
    
    # 지표 카드 (4열 배치)
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("시스템 가동 현황", "Active")
    with m2: st.metric("분석 변수", f"{len(st.session_state['ui_display_vars'])} Ea")
    with m3: st.metric("전문가 가중치", f"{st.session_state['expert_reliability']}")
    with m4: st.metric("최적화 수렴도", st.session_state['convergence_status'])

    tab1, tab2 = st.tabs(["솔루션 시뮬레이션", "데이터 통합 로그"])

    with tab1:
        # 태블릿 화면 비율에 맞춰 좌우 컬럼 자동 조정
        col1, col2 = st.columns([1.2, 1], gap="large")
        
        with col1:
            st.markdown("#### 실시간 파라미터 제어")
            ui_vars = st.session_state['ui_display_vars']
            # 슬라이더를 2열로 배치하여 터치 공간 확보
            inner_cols = st.columns(2)
            for idx, var in enumerate(ui_vars):
                with inner_cols[idx % 2]:
                    b_min, b_max = st.session_state['global_bounds'].get(var, (0, 100))
                    st.session_state['current_inputs'][var] = st.slider(
                        f"{var}", b_min, b_max, 
                        int(st.session_state['current_inputs'].get(var, b_min)),
                        key=f"sl_{var}_{st.session_state['ver']}"
                    )

        with col2:
            st.markdown("#### 전문가 제약 조건 설정")
            selected = st.multiselect("관리 포인트 설정", ui_vars, default=list(st.session_state['expert_constraints'].keys()))
            for v in selected:
                st.session_state['expert_constraints'].setdefault(v, {'limit': st.session_state['current_inputs'].get(v, 0)})
                st.session_state['expert_constraints'][v]['limit'] = st.number_input(f"Target: {v}", value=int(st.session_state['expert_constraints'][v]['limit']))
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.session_state['expert_reliability'] = st.select_slider("전문가 지식 반영 비중", options=[0.0, 0.5, 1.0], value=st.session_state['expert_reliability'])

        st.markdown("<br><hr style='border: 0.5px solid #2d2d30;'><br>", unsafe_allow_html=True)
        
        # 하단 실행 버튼 영역
        act1, act2 = st.columns(2)
        
        def run_analysis(input_vals):
            all_v = st.session_state['global_process_vars']
            ai_prob = st.session_state['model'].predict_proba(st.session_state['scaler'].transform(pd.DataFrame([input_vals], columns=all_v)))[0, 1]
            penalty = sum(abs(input_vals[list(all_v).index(v)] - c['limit']) / (c['limit'] or 1.0) for v, c in st.session_state['expert_constraints'].items())
            return ai_prob + (penalty * st.session_state['expert_reliability'])

        with act1:
            if st.button("현행 조건 정밀 진단"):
                vals = [float(st.session_state['current_inputs'].get(v, 0)) for v in st.session_state['global_process_vars']]
                st.session_state['last_diag_val'] = run_analysis(vals)
                st.session_state['last_opt_df'] = None
                st.session_state['convergence_status'] = "Diagnosed"
                st.rerun()

        with act2:
            if st.button("Ai 최적 솔루션 도출", type="primary"):
                all_v = st.session_state['global_process_vars']
                res = minimize(run_analysis, [float(st.session_state['current_inputs'].get(v, 0)) for v in all_v], method='SLSQP', bounds=[st.session_state['global_bounds'].get(v, (0, 100)) for v in all_v])
                
                if res.success:
                    st.session_state['convergence_status'] = "Success"
                    st.session_state['last_res_val'] = res.fun
                    st.session_state['last_opt_df'] = pd.DataFrame([{v: int(round(val)) for v, val in zip(all_v, res.x) if v in ui_vars}])
                    st.session_state['last_diag_val'] = None
                else:
                    st.session_state['convergence_status'] = "Failed"
                st.rerun()

        # 결과 리포트 영역
        res_val = st.session_state['last_diag_val'] or st.session_state['last_res_val']
        if res_val is not None:
            color = "#f44747" if res_val > 0.4 else "#4ec9b0"
            st.markdown(f"""
                <div style='border-radius: 12px; border-left: 8px solid {color}; padding: 25px; background: #1c1c21; margin-top: 30px;'>
                    <span style='color: #bbbbbb; font-size: 1rem;'>Weld Line 결함 위험 지수</span>
                    <h2 style='color: {color}; font-size: 3rem; margin: 10px 0;'>{res_val*100:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state['last_opt_df'] is not None:
                st.markdown("<h5 style='margin-top:25px; color:#4fc1ff;'>[권장] Ai 기반 최적 사출 조건</h5>", unsafe_allow_html=True)
                st.dataframe(st.session_state['last_opt_df'], use_container_width=True)

    with tab2:
        st.markdown("#### 통합 데이터 로그")
        st.dataframe(st.session_state['df_weld'], use_container_width=True)
else:
    st.info("사이드바에서 데이터를 업로드하여 Ai 엔진을 활성화하십시오.")