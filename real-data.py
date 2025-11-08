import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler 

# =================================================================
# 0. 초기 설정 및 상수
# =================================================================
st.set_page_config(layout="wide", page_title="Weld Line 통합 진단 시스템")

# 공정 변수 정의 (X 변수)
PROCESS_VARS = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
# 종속 변수 정의 (Y 변수)
TARGET_VAR = 'Y_Weld'
# 불량 기준 (0.5 이상이면 1, 미만이면 0)
DEFECT_THRESHOLD = 0.5

# 슬라이더 및 입력 필드의 기본값 정의
DEFAULT_INPUT_VALS = {
    'T_Melt': 230, 'V_Inj': 3, 'P_Pack': 70, 
    'T_Mold': 50, 'Meter': 195, 'VP_Switch_Pos': 14
}

# 시스템 상태 초기화 (세션 상태)
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'df_weld' not in st.session_state:
    st.session_state['df_weld'] = pd.DataFrame()
if 'df_init' not in st.session_state:
    st.session_state['df_init'] = None
if 'df_virtual' not in st.session_state:
    st.session_state['df_virtual'] = None
if 'df_real' not in st.session_state:
    st.session_state['df_real'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
    
# 진단 결과 저장을 위한 세션 상태 추가
if 'current_risk_display' not in st.session_state:
    st.session_state['current_risk_display'] = None
if 'optimization_result' not in st.session_state:
    st.session_state['optimization_result'] = None

# 슬라이더 오류 방지 로직: 초기값을 무조건 float으로 설정
for var, default_val in DEFAULT_INPUT_VALS.items():
    if f'input_{var}' not in st.session_state:
        st.session_state[f'input_{var}'] = float(default_val)

# UI 상태를 위한 세션 상태 추가
if 'conf_level' not in st.session_state:
    st.session_state['conf_level'] = 75.0
# 🌟 노하우 영향 계수 세션 상태 초기화
if 'influence_factor_display_val' not in st.session_state:
    st.session_state['influence_factor_display_val'] = st.session_state['conf_level'] / 100.0

if 'v_inj_qual_apply' not in st.session_state:
    st.session_state['v_inj_qual_apply'] = False
if 'v_inj_quant_apply' not in st.session_state:
    st.session_state['v_inj_quant_apply'] = False
if 't_mold_qual_apply' not in st.session_state:
    st.session_state['t_mold_qual_apply'] = False
if 't_mold_quant_apply' not in st.session_state:
    st.session_state['t_mold_quant_apply'] = False
if 'v_inj_qual_intent' not in st.session_state:
    st.session_state['v_inj_qual_intent'] = 'Keep_Constant'
if 't_mold_qual_intent' not in st.session_state:
    st.session_state['t_mold_qual_intent'] = 'Keep_Constant'

# 🌟 정량적 노하우 입력 값 (퍼센트)
if 'v_inj_quant_percent' not in st.session_state:
    st.session_state['v_inj_quant_percent'] = 0.0
if 't_mold_quant_percent' not in st.session_state:
    st.session_state['t_mold_quant_percent'] = 0.0


# -------------------------------------------------------------
# 🌟 콜백 함수: 전문가 확신 수준 변경 시 영향 계수 업데이트
# -------------------------------------------------------------
def update_influence_factor():
    """전문가 확신 수준 슬라이더 변경 시 영향 계수를 업데이트하고 진단 결과를 초기화합니다."""
    
    # 🌟 슬라이더 키의 값을 직접 가져와서 사용 (가장 최신 값)
    if 'expert_confidence_slider' in st.session_state:
        new_confidence_level = st.session_state['expert_confidence_slider']
    else:
        new_confidence_level = st.session_state['conf_level'] 
        
    new_influence_factor = new_confidence_level / 100.0
    
    # 🌟 'conf_level'과 'influence_factor_display_val' 세션 상태를 업데이트
    st.session_state['conf_level'] = new_confidence_level
    st.session_state['influence_factor_display_val'] = new_influence_factor
    
    # 노하우가 변경되었으므로 진단 및 최적화 결과 초기화
    st.session_state['current_risk_display'] = None 
    st.session_state['optimization_result'] = None 


# =================================================================
# 1. 데이터 로드 및 전처리 함수 (생략, 이전과 동일)
# =================================================================

# ... (load_df_from_uploader, process_weld_data 함수는 생략됨, 이전 코드와 동일)
# ... (train_model, predict_weld_risk 함수는 생략됨, 이전 코드와 동일)

@st.cache_data(show_spinner=False)
def load_df_from_uploader(uploaded_file):
    """업로드된 파일(xlsx, csv)을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"⚠️ 지원하지 않는 파일 형식입니다: .{file_extension}")
                return None
            
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """실제 데이터와 가상 데이터를 결합하고 전처리합니다."""
    
    valid_dataframes = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    
    if not valid_dataframes:
        return pd.DataFrame() 

    df_combined = pd.concat(valid_dataframes, ignore_index=True)
    
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    
    required_cols = PROCESS_VARS + [TARGET_VAR]
    if not all(col in df_combined.columns for col in required_cols):
        st.error("⚠️ 데이터에 필수 컬럼(T_Melt, V_Inj, ..., Y_Weld)이 누락되었습니다. 컬럼 이름을 확인해 주세요.")
        return pd.DataFrame()
        
    df_processed = df_combined[required_cols].copy()
    
    return df_processed

def train_model(df):
    """데이터를 사용하여 로지스틱 회귀 모델을 학습하고 스케일러를 저장합니다."""
    if df.empty:
        return None, None
        
    X = df[PROCESS_VARS]
    Y = df[TARGET_VAR]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, Y)
    
    return model, scaler

def predict_weld_risk(model, scaler, input_data):
    """입력 데이터에 대한 불량 확률을 예측합니다."""
    if model is None or scaler is None:
        return 0.5 
        
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data], columns=PROCESS_VARS)
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()], columns=PROCESS_VARS)
    elif isinstance(input_data, pd.DataFrame) and len(input_data) == 1:
         input_df = input_data[PROCESS_VARS]
    else:
        return 0.5
    
    input_scaled = scaler.transform(input_df)
    
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    
    return prediction_proba


# =================================================================
# 4. Streamlit UI 및 로직
# =================================================================

# -----------------
# 사이드바 (데이터 로드)
# -----------------
with st.sidebar:
    st.header("📂 데이터 및 모델 학습")
    
    uploaded_file_init = st.file_uploader(
        "1. UI 초기 조건 (initial_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="init_file"
    )
    uploaded_file_virtual = st.file_uploader(
        "2. 가상 학습 데이터 (test_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="virtual_file"
    )
    uploaded_file_real = st.file_uploader(
        "3. 해석 학습 데이터 (moldflow_condition.xlsx) [필수]", type=['xlsx', 'csv'], key="real_file"
    )

    st.session_state['df_init'] = load_df_from_uploader(uploaded_file_init)
    st.session_state['df_virtual'] = load_df_from_uploader(uploaded_file_virtual)
    st.session_state['df_real'] = load_df_from_uploader(uploaded_file_real)


    def load_and_train_model():
        st.session_state['current_risk_display'] = None
        st.session_state['optimization_result'] = None
        
        df_weld_processed = process_weld_data(st.session_state['df_virtual'], st.session_state['df_real'])
        st.session_state['df_weld'] = df_weld_processed
        
        if st.session_state['df_weld'].empty:
            st.error("🚨 모델 학습 실패: 필수 데이터(3번 파일)가 로드되지 않았습니다.")
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            return

        model, scaler = train_model(st.session_state['df_weld'])
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

        if model is not None:
            st.success("✅ AI 모델 학습 및 로드 완료! UI에 초기 조건이 반영되었습니다.")
            
            if st.session_state['df_init'] is not None and not st.session_state['df_init'].empty:
                init_row = st.session_state['df_init'].iloc[0]
                for var in PROCESS_VARS:
                    if var in init_row:
                        try:
                            st.session_state[f'input_{var}'] = float(init_row[var])
                        except ValueError:
                            st.warning(f"⚠️ 초기 조건 파일의 '{var}' 값이 유효한 숫자가 아닙니다. 기본값을 유지합니다.")


    st.button("🚀 파일 로드 및 AI 모델 학습 시작", on_click=load_and_train_model)

    st.markdown("---")
    st.header("ℹ️ 시스템 상태 확인")

    if st.session_state['model'] is not None:
        st.success("모델 상태: 학습 완료")
        
        total_count = len(st.session_state['df_weld'])
        defect_count = st.session_state['df_weld'][TARGET_VAR].sum()
        defect_rate = (defect_count / total_count) * 100 if total_count > 0 else 0
        
        st.write(f"총 데이터 개수: **{total_count}개**")
        st.write(f"불량 비율(Y=1): **{defect_rate:.1f}%**")
        
        if defect_rate == 0:
            st.warning("⚠️ 경고: 학습 데이터에 불량(1) 샘플이 0개입니다. 정확한 진단이 어려울 수 있습니다.")
    else:
        st.warning("모델 상태: 학습 필요")
        

# -----------------
# 메인 페이지 (진단 UI)
# -----------------
st.title("Weld Line AI 통합 진단 및 최적화 시스템")

tab1, tab2 = st.tabs(["탭 1. 진단 및 최적 공정 조건 제시", "탭 2. 모델 및 데이터 확인"])

with tab1:
    st.header("A. 현재 공정 조건 입력")
    
    col_melt, col_inj, col_pack = st.columns(3)
    col_mold, col_meter, col_vp = st.columns(3)

    input_vars = {}
    
    # 공정 변수 슬라이더 (On_change는 결과를 초기화하여 재실행을 유도)
    for col, var, label, min_val, max_val, step, unit in zip(
        [col_melt, col_inj, col_pack, col_mold, col_meter, col_vp],
        PROCESS_VARS,
        ['용융 온도', '사출 속도', '보압', '금형 온도', '계량 위치', 'VP 전환 위치'],
        [200.0, 1.0, 50.0, 30.0, 180.0, 10.0],
        [300.0, 10.0, 100.0, 80.0, 200.0, 20.0],
        [5.0, 1.0, 5.0, 5.0, 1.0, 1.0],
        ['°C', 'mm/s', 'MPa', '°C', 'mm', 'mm']
    ):
        with col:
            # 🌟 슬라이더의 on_change는 결과를 초기화하여 재실행을 유도
            input_vars[var] = st.slider(
                f'{label} ({var}) [{unit}]', 
                min_val, 
                max_val, 
                value=st.session_state[f'input_{var}'], 
                step=step, 
                key=f'slider_{var}',
                on_change=lambda: st.session_state.update({'current_risk_display': None, 'optimization_result': None})
            )

    st.markdown("---")
    
    # -------------------------------------------------------------
    # B. 전문가의 정성적/정량적 노하우 입력 (이미지 형식 반영)
    # -------------------------------------------------------------
    st.header("B. 전문가의 정성적/정량적 노하우 입력")

    # 1. 전문가 확신 수준 (반영도)
    st.subheader("1. 전문가 확신 수준")
    st.write("전문가 확신 수준") 
    # 🌟 expert_confidence_slider 값 변경 시 update_influence_factor 콜백 실행
    expert_confidence = st.slider(
        '노하우 반영도 (%)', 
        0.0, 
        100.0, 
        value=st.session_state['conf_level'], 
        step=5.0, 
        label_visibility="collapsed",
        key='expert_confidence_slider',
        on_change=update_influence_factor 
    )
    # st.session_state['conf_level'] 값은 콜백 내에서 업데이트됨
    st.markdown('<div style="margin-top: -20px; font-size: 12px; color: grey;">(0%는 노하우 미반영, 100%는 노하우를 제약 조건으로 강력히 적용)</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. 사출 속도 (extV_Inj)
    # -------------------------------------------------------------
    st.subheader("2. 사출 속도 (extV_Inj)")
    
    col_v_qual, col_v_intent, col_v_quant, col_v_delta = st.columns(4)
    
    with col_v_qual:
        v_inj_qual_apply = st.checkbox(
            '정성적 노하우 적용', 
            value=st.session_state['v_inj_qual_apply'],
            key='v_inj_qual_apply_chk',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['v_inj_qual_apply'] = v_inj_qual_apply
    
    with col_v_intent:
        v_inj_intent = st.selectbox(
            'V_Inj 조절 의도', 
            ['Keep_Constant', 'Increase', 'Decrease'], 
            index=['Keep_Constant', 'Increase', 'Decrease'].index(st.session_state['v_inj_qual_intent']),
            disabled=not v_inj_qual_apply,
            key='intent_v_inj_selectbox',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['v_inj_qual_intent'] = v_inj_intent

    with col_v_quant:
        v_inj_quant_apply = st.checkbox(
            '정량적 노하우 적용', 
            value=st.session_state['v_inj_quant_apply'],
            key='v_inj_quant_apply_chk',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['v_inj_quant_apply'] = v_inj_quant_apply
        
    with col_v_delta:
        # 🌟 정량적 노하우 입력 값 변경 (0~100% 범위)
        st.write('V_Inj 노하우 변화율 (%)')
        v_inj_quant_percent = st.slider(
            'V_Inj 변화율', 
            0.0, 
            100.0, 
            value=st.session_state['v_inj_quant_percent'], 
            step=1.0,
            label_visibility="collapsed",
            disabled=not v_inj_quant_apply,
            key='v_inj_quant_percent_slider',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['v_inj_quant_percent'] = v_inj_quant_percent
    
    # -------------------------------------------------------------
    # 3. 금형 온도 (extT_Mold)
    # -------------------------------------------------------------
    st.subheader("3. 금형 온도 (extT_Mold)")

    col_t_qual, col_t_intent, col_t_quant, col_t_delta = st.columns(4)
    
    with col_t_qual:
        t_mold_qual_apply = st.checkbox(
            '정성적 노하우 적용', 
            value=st.session_state['t_mold_qual_apply'],
            key='t_mold_qual_apply_chk',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['t_mold_qual_apply'] = t_mold_qual_apply
    
    with col_t_intent:
        t_mold_intent = st.selectbox(
            'T_Mold 조절 의도', 
            ['Keep_Constant', 'Increase', 'Decrease'], 
            index=['Keep_Constant', 'Increase', 'Decrease'].index(st.session_state['t_mold_qual_intent']),
            disabled=not t_mold_qual_apply,
            key='intent_t_mold_selectbox',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['t_mold_qual_intent'] = t_mold_intent

    with col_t_quant:
        t_mold_quant_apply = st.checkbox(
            '정량적 노하우 적용', 
            value=st.session_state['t_mold_quant_apply'],
            key='t_mold_quant_apply_chk',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['t_mold_quant_apply'] = t_mold_quant_apply
        
    with col_t_delta:
        # 🌟 정량적 노하우 입력 값 변경 (0~100% 범위)
        st.write('T_Mold 노하우 변화율 (%)')
        t_mold_quant_percent = st.slider(
            'T_Mold 변화율', 
            0.0, 
            100.0, 
            value=st.session_state['t_mold_quant_percent'], 
            step=1.0,
            label_visibility="collapsed",
            disabled=not t_mold_quant_apply,
            key='t_mold_quant_percent_slider',
            on_change=lambda: st.session_state.update({'optimization_result': None})
        )
        st.session_state['t_mold_quant_percent'] = t_mold_quant_percent


    st.markdown("---")

    # -----------------
    # C. 진단 실행 및 결과
    # -----------------
    st.header("C. 진단 실행 및 결과")

    # 🌟 노하우 영향 계수 (세션 상태 값 참조)
    st.write("노하우 영향 계수")
    st.slider(
        '노하우 영향 계수 (0.0~1.0)', 
        0.0, 
        1.0, 
        value=st.session_state['influence_factor_display_val'], 
        step=0.01, 
        label_visibility="collapsed",
        disabled=True
    )
    
    st.markdown("---")


    # -----------------
    # 진단 실행 및 최적화 함수
    # -----------------
    
    def run_diagnosis_callback(input_vars):
        """진단 버튼 클릭 시 현재 조건 진단 실행"""
        if st.session_state['model'] is None:
            st.session_state['current_risk_display'] = "🚨 모델이 학습되지 않았습니다."
            return

        current_risk = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], input_vars)
        st.session_state['current_risk_display'] = current_risk
        st.session_state['optimization_result'] = None 

    
    def run_optimization_callback(input_vars, v_inj_intent, v_inj_quant_percent, v_inj_quant_apply, t_mold_intent, t_mold_quant_percent, t_mold_quant_apply):
        """최적 공정 조건 제시 버튼 클릭 시 실행"""
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        
        if model is None:
            st.session_state['optimization_result'] = {"success": False, "message": "모델이 학습되지 않았습니다."}
            return

        def objective_function(X_array):
            X_df = pd.DataFrame([X_array], columns=PROCESS_VARS)
            return predict_weld_risk(model, scaler, X_df.iloc[0].to_dict())

        X0 = np.array([input_vars[var] for var in PROCESS_VARS], dtype=float)
        
        # 🌟 최적화에 사용될 노하우 반영 계수는 세션 상태에서 가져옴
        influence_factor = st.session_state['influence_factor_display_val'] 

        constraints = []
        
        fixed_vars = ['T_Melt', 'P_Pack', 'Meter', 'VP_Switch_Pos']
        for var in fixed_vars:
            idx = PROCESS_VARS.index(var)
            constraints.append({'type': 'eq', 
                                 'fun': lambda X, idx=idx, val=X0[idx]: X[idx] - val})

        # =========================================================
        # V_Inj 노하우 제약 (Bounds 설정) - 퍼센트 반영
        # =========================================================
        v_min_global, v_max_global = 1.0, 10.0
        v_min_opt, v_max_opt = v_min_global, v_max_global
        
        current_v_inj = input_vars['V_Inj']
        
        if v_inj_quant_apply or (v_inj_qual_apply and v_inj_intent != 'Keep_Constant'):
            # 🌟 현재 값에 대한 퍼센트 변화량 계산
            delta_v_inj = current_v_inj * (v_inj_quant_percent / 100.0) 
            final_delta = delta_v_inj * influence_factor 
            
            if v_inj_intent == 'Increase':
                v_min_opt = max(v_min_global, current_v_inj + final_delta)
            elif v_inj_intent == 'Decrease':
                v_max_opt = min(v_max_global, current_v_inj - final_delta)
        elif v_inj_qual_apply and v_inj_intent == 'Keep_Constant':
             v_min_opt = current_v_inj
             v_max_opt = current_v_inj


        # =========================================================
        # T_Mold 노하우 제약 (Bounds 설정) - 퍼센트 반영
        # =========================================================
        t_min_global, t_max_global = 30.0, 80.0
        t_min_opt, t_max_opt = t_min_global, t_max_global
        
        current_t_mold = input_vars['T_Mold']
        
        if t_mold_quant_apply or (t_mold_qual_apply and t_mold_intent != 'Keep_Constant'):
            # 🌟 현재 값에 대한 퍼센트 변화량 계산
            delta_t_mold = current_t_mold * (t_mold_quant_percent / 100.0)
            final_delta = delta_t_mold * influence_factor
            
            if t_mold_intent == 'Increase':
                t_min_opt = max(t_min_global, current_t_mold + final_delta)
            elif t_mold_intent == 'Decrease':
                t_max_opt = min(t_max_global, current_t_mold - final_delta)
        elif t_mold_qual_apply and t_mold_intent == 'Keep_Constant':
             t_min_opt = current_t_mold
             t_max_opt = current_t_mold

        # 변수별 경계 설정 (Bounds)
        bounds = [
            (200.0, 300.0),      
            (v_min_opt, v_max_opt), 
            (50.0, 100.0),      
            (t_min_opt, t_max_opt), 
            (180.0, 200.0),     
            (10.0, 20.0)        
        ]

        try:
            result = minimize(objective_function, X0, method='SLSQP', bounds=bounds, constraints=constraints)
        
            if result.success:
                opt_params = {PROCESS_VARS[i]: round(result.x[i], 1) for i in range(len(PROCESS_VARS))}
                opt_risk = predict_weld_risk(model, scaler, opt_params)
                
                st.session_state['optimization_result'] = {
                    "success": True,
                    "opt_params": opt_params,
                    "opt_risk": opt_risk,
                    "influence_factor": influence_factor
                }
            else:
                st.session_state['optimization_result'] = {"success": False, "message": f"최적화 실패: {result.message}"}

        except Exception as e:
            st.session_state['optimization_result'] = {"success": False, "message": f"최적화 실행 중 치명적인 오류 발생: {e}"}

    # -----------------
    # 버튼 실행
    # -----------------
    col_diag, col_opt = st.columns([1,1])
    with col_diag:
        st.button("🔴 Weld Line 통합 진단 실행", 
                  on_click=run_diagnosis_callback, 
                  args=(input_vars,), 
                  use_container_width=True)
    with col_opt:
        st.button("✨ 최적 공정 조건 제시", 
                  on_click=run_optimization_callback, 
                  args=(input_vars, 
                        v_inj_intent, st.session_state['v_inj_quant_percent'], v_inj_quant_apply,
                        t_mold_intent, st.session_state['t_mold_quant_percent'], t_mold_quant_apply), 
                  use_container_width=True)

    st.markdown("---")
    st.header("D. 진단 및 최적화 결과")

    # 1. 현재 조건 진단 결과 출력
    if st.session_state['current_risk_display'] is not None:
        if isinstance(st.session_state['current_risk_display'], float):
            current_risk = st.session_state['current_risk_display']
            st.subheader("1. 현재 조건 진단")
            st.info(f"🟢 현재 조건에서의 불량 위험 확률: **{current_risk*100:.2f}%**")
            
            if current_risk >= DEFECT_THRESHOLD:
                st.error("🔴 위험도 높음: 즉시 최적화 조건을 검토하세요.")
            else:
                st.success("🟢 위험도 낮음: 현재 조건을 유지해도 좋습니다.")
        else:
             st.warning(f"⚠️ 진단 오류: {st.session_state['current_risk_display']}")
    else:
        st.info("⬆️ 상단 버튼을 눌러 **'Weld Line 통합 진단'**을 먼저 실행하세요.")
        

    # 2. 최적화 결과 출력
    if st.session_state['optimization_result'] is not None:
        st.subheader("2. 최적 공정 조건 제시")
        result = st.session_state['optimization_result']
        
        if result["success"]:
            opt_params = result["opt_params"]
            opt_risk = result["opt_risk"]
            
            st.success("✨ 최적 공정 조건 제시 결과")
            st.write(f"**최소 불량 위험 확률:** **{opt_risk*100:.2f}%**")
            
            opt_table = pd.DataFrame([opt_params])
            opt_table = opt_table.T.rename(columns={0: '최적 공정 조건'})
            st.dataframe(opt_table)
            
            st.markdown("##### 🔍 최적화 요약")
            
            summary_data = {}
            for var in PROCESS_VARS:
                current_val = round(input_vars[var], 1)
                opt_val = opt_params[var]
                if current_val != opt_val:
                    change = "↑ 상향" if opt_val > current_val else "↓ 하향"
                    summary_data[var] = f"{opt_val} ({change})"
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data.values(), index=summary_data.keys(), columns=['변화된 조건'])
                summary_df.index.name = '변수'
                st.table(summary_df)
            else:
                st.info("현재 조건이 이미 최적 조건에 가깝거나, 노하우 제약 조건으로 인해 더 이상 개선되지 않았습니다.")
                
        else:
            st.error(f"⚠️ 최적화 실패: {result['message']}")


with tab2:
    st.header("모델 및 데이터 확인")
    
    if st.session_state['model'] is not None:
        model = st.session_state['model']
        st.subheader("1. 학습된 로지스틱 회귀 모델 계수")
        
        coefficients = pd.DataFrame({
            '변수': ['(절편)'] + PROCESS_VARS,
            '계수(Coefficient)': [model.intercept_[0]] + list(model.coef_[0])
        })
        st.dataframe(coefficients.set_index('변수'))
        st.info("💡 계수의 절대값이 클수록 Weld Line 불량 위험 예측에 미치는 영향이 큽니다.")

        st.subheader("2. 학습 데이터 미리보기")
        if not st.session_state['df_weld'].empty:
            st.dataframe(st.session_state['df_weld'])
        else:
            st.warning("학습 데이터가 없습니다.")
    else:
        st.warning("모델 학습이 필요합니다.")
