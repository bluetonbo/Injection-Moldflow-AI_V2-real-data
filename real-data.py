import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize 

# -----------------------------------------------------------------------------
# 0. 전역 설정 (GLOBAL CONFIGURATION)
# -----------------------------------------------------------------------------
KNOWHOW_FEATURES = [
    'Expert_Confidence', 
    'V_Inj_Intent_Low_Decrease', 
    'V_Inj_Intent_High_Increase',
    'T_Mold_Intent_Low_Decrease',
    'T_Mold_Intent_High_Increase',
    'V_Inj_Delta_Scaled',
    'T_Mold_Delta_Scaled'
]

# 기본 초기값 (initial_condition.xlsx 파일이 없을 경우 사용)
DEFAULT_INITIAL_VALS = {
    'T_Melt': 240.0, 'V_Inj': 80.0, 'P_Pack': 80.0, 
    'T_Mold': 80.0, 'Meter': 18.00, 'VP_Switch_Pos': 10.50
}


# -----------------------------------------------------------------------------
# 1. 데이터 로딩 및 모델 학습 로직
# -----------------------------------------------------------------------------

def load_df_from_uploader(uploaded_file):
    """업로드된 파일 객체에서 Pandas DataFrame을 로드합니다."""
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            st.error(f"⚠️ 파일 로드 오류: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """업로드된 두 DataFrame을 병합하고 학습을 위한 컬럼을 처리합니다."""
    
    # df_virtual 또는 df_real 중 하나만 있어도 학습 데이터로 사용
    df_combined = pd.concat([df_real, df_virtual], ignore_index=True)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    
    if 'Expert_Confidence' not in df_combined.columns:
        df_combined['Expert_Confidence'] = 75 
        
    df_combined['T_Weld'] = df_combined['T_Melt'] * 0.8 + df_combined['T_Mold'] * 0.2 + df_combined['V_Inj'] * 0.1
    df_combined['t_Fill'] = 3.0 - 0.015 * df_combined['V_Inj']
    
    # 임시 Delta 값 생성 (데이터 다양성 확보용)
    if 'V_Inj_Delta' not in df_combined.columns or 'T_Mold_Delta' not in df_combined.columns:
        df_combined['V_Inj_Delta'] = 0.0
        if 'V_Inj_Intent' in df_combined.columns:
            df_combined.loc[df_combined['V_Inj_Intent'].astype(str).str.contains('Increase'), 'V_Inj_Delta'] = 10.0
            df_combined.loc[df_combined['V_Inj_Intent'].astype(str).str.contains('Decrease'), 'V_Inj_Delta'] = -5.0
        
        df_combined['T_Mold_Delta'] = 0.0
        if 'T_Mold_Intent' in df_combined.columns:
            df_combined.loc[df_combined['T_Mold_Intent'].astype(str).str.contains('Increase'), 'T_Mold_Delta'] = 8.0
            df_combined.loc[df_combined['T_Mold_Intent'].astype(str).str.contains('Decrease'), 'T_Mold_Delta'] = -4.0

    # Delta Scaler 저장 (UI 입력값 스케일링을 위해)
    try:
        st.session_state['scaler_delta_v'] = StandardScaler().fit(df_combined[['V_Inj_Delta']])
        st.session_state['scaler_delta_t'] = StandardScaler().fit(df_combined[['T_Mold_Delta']])
    except ValueError:
        st.session_state['scaler_delta_v'] = StandardScaler()
        st.session_state['scaler_delta_v'].fit(np.array([0.0, 1.0]).reshape(-1, 1))
        st.session_state['scaler_delta_t'] = StandardScaler()
        st.session_state['scaler_delta_t'].fit(np.array([0.0, 1.0]).reshape(-1, 1))
        st.warning("⚠️ V_Inj_Delta 또는 T_Mold_Delta 값이 데이터에 없어 임시 스케일러를 사용합니다.")

    df_combined['V_Inj_Delta_Scaled'] = st.session_state['scaler_delta_v'].transform(df_combined[['V_Inj_Delta']])
    df_combined['T_Mold_Delta_Scaled'] = st.session_state['scaler_delta_t'].transform(df_combined[['T_Mold_Delta']])
    
    return df_combined

@st.cache_resource
def train_model(df):
    """모델을 학습하고 평가합니다."""
    
    X = df.drop(columns=['L_Weld', 'Y_Weld', 'V_Inj_Delta', 'T_Mold_Delta'])
    y = df['Y_Weld']

    if len(y.unique()) < 2:
        # 데이터가 있지만 불량(1) 샘플이 없는 경우
        st.error(f"🚨 치명적 오류: 학습 데이터에 불량(1) 샘플이 부족합니다. 현재 불량률: {df['Y_Weld'].mean()*100:.1f}%.")
        raise ValueError("불량 샘플이 부족합니다.")
    
    X = pd.get_dummies(X, columns=['V_Inj_Intent', 'T_Mold_Intent'], drop_first=True)
    
    numerical_features = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos', 'T_Weld', 't_Fill']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X.columns.tolist(), accuracy, len(df), df['Y_Weld'].mean()

def get_knowhow_inputs():
    """UI 설정에 따른 최종 노하우 딕셔너리를 반환합니다."""
    
    expert_confidence = st.session_state.get('Expert_Confidence_slider', 75)
    
    # 사출 속도 노하우
    if st.session_state.get('V_Inj_Intent_active', False): 
        v_inj_intent = st.session_state.get('V_Inj_Intent_select', 'Keep_Constant')
    else:
        v_inj_intent = 'Keep_Constant' 
        
    if st.session_state.get('V_Inj_Delta_active', False): 
        v_inj_delta = st.session_state.get('V_Inj_Delta_slider', 0.0)
    else:
        v_inj_delta = 0.0 

    # 금형 온도 노하우
    if st.session_state.get('T_Mold_Intent_active', False): 
        t_mold_intent = st.session_state.get('T_Mold_Intent_select', 'Keep_Constant')
    else:
        t_mold_intent = 'Keep_Constant' 
        
    if st.session_state.get('T_Mold_Delta_active', False): 
        t_mold_delta = st.session_state.get('T_Mold_Delta_slider', 0.0)
    else:
        t_mold_delta = 0.0 
        
    return {
        'Expert_Confidence': expert_confidence,
        'V_Inj_Intent': v_inj_intent, 
        'T_Mold_Intent': t_mold_intent,
        'V_Inj_Delta': v_inj_delta,
        'T_Mold_Delta': t_mold_delta
    }


def predict_weld_line_risk(model, scaler, feature_names, input_data, knowhow_influence_factor):
    """Weld Line 불량 위험 확률을 예측합니다."""
    
    df_input = pd.DataFrame([input_data])
    scaler_delta_v = st.session_state['scaler_delta_v']
    scaler_delta_t = st.session_state['scaler_delta_t']
    
    # Delta 값 스케일링
    v_inj_delta_scaled = scaler_delta_v.transform(np.array(input_data['V_Inj_Delta']).reshape(-1, 1))[0][0]
    t_mold_delta_scaled = scaler_delta_t.transform(np.array(input_data['T_Mold_Delta']).reshape(-1, 1))[0][0]

    df_input['V_Inj_Delta_Scaled'] = v_inj_delta_scaled
    df_input['T_Mold_Delta_Scaled'] = t_mold_delta_scaled
    
    df_input = pd.get_dummies(df_input, columns=['V_Inj_Intent', 'T_Mold_Intent'], drop_first=True)
    
    # 피처 정렬
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[feature_names]
    numerical_features = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos', 'T_Weld', 't_Fill']
    df_input[numerical_features] = scaler.transform(df_input[numerical_features])

    # 선형 예측
    coef_dict = dict(zip(feature_names, model.coef_[0]))
    input_vector = df_input.iloc[0]

    process_linear_term = 0
    knowhow_linear_term = 0
    
    for feature_name, coef_value in coef_dict.items():
        input_value = input_vector[feature_name]
        if feature_name in KNOWHOW_FEATURES:
            knowhow_linear_term += coef_value * input_value
        else:
            process_linear_term += coef_value * input_value
            
    adjusted_linear_term = model.intercept_[0] + process_linear_term + (knowhow_influence_factor * knowhow_linear_term)
    
    # 위험 확률 계산 (로지스틱 함수)
    risk_prob = 1 / (1 + np.exp(-adjusted_linear_term)) 
    prediction = 1 if risk_prob > 0.5 else 0

    return risk_prob, prediction

def find_optimal_conditions(model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, initial_guess):
    """최적 공정 조건을 찾습니다."""
    
    opt_var_names = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
    bounds = [
        (230, 260),  # T_Melt
        (50, 110),   # V_Inj
        (60, 100),   # P_Pack
        (50, 90),    # T_Mold
        (15.00, 25.00), # Meter
        (8.00, 12.00)    # VP_Switch_Pos
    ]
    
    def objective_function(X_opt, model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor):
        
        T_Melt, V_Inj, P_Pack, T_Mold, Meter, VP_Switch_Pos = X_opt
        
        T_Weld = T_Melt * 0.8 + T_Mold * 0.2 + V_Inj * 0.1
        t_Fill = 3.0 - 0.015 * V_Inj
        
        input_data = {
            'T_Melt': T_Melt, 'V_Inj': V_Inj, 'P_Pack': P_Pack, 'T_Mold': T_Mold,
            'Meter': Meter, 'VP_Switch_Pos': VP_Switch_Pos, 'T_Weld': T_Weld, 't_Fill': t_Fill,
            'Expert_Confidence': knowhow_inputs['Expert_Confidence'],
            'V_Inj_Intent': knowhow_inputs['V_Inj_Intent'], 
            'T_Mold_Intent': knowhow_inputs['T_Mold_Intent'],
            'V_Inj_Delta': knowhow_inputs['V_Inj_Delta'],
            'T_Mold_Delta': knowhow_inputs['T_Mold_Delta']
        }
        
        df_input = pd.DataFrame([input_data])
        
        scaler_delta_v = st.session_state['scaler_delta_v']
        scaler_delta_t = st.session_state['scaler_delta_t']
        
        v_inj_delta_scaled = scaler_delta_v.transform(np.array(input_data['V_Inj_Delta']).reshape(-1, 1))[0][0]
        t_mold_delta_scaled = scaler_delta_t.transform(np.array(input_data['T_Mold_Delta']).reshape(-1, 1))[0][0]
        
        df_input['V_Inj_Delta_Scaled'] = v_inj_delta_scaled
        df_input['T_Mold_Delta_Scaled'] = t_mold_delta_scaled

        df_input = pd.get_dummies(df_input, columns=['V_Inj_Intent', 'T_Mold_Intent'], drop_first=True)

        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_names]
        numerical_features = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos', 'T_Weld', 't_Fill']
        df_input[numerical_features] = scaler.transform(df_input[numerical_features])
        
        coef_dict = dict(zip(feature_names, model.coef_[0]))
        input_vector = df_input.iloc[0]

        process_linear_term = 0
        knowhow_linear_term = 0
        
        for feature_name, coef_value in coef_dict.items():
            input_value = input_vector[feature_name]
            if feature_name in KNOWHOW_FEATURES:
                knowhow_linear_term += coef_value * input_value
            else:
                process_linear_term += coef_value * input_value
                
        adjusted_linear_term = model.intercept_[0] + process_linear_term + (knowhow_influence_factor * knowhow_linear_term)
        
        risk_prob = 1 / (1 + np.exp(-adjusted_linear_term)) 
        
        # ⭐️ 사출 속도 방향 페널티 ⭐️
        V_Inj_current = st.session_state.get('V_Inj_current_for_penalty', DEFAULT_INITIAL_VALS['V_Inj']) 
        V_Inj_delta_input = knowhow_inputs['V_Inj_Delta']
        
        penalty_term = 0
        penalty_strength = 0.005 
        
        # V_Inj_Delta가 양수(속도 증가 의도)인데 최적화된 V_Inj가 현재보다 낮을 경우 페널티
        if V_Inj_delta_input > 0.5 and V_Inj < V_Inj_current:
            penalty_term += (V_Inj_current - V_Inj) * penalty_strength
                
        # V_Inj_Delta가 음수(속도 감소 의도)인데 최적화된 V_Inj가 현재보다 높을 경우 페널티
        elif V_Inj_delta_input < -0.5 and V_Inj > V_Inj_current:
            penalty_term += (V_Inj - V_Inj_current) * penalty_strength

        # 위험 확률을 최소화하는 것이 목적이므로, 목적 함수(Objective Function)로 반환
        return risk_prob + penalty_term

    result = minimize(
        objective_function, 
        initial_guess, 
        args=(model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor),
        method='SLSQP',
        bounds=bounds
    )
    
    optimal_conditions = dict(zip(opt_var_names, result.x))
    optimal_risk = result.fun * 100
    
    return optimal_conditions, optimal_risk, result.success

def run_sensitivity_analysis(model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, current_input):
    """주요 세 변수에 대한 민감도 분석을 수행합니다."""
    
    analysis_results = {}
    variables_to_sweep = {
        'T_Melt': {'min': 230, 'max': 260, 'steps': 20, 'unit': '°C'},
        'V_Inj': {'min': 50, 'max': 110, 'steps': 20, 'unit': 'mm/s'},
        'T_Mold': {'min': 50, 'max': 90, 'steps': 20, 'unit': '°C'}
    }

    # 현재 입력 데이터를 복사 (다른 변수는 고정)
    base_input = current_input.copy() 

    for var_name, config in variables_to_sweep.items():
        sweep_values = np.linspace(config['min'], config['max'], config['steps'])
        risks = []
        
        for val in sweep_values:
            # 1. 변수 값 변경
            temp_input = base_input.copy()
            temp_input[var_name] = val
            
            # 2. 파생 변수 업데이트 (T_Weld, t_Fill)
            temp_input['T_Weld'] = temp_input['T_Melt'] * 0.8 + temp_input['T_Mold'] * 0.2 + temp_input['V_Inj'] * 0.1
            temp_input['t_Fill'] = 3.0 - 0.015 * temp_input['V_Inj']
            
            # 3. 위험도 예측
            risk_prob, _ = predict_weld_line_risk(
                model, scaler, feature_names, temp_input, knowhow_influence_factor
            )
            risks.append(risk_prob * 100) # 퍼센트로 저장
            
        analysis_results[var_name] = pd.DataFrame({
            var_name: sweep_values, 
            'Weld_Risk (%)': risks
        })

    return analysis_results

# -----------------------------------------------------------------------------
# 2. STREAMLIT UI 및 세션 관리
# -----------------------------------------------------------------------------

def set_initial_vals(df_init):
    """업로드된 초기 조건 파일에서 값을 로드하거나 기본값을 사용합니다."""
    if df_init is not None and not df_init.empty:
        df_init = df_init.iloc[0]
        st.session_state['initial_values'] = {
            'T_Melt': float(df_init.get('T_Melt', DEFAULT_INITIAL_VALS['T_Melt'])),
            'V_Inj': float(df_init.get('V_Inj', DEFAULT_INITIAL_VALS['V_Inj'])),
            'P_Pack': float(df_init.get('P_Pack', DEFAULT_INITIAL_VALS['P_Pack'])),
            'T_Mold': float(df_init.get('T_Mold', DEFAULT_INITIAL_VALS['T_Mold'])),
            'Meter': float(df_init.get('Meter', DEFAULT_INITIAL_VALS['Meter'])),
            'VP_Switch_Pos': float(df_init.get('VP_Switch_Pos', DEFAULT_INITIAL_VALS['VP_Switch_Pos']))
        }
    else:
        st.session_state['initial_values'] = DEFAULT_INITIAL_VALS.copy()

def load_and_train_model():
    """파일을 로드하고 모델을 학습합니다. (최소 2번 또는 3번 파일 중 하나는 필수)"""
    
    uploaded_init = st.session_state.get('init_uploader')
    uploaded_virtual = st.session_state.get('virtual_uploader')
    uploaded_real = st.session_state.get('real_uploader') 

    # ⭐️ 수정된 필수 파일 확인 로직: 2번 또는 3번 파일 중 하나라도 있으면 진행 ⭐️
    if uploaded_real is None and uploaded_virtual is None:
        st.error("🚨 필수 파일 경고: AI 모델 학습을 위해 최소한 **2번 또는 3번 파일** 중 하나를 업로드해야 합니다.")
        st.session_state['model_loaded'] = False
        return

    with st.spinner('데이터 처리 및 AI 모델 학습 중...'):
        
        # 1. 파일 로드 및 기본값 처리
        df_init = load_df_from_uploader(uploaded_init)
        df_real = load_df_from_uploader(uploaded_real)
        df_virtual = load_df_from_uploader(uploaded_virtual)
        
        # NoneType 처리
        if df_real is None: df_real = pd.DataFrame()
        if df_virtual is None: df_virtual = pd.DataFrame()
        
        # 2. 초기값 설정
        set_initial_vals(df_init)
        
        # 3. 데이터 병합 및 처리
        st.session_state['df_weld'] = process_weld_data(df_virtual, df_real)
        st.session_state['virtual_data_size'] = len(df_virtual)
        st.session_state['real_data_size'] = len(df_real)
        
        # 4. 학습 가능성 확인
        if len(st.session_state['df_weld']) < 10: 
            st.error(f"🚨 학습 데이터가 너무 작습니다. 현재 데이터 크기: {len(st.session_state['df_weld'])}개. 최소 10개 이상을 권장합니다.")
            st.session_state['model_loaded'] = False
            return
        
        # 5. 모델 학습
        try:
            st.cache_resource.clear() 
            st.session_state['model'], st.session_state['scaler'], st.session_state['feature_names'], st.session_state['accuracy'], st.session_state['data_size'], st.session_state['defect_rate'] = train_model(st.session_state['df_weld'])
            st.session_state['model_loaded'] = True
            st.session_state['executed'] = False 
            st.session_state['optimal_executed'] = False 
            st.success("✅ AI 모델 학습 및 로드 완료! 초기 조건이 UI에 반영되었습니다.")
        except ValueError as e:
             st.session_state['model_loaded'] = False
             st.error(f"모델 학습 실패: {e}")


def run_optimization():
    if not st.session_state.get('model_loaded', False):
        st.error("AI 모델이 로드되지 않았습니다. 먼저 모델을 학습시켜 주세요.")
        st.session_state['optimal_executed'] = False
        return

    try:
        knowhow_inputs = get_knowhow_inputs()
        knowhow_influence_factor = st.session_state['knowhow_factor']
        
        # 페널티 로직을 위해 현재 V_Inj 슬라이더 값을 저장
        st.session_state['V_Inj_current_for_penalty'] = st.session_state['V_Inj_slider']
        
        # 1. 초기 추측값 A: 현재 UI 공정 조건 사용
        initial_guess_A = [
            st.session_state['T_Melt_slider'],
            st.session_state['V_Inj_slider'],
            st.session_state['P_Pack_slider'],
            st.session_state['T_Mold_slider'],
            st.session_state['Meter_slider'],
            st.session_state['VP_Switch_Pos_slider']
        ]

    except KeyError as e:
        st.error(f"UI 입력값을 가져오는 중 오류 발생: {e}.")
        st.session_state['optimal_executed'] = False
        return

    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    
    # 2. 초기 추측값 B: 탐색 범위 중앙 사용
    initial_guess_B = [245.0, 80.0, 80.0, 70.0, 20.00, 10.00] 
    
    # 3. 초기 추측값 C: T_Mold 최소값 설정
    initial_guess_C = [245.0, 80.0, 80.0, 50.0, 20.00, 10.00] 
    
    
    best_risk = 101.0 
    best_conditions = None
    best_success = False
    
    with st.spinner('✨ 최적 조건 탐색 중... (3가지 초기 지점 시도)'):
        
        # 1. 시도 A
        opt_cond_A, opt_risk_A, success_A = find_optimal_conditions(
            model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, initial_guess_A
        )
        if success_A and opt_risk_A < best_risk:
            best_risk = opt_risk_A
            best_conditions = opt_cond_A
            best_success = True
            
        # 2. 시도 B
        opt_cond_B, opt_risk_B, success_B = find_optimal_conditions(
            model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, initial_guess_B
        )
        if success_B and opt_risk_B < best_risk:
            best_risk = opt_risk_B
            best_conditions = opt_cond_B
            best_success = True

        # 3. 시도 C
        opt_cond_C, opt_risk_C, success_C = find_optimal_conditions(
            model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, initial_guess_C
        )
        if success_C and opt_risk_C < best_risk:
            best_risk = opt_risk_C
            best_conditions = opt_cond_C
            best_success = True

    # 4. 최적 결과 저장
    if best_success:
        st.session_state['optimal_conditions'] = best_conditions
        st.session_state['optimal_risk'] = best_risk
        st.session_state['optimal_executed'] = True
        st.session_state['optimal_success'] = True
    else:
        st.session_state['optimal_executed'] = True
        st.session_state['optimal_success'] = False


# -----------------------------------------------------------------------------
# 3. STREAMLIT UI 레이아웃
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Weld Line AI 진단 시스템")
st.header("Weld Line AI 진단 시스템", divider='rainbow')


# --- 사이드바 ---
with st.sidebar:
    st.title("📂 데이터 파일 업로드 및 모델 학습")
    # 메시지 수정: 3번 파일이 필수가 아님을 반영
    st.info("AI 모델 학습을 위해 **2번 또는 3번 파일 중 하나**는 최소한 업로드해야 합니다.")
    
    # 파일 업로더
    st.file_uploader("1. UI 초기 조건 (initial_condition.xlsx) [선택]", type=['xlsx'], key='init_uploader')
    # 2번 파일만으로 학습 가능
    st.file_uploader("2. 가상 학습 데이터 (test_condition.xlsx) [학습 데이터]", type=['xlsx'], key='virtual_uploader')
    # 3번 파일은 선택 사항으로 변경 (단독으로 학습 데이터로 사용 가능)
    st.file_uploader("3. 시뮬레이션 학습 데이터 (moldflow_condition.xlsx) [학습 데이터]", type=['xlsx'], key='real_uploader')
    
    # 로드 및 학습 버튼
    st.button(
        "🚀 파일 로드 및 AI 모델 학습 시작", 
        on_click=load_and_train_model, 
        use_container_width=True, 
        type='primary'
    )
    
    st.markdown("---")
    
    st.subheader("시스템 상태")
    if st.session_state.get('model_loaded', False):
        st.markdown(f"""
        --- 모델: Weld Line 불량 예측 모델 ---
        **정확도 (Accuracy):** {st.session_state['accuracy']:.4f}
        **전체 데이터 수:** {st.session_state['data_size']}개, **불량률:** {st.session_state['defect_rate']*100:.1f}%
        **시뮬레이션 데이터 (3번):** {st.session_state.get('real_data_size', 'N/A')}개
        **가상 데이터 (2번):** {st.session_state.get('virtual_data_size', 'N/A')}개
        """)
    else:
        st.warning("파일을 업로드하고 'AI 모델 학습 시작' 버튼을 눌러주세요.")


if not st.session_state.get('model_loaded', False):
    st.error("데이터 파일이 업로드되고 AI 모델이 학습될 때까지 시스템을 사용할 수 없습니다. **최소한 2번 또는 3번 파일 중 하나**를 업로드하고 학습을 시작해주세요.")
    st.stop() 

if 'initial_values' not in st.session_state:
    set_initial_vals(None) 
    
initial_vals = st.session_state['initial_values'] 

# 탭
tab1, tab2, tab3 = st.tabs(["1. Weld Line 공정 진단 (핵심)", "2. 모델 및 데이터 검토", "3. 민감도 분석"])

with tab1:
    st.subheader("A. 현재 공정 조건 입력")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    T_Melt = col1.slider("용융 온도 (T_Melt, °C)", 230, 260, int(initial_vals['T_Melt']), 1, key='T_Melt_slider')
    V_Inj = col2.slider("사출 속도 (V_Inj, mm/s)", 50, 110, int(initial_vals['V_Inj']), 1, key='V_Inj_slider')
    P_Pack = col3.slider("보압 (P_Pack, MPa)", 60, 100, int(initial_vals['P_Pack']), 1, key='P_Pack_slider')

    Meter = col4.slider("계량 거리 (Meter, mm)", 15.00, 25.00, float(initial_vals['Meter']), 0.01, key='Meter_slider')
    VP_Switch_Pos = col5.slider("VP 절환 위치 (VP_Switch_Pos, mm)", 8.00, 12.00, float(initial_vals['VP_Switch_Pos']), 0.01, key='VP_Switch_Pos_slider')
    T_Mold = col6.slider("금형 온도 (T_Mold, °C)", 50, 90, int(initial_vals['T_Mold']), 1, key='T_Mold_slider')

    st.markdown("---")
    
    st.subheader("B. 전문가의 정성/정량적 노하우 입력")
    
    # 1. 전문가 확신도
    st.markdown("##### 1. 전문가 확신도")
    Expert_Confidence = st.slider(
        '전문가 확신 수준', 50, 100, 75, 1,
        key='Expert_Confidence_slider'
    )
    st.markdown("---")
    
    # 2. 사출 속도 노하우
    st.markdown("##### 2. 사출 속도 노하우 ($\text{V\_Inj}$)")
    col_v_intent_check, col_v_intent, col_v_delta_check, col_v_delta = st.columns([1, 2, 1, 2])
    
    # 사출 속도 - 정성적 (의도)
    col_v_intent_check.checkbox("정성적 노하우 적용", value=False, key='V_Inj_Intent_active')
    V_Inj_Intent = col_v_intent.selectbox(
        'V_Inj 조절 의도',
        ('Keep_Constant', 'High_Increase', 'Low_Decrease'),
        index=0,
        key='V_Inj_Intent_select',
        disabled=not st.session_state.get('V_Inj_Intent_active', False)
    )

    # 사출 속도 - 정량적 (변화량)
    col_v_delta_check.checkbox("정량적 노하우 적용", value=False, key='V_Inj_Delta_active')
    V_Inj_Delta = col_v_delta.slider(
        'V_Inj 노하우 변화량 ($\Delta V_{Inj}$, mm/s)',
        -15.0, 15.0, 0.0, 0.5,
        key='V_Inj_Delta_slider',
        disabled=not st.session_state.get('V_Inj_Delta_active', False)
    )
    st.markdown("---")

    # 3. 금형 온도 노하우
    st.markdown("##### 3. 금형 온도 노하우 ($\text{T\_Mold}$)")
    col_t_intent_check, col_t_intent, col_t_delta_check, col_t_delta = st.columns([1, 2, 1, 2])
    
    # 금형 온도 - 정성적 (의도)
    col_t_intent_check.checkbox("정성적 노하우 적용", value=False, key='T_Mold_Intent_active')
    T_Mold_Intent = col_t_intent.selectbox(
        'T_Mold 조절 의도',
        ('Keep_Constant', 'High_Increase', 'Low_Decrease'),
        index=0,
        key='T_Mold_Intent_select',
        disabled=not st.session_state.get('T_Mold_Intent_active', False)
    )
    
    # 금형 온도 - 정량적 (변화량)
    col_t_delta_check.checkbox("정량적 노하우 적용", value=False, key='T_Mold_Delta_active')
    T_Mold_Delta = col_t_delta.slider(
        'T_Mold 노하우 변화량 ($\Delta T_{Mold}$, °C)',
        -10.0, 10.0, 0.0, 0.5,
        key='T_Mold_Delta_slider',
        disabled=not st.session_state.get('T_Mold_Delta_active', False)
    )

    st.markdown("---")

    st.subheader("C. 진단 실행 및 결과")
    
    knowhow_influence_factor = st.slider(
        "노하우 영향 계수",
        0.0, 5.0, 1.0, 0.1,
        key="knowhow_factor",
        help="0.0: 노하우 변수 영향 제거, 1.0: 기본 학습된 영향, 5.0: 5배 증폭된 영향"
    )
    st.markdown("---")


    T_Weld = T_Melt * 0.8 + T_Mold * 0.2 + V_Inj * 0.1
    t_Fill = 3.0 - 0.015 * V_Inj
    
    current_knowhow_inputs = get_knowhow_inputs()

    input_data = {
        'T_Melt': T_Melt, 'V_Inj': V_Inj, 'P_Pack': P_Pack, 'T_Mold': T_Mold,
        'Meter': Meter, 'VP_Switch_Pos': VP_Switch_Pos, 'T_Weld': T_Weld, 't_Fill': t_Fill,
        'Expert_Confidence': current_knowhow_inputs['Expert_Confidence'],
        'V_Inj_Intent': current_knowhow_inputs['V_Inj_Intent'], 
        'T_Mold_Intent': current_knowhow_inputs['T_Mold_Intent'],
        'V_Inj_Delta': current_knowhow_inputs['V_Inj_Delta'],
        'T_Mold_Delta': current_knowhow_inputs['T_Mold_Delta']
    }
    
    col_diag, col_opt_placeholder = st.columns(2)
    
    # 통합 진단 버튼
    if col_diag.button("🔴 Weld Line 통합 진단 실행", use_container_width=True, type='primary'):
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        
        risk_prob, prediction = predict_weld_line_risk(
            model, scaler, feature_names, input_data, knowhow_influence_factor 
        )
        st.session_state['risk_prob'] = risk_prob
        st.session_state['prediction'] = prediction
        st.session_state['executed'] = True
        
        # 민감도 분석을 위해 현재 입력 데이터 저장
        st.session_state['current_input_for_sensitivity'] = input_data
    
    
    st.subheader("✨ 최적 공정 조건 솔루션")
    
    # 🚨 노하우 집중 모드 제거 
    st.button(
        "✨ 최적 공정 조건 제시", 
        use_container_width=True, 
        type='secondary',
        on_click=run_optimization,
        help="현재 설정된 노하우와 노하우 영향 계수를 반영하여 Weld Line 불량 위험을 최소화하는 최적 공정 조건을 탐색합니다."
    )

    
    st.subheader("💡 진단 결과")
    if st.session_state.get('executed', False):
        risk_prob = st.session_state['risk_prob']
        
        if risk_prob > 0.5:
            st.error(f"🔴 AI 모델 경고! Weld Line 불량 위험 확률: {risk_prob*100:.1f}% (노하우 계수: {knowhow_influence_factor:.1f})", icon="🚨")
            st.warning("현재 공정 조건과 전문가 노하우는 불량 위험을 높이고 있습니다. 사출 속도나 금형 온도를 안전 범위로 조정해 보세요.")
            
        else:
            st.success(f"✅ 현재 조건 양호. (AI 예측 위험도: {risk_prob*100:.1f}%, 노하우 계수: {knowhow_influence_factor:.1f})", icon="👍")
            st.info("현재 공정은 안정적입니다. 노하우 영향 계수를 조절하여 AI 예측의 안정성을 확인해 보세요.")

    else:
        st.info("현재 공정 진단이 실행되지 않았습니다. '🔴 Weld Line 통합 진단 실행' 버튼을 눌러주세요.")

    
    st.markdown("---")
    
    
    if st.session_state.get('optimal_executed', False):
        if st.session_state['optimal_success']:
            opt_cond = st.session_state['optimal_conditions']
            opt_risk = st.session_state['optimal_risk']
            knowhow_factor_used = st.session_state['knowhow_factor']
            
            st.success(f"계산 완료! **최소 불량 위험 확률: {opt_risk:.2f}%**")
            
            opt_df = pd.DataFrame({
                '변수': ['용융 온도 (T_Melt)', '사출 속도 (V_Inj)', '보압 (P_Pack)', '금형 온도 (T_Mold)', '계량 거리 (Meter)', 'VP 절환 위치 (VP_Switch_Pos)'],
                '최적 값': [
                    f"{opt_cond['T_Melt']:.0f} °C", 
                    f"{opt_cond['V_Inj']:.0f} mm/s", 
                    f"{opt_cond['P_Pack']:.0f} MPa", 
                    f"{opt_cond['T_Mold']:.0f} °C", 
                    f"{opt_cond['Meter']:.2f} mm", 
                    f"{opt_cond['VP_Switch_Pos']:.2f} mm"
                ]
            })
            st.table(opt_df)
            st.caption(f"이 최적 조건은 현재 설정된 **정성/정량적 노하우**와 **노하우 영향 계수 ({knowhow_factor_used:.1f})**를 반영한 결과입니다.")
            
        else:
            st.warning("최적화 계산에 실패했거나, 세 가지 초기 추측 모두 현재 조건보다 낮은 위험도를 찾지 못했습니다. 입력 조건을 다시 확인해 주세요.")
    else:
        st.info("'✨ 최적 공정 조건 제시' 버튼을 눌러 Weld Line 불량 위험을 최소화하는 최적 공정 조건을 찾아보세요.")


with tab2:
    st.header("상세 모델 학습 결과 및 데이터 미리보기")
    
    st.subheader("AI 모델 학습 요약")
    st.markdown("AI 모델은 **로지스틱 회귀 (Logistic Regression)** 모델을 사용하여 학습되었습니다.")
    st.metric(label="AI 모델 정확도 (테스트 세트)", value=f"{st.session_state['accuracy'] * 100:.2f}%")
    st.metric(label="통합 데이터 총 크기", value=f"{st.session_state['data_size']}개")
    st.metric(label="통합 데이터 세트 불량률", value=f"{st.session_state['defect_rate'] * 100:.1f}%")
    
    st.markdown("---")
    
    st.subheader("모델 계수 시각화")
    if 'model' in st.session_state and 'feature_names' in st.session_state:
        model = st.session_state['model']
        feature_names = st.session_state['feature_names']
        
        coef_df = pd.DataFrame({
            '특징 (Feature)': feature_names,
            '계수 (Coefficient)': model.coef_[0]
        })
        
        coef_df['유형'] = '공정'
        coef_df.loc[coef_df['특징 (Feature)'].isin(KNOWHOW_FEATURES), '유형'] = '노하우'
        
        st.dataframe(coef_df.sort_values(by='계수 (Coefficient)', ascending=False), height=400)
        st.caption("계수의 절댓값이 클수록 불량 위험 확률에 미치는 영향이 큽니다. 양수(+)는 위험 증가, 음수(-)는 위험 감소를 의미합니다.")
        
        st.markdown("**사출 속도 관련 계수 (검토 필요):**")
        v_inj_coefs = coef_df[coef_df['특징 (Feature)'].str.contains('V_Inj') | coef_df['특징 (Feature)'].str.contains('t_Fill')]
        st.dataframe(v_inj_coefs)

        st.warning("""
        **[사출 속도 역추세 진단]**
        노하우와 최적화 방향이 상충하는 문제에 대비하여 **목적 함수에 페널티가 추가**되었습니다.
        계수의 부호가 직관과 상충된다면, 이는 데이터 내에서 모델이 학습한 추세가 노하우와 충돌하기 때문입니다.
        """)
        
    st.markdown("---")
    st.subheader("통합 학습 데이터 세트 (시뮬레이션 + 가상)")
    if 'df_weld' in st.session_state:
        st.caption("업로드된 시뮬레이션 및 가상 데이터를 병합하여 학습에 사용된 데이터입니다.")
        st.dataframe(st.session_state['df_weld'].head(20))
    else:
        st.info("학습 데이터가 로드되지 않았습니다.")

with tab3:
    st.header("민감도 분석 📊")
    st.info("현재 설정된 공정 조건 및 전문가 노하우를 기준으로, 주요 변수 변화에 따른 Weld Line 불량 위험 확률 변화를 분석합니다.")
    
    if st.session_state.get('model_loaded', False):
        
        if 'current_input_for_sensitivity' not in st.session_state:
            st.warning("⚠️ 먼저 **'1. Weld Line 공정 진단 (핵심)'** 탭에서 공정 조건을 설정하고 **'🔴 Weld Line 통합 진단 실행'** 버튼을 눌러주세요. 현재 UI 값으로 임시 분석을 시작합니다.")
            
            current_knowhow_inputs = get_knowhow_inputs()
            initial_vals_copy = st.session_state.get('initial_values', DEFAULT_INITIAL_VALS)
            
            base_input = {
                'T_Melt': st.session_state.get('T_Melt_slider', initial_vals_copy['T_Melt']),
                'V_Inj': st.session_state.get('V_Inj_slider', initial_vals_copy['V_Inj']),
                'P_Pack': st.session_state.get('P_Pack_slider', initial_vals_copy['P_Pack']),
                'T_Mold': st.session_state.get('T_Mold_slider', initial_vals_copy['T_Mold']),
                'Meter': st.session_state.get('Meter_slider', initial_vals_copy['Meter']),
                'VP_Switch_Pos': st.session_state.get('VP_Switch_Pos_slider', initial_vals_copy['VP_Switch_Pos']),
                'Expert_Confidence': current_knowhow_inputs['Expert_Confidence'],
                'V_Inj_Intent': current_knowhow_inputs['V_Inj_Intent'], 
                'T_Mold_Intent': current_knowhow_inputs['T_Mold_Intent'],
                'V_Inj_Delta': current_knowhow_inputs['V_Inj_Delta'],
                'T_Mold_Delta': current_knowhow_inputs['T_Mold_Delta']
            }
        else:
            base_input = st.session_state['current_input_for_sensitivity'] 

        # 분석 실행
        with st.spinner('민감도 분석 시뮬레이션 중...'):
            # base_input은 이미 knowhow_inputs를 포함하고 있으므로, 4번째 인자로 base_input을 재사용합니다.
            sensitivity_data = run_sensitivity_analysis(
                st.session_state['model'], 
                st.session_state['scaler'], 
                st.session_state['feature_names'], 
                base_input, 
                st.session_state['knowhow_factor'], 
                base_input
            )
        
        st.success("민감도 분석 완료! 현재 공정 변수들의 위험 변화 곡선을 확인하세요.")
        
        st.markdown("---")
        st.subheader("용융 온도 ($T_{Melt}$) 민감도 분석")
        st.line_chart(sensitivity_data['T_Melt'], x='T_Melt', y='Weld_Risk (%)')
        st.caption(f"다른 변수는 고정하고 $T_{Melt}$를 변화시킬 때의 위험 확률 변화. **현재 값: {base_input['T_Melt']:.1f}°C**")

        st.subheader("사출 속도 ($V_{Inj}$) 민감도 분석")
        st.line_chart(sensitivity_data['V_Inj'], x='V_Inj', y='Weld_Risk (%)')
        st.caption(f"다른 변수는 고정하고 $V_{Inj}$를 변화시킬 때의 위험 확률 변화. **현재 값: {base_input['V_Inj']:.1f} mm/s**")
        
        st.subheader("금형 온도 ($T_{Mold}$) 민감도 분석")
        st.line_chart(sensitivity_data['T_Mold'], x='T_Mold', y='Weld_Risk (%)')
        st.caption(f"다른 변수는 고정하고 $T_{Mold}$를 변화시킬 때의 위험 확률 변화. **현재 값: {base_input['T_Mold']:.1f}°C**")


    else:
        st.error("AI 모델이 로드되지 않아 민감도 분석을 수행할 수 없습니다.")
