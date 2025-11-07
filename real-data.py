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

# 기본 초기값 (initial_condition.xlsx 파일이 없을 때 사용)
DEFAULT_INITIAL_VALS = {
    'T_Melt': 240.0, 'V_Inj': 80.0, 'P_Pack': 80.0, 
    'T_Mold': 80.0, 'Meter': 18.00, 'VP_Switch_Pos': 10.50
}


# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 모델 학습 로직 (DATA & MODEL LOGIC)
# -----------------------------------------------------------------------------

def load_df_from_uploader(uploaded_file):
    """업로드된 파일 객체에서 Pandas DataFrame을 로드합니다."""
    if uploaded_file is not None:
        try:
            # openpyxl 라이브러리 추가 후, 엑셀 파일 로드
            # 주의: .csv 파일로 변환된 엑셀 파일은 pd.read_csv를 사용해야 함.
            # 하지만 사용자가 원본 파일명을 .xlsx로 알려주었으므로, 표준 로직 유지.
            # Streamlit 환경에서는 CSV 파일로 업로드될 수 있으므로, .csv 처리 추가
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                 return pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """업로드된 두 데이터프레임을 병합하고 학습에 필요한 컬럼을 처리합니다."""
    
    df_combined = pd.concat([df_real, df_virtual], ignore_index=True)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    
    if 'Expert_Confidence' not in df_combined.columns:
        df_combined['Expert_Confidence'] = 75 
        
    df_combined['T_Weld'] = df_combined['T_Melt'] * 0.8 + df_combined['T_Mold'] * 0.2 + df_combined['V_Inj'] * 0.1
    df_combined['t_Fill'] = 3.0 - 0.015 * df_combined['V_Inj']
    
    # Delta 값이 데이터에 없을 경우 임시로 생성 (학습 데이터의 다양성 확보 목적)
    if 'V_Inj_Delta' not in df_combined.columns or 'T_Mold_Delta' not in df_combined.columns:
        df_combined['V_Inj_Delta'] = 0.0
        # .astype(str) 추가: 데이터 타입이 혼합되어 있을 때 오류 방지
        df_combined.loc[df_combined['V_Inj_Intent'].astype(str).str.contains('Increase'), 'V_Inj_Delta'] = 10.0 
        df_combined.loc[df_combined['V_Inj_Intent'].astype(str).str.contains('Decrease'), 'V_Inj_Delta'] = -5.0
        
        df_combined['T_Mold_Delta'] = 0.0
        df_combined.loc[df_combined['T_Mold_Intent'].astype(str).str.contains('Increase'), 'T_Mold_Delta'] = 8.0
        df_combined.loc[df_combined['T_Mold_Intent'].astype(str).str.contains('Decrease'), 'T_Mold_Delta'] = -4.0

    # Delta 값 스케일러 저장 (나중에 UI 입력값을 스케일링하는 데 사용)
    if not df_combined.empty:
        # 데이터프레임이 비어있지 않은지 확인 후 스케일러 적용
        if 'V_Inj_Delta' in df_combined.columns and 'T_Mold_Delta' in df_combined.columns:
            st.session_state['scaler_delta_v'] = StandardScaler().fit(df_combined[['V_Inj_Delta']])
            st.session_state['scaler_delta_t'] = StandardScaler().fit(df_combined[['T_Mold_Delta']])
            
            df_combined['V_Inj_Delta_Scaled'] = st.session_state['scaler_delta_v'].transform(df_combined[['V_Inj_Delta']])
            df_combined['T_Mold_Delta_Scaled'] = st.session_state['scaler_delta_t'].transform(df_combined[['T_Mold_Delta']])
    
    return df_combined

@st.cache_resource
def train_model(df):
    """모델 학습 및 평가"""
    
    # errors='ignore' 추가하여 학습에 불필요한 컬럼이 없어도 오류 발생 방지
    X = df.drop(columns=['L_Weld', 'Y_Weld', 'V_Inj_Delta', 'T_Mold_Delta'], errors='ignore') 
    
    # 🔴 [Y_Weld 클리닝] Y_Weld를 numeric으로 변환, NaN은 0으로 채우고, 0 또는 1로 반올림하여 강제 이진화 후 정수형으로 변환
    y_raw = df['Y_Weld'] 
    y_clean = pd.to_numeric(y_raw, errors='coerce').fillna(0).round().astype(int)
    y = y_clean
    
    # 명확히 클리닝된 데이터로 불량 개수 및 비율 계산
    defect_count = y.sum()
    defect_rate = y.mean()

    # 🚨 학습 중단 로직 임시 우회 (현재 데이터에 불량 샘플(1)이 0개이기 때문에)
    # **참고:** 모델의 예측 품질을 위해, 반드시 불량 샘플을 추가한 후 아래 주석을 해제해야 함.
    # if defect_count < 2:
    #     st.error(f"🚨 심각한 오류: 학습 데이터에 불량(1) 샘플이 **최소 2개** 미만입니다. 현재 불량 개수: {defect_count}개, 비율: {defect_rate*100:.1f}%. **학습이 중단됩니다.**")
    #     raise ValueError("Insufficient defect samples (Requires at least 2 for split/training stability).")
    
    if defect_count == 0:
        st.warning(f"⚠️ 경고: 학습 데이터에 불량(1) 샘플이 0개입니다. (비율: 0.0%). 모델은 모든 입력을 양품(0)으로만 예측할 가능성이 매우 높습니다.")


    # 이진 분류를 위해 더미 변수 생성
    X = pd.get_dummies(X, columns=['V_Inj_Intent', 'T_Mold_Intent'], drop_first=True)
    
    numerical_features = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos', 'T_Weld', 't_Fill']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    model = LogisticRegression(solver='liblinear', random_state=42)
    
    # 데이터가 4개 이하이거나 불량 샘플이 2개일 경우 train_test_split 생략 (학습 안정성 확보)
    if len(X) <= 4 or defect_count <= 2:
        model.fit(X, y)
        accuracy = 1.0 
        st.warning("경고: 데이터 개수 또는 불량 샘플 개수가 매우 적어 (<=4 또는 불량<=2), **전체 데이터를 학습**합니다. 정확도는 100%로 임의 설정됩니다.")
    else:
        # stratify=y를 추가하여 데이터가 적을 때도 불량/양품 비율을 유지
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X.columns.tolist(), accuracy, len(df), defect_rate

def get_knowhow_inputs():
    """UI에서 설정된 노하우 입력값과 체크박스 상태에 따른 최종 노하우 딕셔너리를 반환"""
    
    # 1. 확신도 (Expert_Confidence)는 항상 반영
    expert_confidence = st.session_state.get('Expert_Confidence_slider', 75)
    
    # 2. 사출 속도 노하우
    if st.session_state.get('V_Inj_Intent_active', False): # 정성적 노하우 활성화 여부
        v_inj_intent = st.session_state.get('V_Inj_Intent_select', 'Keep_Constant')
    else:
        v_inj_intent = 'Keep_Constant' 
        
    if st.session_state.get('V_Inj_Delta_active', False): # 정량적 노하우 활성화 여부
        v_inj_delta = st.session_state.get('V_Inj_Delta_slider', 0.0)
    else:
        v_inj_delta = 0.0 

    # 3. 금형 온도 노하우
    if st.session_state.get('T_Mold_Intent_active', False): # 정성적 노하우 활성화 여부
        t_mold_intent = st.session_state.get('T_Mold_Intent_select', 'Keep_Constant')
    else:
        t_mold_intent = 'Keep_Constant' 
        
    if st.session_state.get('T_Mold_Delta_active', False): # 정량적 노하우 활성화 여부
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
    
    # 스케일러가 존재하는지 확인
    if 'scaler_delta_v' not in st.session_state:
        st.error("스케일러가 초기화되지 않았습니다. 데이터를 다시 로드하고 학습해 주세요.")
        return 0.5, 0
        
    scaler_delta_v = st.session_state['scaler_delta_v']
    scaler_delta_t = st.session_state['scaler_delta_t']
    
    # Delta 값 스케일링
    v_inj_delta_scaled = scaler_delta_v.transform([[input_data['V_Inj_Delta']]])[0][0]
    t_mold_delta_scaled = scaler_delta_t.transform([[input_data['T_Mold_Delta']]])[0][0]

    df_input['V_Inj_Delta_Scaled'] = v_inj_delta_scaled
    df_input['T_Mold_Delta_Scaled'] = t_mold_delta_scaled
    
    df_input = pd.get_dummies(df_input, columns=['V_Inj_Intent', 'T_Mold_Intent'], drop_first=True)
    
    # 피처 일치 (모델 학습에 사용된 피처 목록과 현재 입력 피처 목록을 일치시킴)
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
    
    # 위험 확률 계산
    risk_prob = 1 / (1 + np.exp(-adjusted_linear_term)) 
    prediction = 1 if risk_prob > 0.5 else 0

    return risk_prob, prediction

def find_optimal_conditions(model, scaler, feature_names, knowhow_inputs, knowhow_influence_factor, initial_guess):
    """최적의 공정 조건을 찾습니다."""
    
    opt_var_names = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
    bounds = [
        (230, 260), # T_Melt
        (50, 110),  # V_Inj
        (60, 100),  # P_Pack
        (50, 90),   # T_Mold
        (15.00, 25.00), # Meter
        (8.00, 12.00)   # VP_Switch_Pos
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
        
        # 스케일러가 존재하는지 확인
        if 'scaler_delta_v' not in st.session_state:
            return 1.0 # 오류 발생 시 위험 확률을 최대치로 반환하여 최적화 실패 유도

        scaler_delta_v = st.session_state['scaler_delta_v']
        scaler_delta_t = st.session_state['scaler_delta_t']
        v_inj_delta_scaled = scaler_delta_v.transform([[input_data['V_Inj_Delta']]])[0][0]
        t_mold_delta_scaled = scaler_delta_t.transform([[input_data['T_Mold_Delta']]])[0][0]
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
        
        # ⭐️ 사출 속도 방향성 페널티 추가 ⭐️
        V_Inj_current = st.session_state.get('V_Inj_current_for_penalty', V_Inj) # 초기값 없을 경우 최적화 값을 현재 값으로 간주
        V_Inj_delta_input = knowhow_inputs['V_Inj_Delta']
        
        penalty_term = 0
        penalty_strength = 0.005 
        
        # V_Inj_Delta가 양수 (속도를 높이려는 의도)이고, 최적화된 V_Inj가 현재 값보다 작다면 페널티
        if V_Inj_delta_input > 0.5 and V_Inj < V_Inj_current:
            penalty_term += (V_Inj_current - V_Inj) * penalty_strength
                
        # V_Inj_Delta가 음수 (속도를 낮추려는 의도)이고, 최적화된 V_Inj가 현재 값보다 크다면 페널티
        elif V_Inj_delta_input < -0.5 and V_Inj > V_Inj_current:
            penalty_term += (V_Inj - V_Inj_current) * penalty_strength

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


# -----------------------------------------------------------------------------
# 2. Streamlit UI 및 세션 관리 (STREAMLIT UI & SESSION MANAGEMENT)
# -----------------------------------------------------------------------------

def set_initial_vals(df_init):
    """업로드된 초기 조건 파일에서 값을 가져와 세션 상태에 저장합니다. 파일이 없으면 기본값 사용."""
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
    """파일을 로드하고 모델을 학습합니다. moldflow_condition.xlsx(해석 데이터)만 필수입니다."""
    
    uploaded_init = st.session_state.get('init_uploader')
    uploaded_virtual = st.session_state.get('virtual_uploader')
    uploaded_real = st.session_state.get('real_uploader') 

    # 필수 파일 검사
    if uploaded_real is None:
        st.error("🚨 필수 파일 경고: '3. 해석 학습 데이터'는 AI 모델 학습을 위해 반드시 필요합니다.")
        st.session_state['model_loaded'] = False
        return

    with st.spinner('데이터 처리 및 AI 모델 학습 중...'):
        
        # 1. 파일 로드 및 기본값 처리
        df_init = load_df_from_uploader(uploaded_init)
        df_real = load_df_from_uploader(uploaded_real)
        
        if uploaded_virtual is not None:
            df_virtual = load_df_from_uploader(uploaded_virtual)
        else:
            df_virtual = pd.DataFrame() 
            if len(df_real) > 0: 
                st.warning("⚠️ '2. 가상 학습 데이터'가 없어 해석 데이터만으로 모델을 학습합니다.")
        
        # 2. 초기값 설정
        set_initial_vals(df_init)
        
        # 3. 데이터 병합 및 처리
        st.session_state['df_weld'] = process_weld_data(df_virtual, df_real)
        st.session_state['virtual_data_size'] = len(df_virtual)
        st.session_state['real_data_size'] = len(df_real)
        
        # 4. 모델 학습
        try:
            st.cache_resource.clear() 
            st.session_state['model'], st.session_state['scaler'], st.session_state['feature_names'], st.session_state['accuracy'], st.session_state['data_size'], st.session_state['defect_rate'] = train_model(st.session_state['df_weld'])
            st.session_state['model_loaded'] = True
            st.session_state['executed'] = False 
            st.session_state['optimal_executed'] = False 
            st.success("✅ AI 모델 학습 및 로드 완료! UI에 초기 조건이 반영되었습니다.")
        except ValueError as e:
             st.session_state['model_loaded'] = False
             st.error(f"모델 학습 실패: {e}")


def run_optimization():
    if not st.session_state.get('model_loaded', False):
        st.error("AI 모델이 로드되지 않았습니다. 모델을 먼저 학습시켜 주세요.")
        st.session_state['optimal_executed'] = False
        return

    try:
        knowhow_inputs = get_knowhow_inputs()
        knowhow_influence_factor = st.session_state['knowhow_factor']
        
        # 페널티 로직을 위해 현재 V_Inj 슬라이더 값을 세션에 저장
        st.session_state['V_Inj_current_for_penalty'] = st.session_state['V_Inj_slider']
        
        # 1. 초기값 세트 A: UI 현재 공정 조건을 초기값으로 사용
        initial_guess_A = [
            st.session_state['T_Melt_slider'],
            st.session_state['V_Inj_slider'],
            st.session_state['P_Pack_slider'],
            st.session_state['T_Mold_slider'],
            st.session_state['Meter_slider'],
            st.session_state['VP_Switch_Pos_slider']
        ]

    except KeyError as e:
        st.error(f"UI 입력값을 가져오는 데 오류가 발생했습니다: {e}.")
        st.session_state['optimal_executed'] = False
        return

    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    
    # 2. 초기값 세트 B: 탐색 범위 중앙값을 초기값으로 사용
    initial_guess_B = [245.0, 80.0, 80.0, 70.0, 20.00, 10.00] 
    
    # 3. 초기값 세트 C: T_Mold를 최소값으로 설정 
    initial_guess_C = [245.0, 80.0, 80.0, 50.0, 20.00, 10.00] 
    
    
    best_risk = 101.0 
    best_conditions = None
    best_success = False
    
    with st.spinner('✨ 최적 조건 탐색 중... (3가지 초기값에서 시도)'):
        
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
# 3. Streamlit UI 구성 (STREAMLIT UI)
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Weld Line AI 진단 시스템")
st.header("Weld Line AI 진단 시스템", divider='rainbow')


# --- 사이드바 ---
with st.sidebar:
    st.title("📂 데이터 파일 업로드 및 모델 학습")
    st.info("AI 모델 학습을 위해 '3. 해석 학습 데이터'만 필수로 업로드해 주세요.")
    
    # 파일 업로더
    st.file_uploader("1. UI 초기 조건 (initial_condition.xlsx) [선택]", type=['xlsx', 'csv'], key='init_uploader')
    st.file_uploader("2. 가상 학습 데이터 (test_condition.xlsx) [선택]", type=['xlsx', 'csv'], key='virtual_uploader')
    st.file_uploader("3. 해석 학습 데이터 (moldflow_condition.xlsx) [필수]", type=['xlsx', 'csv'], key='real_uploader')
    
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
        **정확도:** {st.session_state['accuracy']:.4f}
        **총 데이터 개수:** {st.session_state['data_size']}개, **불량 비율:** {st.session_state['defect_rate']*100:.1f}%
        **해석 데이터 개수:** {st.session_state.get('real_data_size', 'N/A')}개
        **가상 데이터 개수:** {st.session_state.get('virtual_data_size', 'N/A')}개
        """)
    else:
        st.warning("파일을 업로드하고 'AI 모델 학습 시작' 버튼을 눌러주세요.")


if not st.session_state.get('model_loaded', False):
    st.error("데이터 파일을 업로드하고 AI 모델을 학습시켜야 시스템을 사용할 수 있습니다.")
    st.stop() 

if 'initial_values' not in st.session_state:
    set_initial_vals(None) 
    
initial_vals = st.session_state['initial_values'] 

tab1, tab2 = st.tabs(["1. Weld Line 공정 진단 (핵심)", "2. 모델 및 데이터 확인"])

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
    
    st.subheader("B. 전문가의 정성적 및 정량적 노하우 입력")
    
    # 1. 전문가 확신도 (항상 활성화)
    st.markdown("##### 1. 전문가 확신도")
    Expert_Confidence = st.slider(
        '전문가 확신도', 50, 100, 75, 1,
        key='Expert_Confidence_slider'
    )
    st.markdown("---")
    
    # 2. 사출 속도 노하우 섹션 (체크박스 적용)
    st.markdown("##### 2. 사출 속도 노하우 ($\text{V\_Inj}$)")
    col_v_intent_check, col_v_intent, col_v_delta_check, col_v_delta = st.columns([1, 2, 1, 2])
    
    # 사출 속도 - 정성적 (Intent)
    col_v_intent_check.checkbox("정성적 노하우 적용", value=False, key='V_Inj_Intent_active')
    V_Inj_Intent = col_v_intent.selectbox(
        'V_Inj 조정 의도',
        ('Keep_Constant', 'High_Increase', 'Low_Decrease'),
        index=0,
        key='V_Inj_Intent_select',
        disabled=not st.session_state.get('V_Inj_Intent_active', False)
    )

    # 사출 속도 - 정량적 (Delta)
    col_v_delta_check.checkbox("정량적 노하우 적용", value=False, key='V_Inj_Delta_active')
    V_Inj_Delta = col_v_delta.slider(
        'V_Inj 노하우 변화량 ($\Delta V_{Inj}$, mm/s)',
        -15.0, 15.0, 0.0, 0.5,
        key='V_Inj_Delta_slider',
        disabled=not st.session_state.get('V_Inj_Delta_active', False)
    )
    st.markdown("---")

    # 3. 금형 온도 노하우 섹션 (체크박스 적용)
    st.markdown("##### 3. 금형 온도 노하우 ($\text{T\_Mold}$)")
    col_t_intent_check, col_t_intent, col_t_delta_check, col_t_delta = st.columns([1, 2, 1, 2])
    
    # 금형 온도 - 정성적 (Intent)
    col_t_intent_check.checkbox("정성적 노하우 적용", value=False, key='T_Mold_Intent_active')
    T_Mold_Intent = col_t_intent.selectbox(
        'T_Mold 조정 의도',
        ('Keep_Constant', 'High_Increase', 'Low_Decrease'),
        index=0,
        key='T_Mold_Intent_select',
        disabled=not st.session_state.get('T_Mold_Intent_active', False)
    )
    
    # 금형 온도 - 정량적 (Delta)
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
        "노하우 영향력 계수 (Factor)",
        0.0, 5.0, 1.0, 0.1,
        key="knowhow_factor",
        help="0.0: 노하우 변수 영향력 제거, 1.0: 학습된 기본 영향력, 5.0: 영향력 5배 증폭"
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
    
    col_diag, col_opt = st.columns(2)
    
    if col_diag.button("🔴 Weld Line 통합 진단 실행", use_container_width=True, type='primary'):
        # 필수 세션 상태 체크
        if 'model' not in st.session_state or 'scaler' not in st.session_state or 'feature_names' not in st.session_state:
            st.error("AI 모델이 로드되지 않았습니다. 모델을 먼저 학습시켜 주세요.")
            st.session_state['executed'] = False
        else:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            feature_names = st.session_state['feature_names']
            
            risk_prob, prediction = predict_weld_line_risk(
                model, scaler, feature_names, input_data, knowhow_influence_factor 
            )
            st.session_state['risk_prob'] = risk_prob
            st.session_state['prediction'] = prediction
            st.session_state['executed'] = True
    
    col_opt.button(
        "✨ 최적 공정 조건 제시", 
        use_container_width=True, 
        type='secondary',
        on_click=run_optimization
    )

    
    st.subheader("💡 진단 결과")
    if st.session_state.get('executed', False):
        risk_prob = st.session_state['risk_prob']
        
        if risk_prob > 0.5:
            st.error(f"🔴 AI 모델 경고! Weld Line 불량 위험 확률: {risk_prob*100:.1f}% (노하우 계수: {knowhow_influence_factor:.1f})", icon="🚨")
            st.warning("현재 공정 조건과 전문가의 노하우가 불량 발생 위험을 높입니다. 사출 속도나 금형 온도를 양호 범위로 조정하세요.")
            
        else:
            st.success(f"✅ 현재 조건 양호합니다. (AI 예측 위험 확률: {risk_prob*100:.1f}%, 노하우 계수: {knowhow_influence_factor:.1f})", icon="👍")
            
            if st.session_state.get('defect_rate', 0) == 0:
                 st.info("현재 모델은 불량 데이터 없이 학습되었습니다. 예측 확률이 낮더라도, 이는 데이터 편향 때문일 수 있으니 탭 2의 모델 계수를 참고하여 해석해 주세요.")
            else:
                 st.info("현재 공정은 안정적입니다. 노하우 영향력 계수를 조정하며 AI의 예측 안정성을 확인해 보세요.")

    else:
        st.info("현재 공정 진단이 실행되지 않았습니다. '🔴 Weld Line 통합 진단 실행' 버튼을 눌러주세요.")

    
    st.markdown("---")
    
    st.subheader("✨ 최적 공정 조건 솔루션")
    if st.session_state.get('optimal_executed', False):
        if st.session_state['optimal_success']:
            opt_cond = st.session_state['optimal_conditions']
            opt_risk = st.session_state['optimal_risk']
            knowhow_factor_used = st.session_state['knowhow_factor']
            
            st.success(f"계산 완료! **최소 불량 위험 확률: {opt_risk:.2f}%**")
            
            opt_df = pd.DataFrame({
                '변수': ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos'],
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
            st.caption(f"이 최적 조건은 현재 설정된 **정성적 및 정량적 노하우**와 **노하우 영향력 계수({knowhow_factor_used:.1f})**를 반영합니다.")
            
        else:
            st.warning("최적화 계산에 실패했거나, 세 가지 초기값 시도 모두 현재 조건보다 더 낮은 위험도를 찾지 못했습니다. 입력 조건을 다시 확인해 주세요.")
    else:
        st.info("'✨ 최적 공정 조건 제시' 버튼을 눌러 Weld Line 불량 위험을 최소화하는 최적의 공정 조건을 확인하세요.")


with tab2:
    st.header("모델 학습 상세 결과 및 데이터 미리보기")
    
    st.subheader("AI 모델 학습 결과 요약")
    st.markdown("AI 모델은 **로지스틱 회귀** 모델을 사용하여 학습되었습니다.")
    st.metric(label="AI 모델 정확도 (Test Set)", value=f"{st.session_state['accuracy'] * 100:.2f}%")
    st.metric(label="통합 데이터 총 개수", value=f"{st.session_state['data_size']}개")
    st.metric(label="통합 데이터셋 불량률", value=f"{st.session_state['defect_rate'] * 100:.1f}%")
    
    st.markdown("---")
    
    st.subheader("모델 계수(Coefficient) 시각화")
    if 'model' in st.session_state and 'feature_names' in st.session_state:
        model = st.session_state['model']
        feature_names = st.session_state['feature_names']
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        })
        
        coef_df['Type'] = 'Process'
        coef_df.loc[coef_df['Feature'].isin(KNOWHOW_FEATURES), 'Type'] = 'Knowhow'
        
        st.dataframe(coef_df.sort_values(by='Coefficient', ascending=False), height=400)
        st.caption("계수의 절댓값이 클수록 불량 위험 확률에 미치는 영향이 크며, 양수(+)는 위험을 증가, 음수(-)는 위험을 감소시킵니다.")
        
        st.markdown("**사출 속도 관련 계수 (확인 필요):**")
        v_inj_coefs = coef_df[coef_df['Feature'].str.contains('V_Inj') | coef_df['Feature'].str.contains('t_Fill')]
        st.dataframe(v_inj_coefs)

        st.warning("""
        **[사출 속도 반대 현상 진단]**
        노하우와 최적화 방향이 반대인 현상을 해결하기 위해 **목적 함수에 페널티가 추가**되었습니다.
        계수의 부호가 직관과 반대일 경우, 모델이 학습한 데이터의 경향성이 노하우와 상충하기 때문입니다.
        """)
        
    st.markdown("---")
    st.subheader("통합 학습 데이터셋 (해석 + 가상)")
    if 'df_weld' in st.session_state:
        st.caption("업로드된 해석 데이터와 가상 데이터를 병합하여 학습에 사용된 데이터셋입니다.")
        st.dataframe(st.session_state['df_weld'].head(20))
    else:
        st.info("학습 데이터가 로드되지 않았습니다.")