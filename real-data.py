import streamlit as st
import pandas as pd
import numpy as np요
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize

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

# -------------------------------------------------------------
# 🌟 오류 해결을 위한 핵심 수정: 슬라이더 초기값을 여기서 무조건 float으로 설정!
# -------------------------------------------------------------
for var, default_val in DEFAULT_INPUT_VALS.items():
    # 'input_T_Melt' 등의 키가 세션에 없으면 기본값을 할당하고, float으로 명시적 변환
    if f'input_{var}' not in st.session_state:
        st.session_state[f'input_{var}'] = float(default_val)
# -------------------------------------------------------------


# =================================================================
# 1. 데이터 로드 및 전처리 함수
# =================================================================

@st.cache_data(show_spinner=False)
def load_df_from_uploader(uploaded_file):
    """업로드된 파일(xlsx, csv)을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            # 파일 확장자를 확인하여 로드 함수 결정
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                # openpyxl 종속성 사용 (requirements.txt에 추가됨)
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"⚠️ 지원하지 않는 파일 형식입니다: .{file_extension}")
                return None
            
            # 컬럼명 앞뒤 공백 제거
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """실제 데이터와 가상 데이터를 결합하고 전처리합니다."""
    
    # None이 아니거나 비어있지 않은 DataFrame만 필터링하여 결합 (오류 방지 로직)
    valid_dataframes = [df for df in [df_real, df_virtual] if df is not None and not df.empty]
    
    if not valid_dataframes:
        st.warning("⚠️ 학습에 사용할 유효한 데이터(moldflow_condition.xlsx)가 로드되지 않았습니다.")
        return pd.DataFrame() # 빈 DataFrame 반환하여 에러 방지

    df_combined = pd.concat(valid_dataframes, ignore_index=True)
    
    # Y_Weld를 불량(1) / 정상(0)으로 이진화
    df_combined[TARGET_VAR] = np.where(df_combined[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
    
    # 필요한 컬럼만 선택
    required_cols = PROCESS_VARS + [TARGET_VAR]
    if not all(col in df_combined.columns for col in required_cols):
        st.error("⚠️ 데이터에 필수 컬럼(T_Melt, V_Inj, ..., Y_Weld)이 누락되었습니다. 컬럼 이름을 확인해 주세요.")
        return pd.DataFrame()
        
    df_processed = df_combined[required_cols].copy()
    
    return df_processed

# =================================================================
# 2. 모델 학습 함수
# =================================================================

def train_model(df):
    """데이터를 사용하여 로지스틱 회귀 모델을 학습하고 스케일러를 저장합니다."""
    if df.empty:
        st.error("⚠️ 학습할 데이터가 비어 있습니다. 파일이 올바르게 로드되었는지 확인해 주세요.")
        return None, None
        
    # X와 Y 분리
    X = df[PROCESS_VARS]
    Y = df[TARGET_VAR]
    
    # 스케일링 (MinMaxScaler 사용을 가정)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 모델 학습 (로지스틱 회귀)
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, Y)
    
    return model, scaler

# =================================================================
# 3. 예측 및 최적화 함수
# =================================================================

def predict_weld_risk(model, scaler, input_data):
    """입력 데이터에 대한 불량 확률을 예측합니다."""
    if model is None or scaler is None:
        # st.warning("모델 학습이 필요합니다.") # 예측 함수 내에서는 경고 표시 자제
        return 0.5 # 모델이 없으면 중간값 반환
        
    # 입력 데이터를 DataFrame으로 변환 (컬럼 순서 유지)
    input_df = pd.DataFrame([input_data], columns=PROCESS_VARS)
    
    # 스케일링
    input_scaled = scaler.transform(input_df)
    
    # 예측 확률 (불량=1일 확률)
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
    
    # 파일 업로더
    uploaded_file_init = st.file_uploader(
        "1. UI 초기 조건 (initial_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="init_file"
    )
    uploaded_file_virtual = st.file_uploader(
        "2. 가상 학습 데이터 (test_condition.xlsx) [선택]", type=['xlsx', 'csv'], key="virtual_file"
    )
    uploaded_file_real = st.file_uploader(
        "3. 해석 학습 데이터 (moldflow_condition.xlsx) [필수]", type=['xlsx', 'csv'], key="real_file"
    )

    # 세션 상태에 파일 로드 (함수 호출) - 캐싱 함수 사용
    st.session_state['df_init'] = load_df_from_uploader(uploaded_file_init)
    st.session_state['df_virtual'] = load_df_from_uploader(uploaded_file_virtual)
    st.session_state['df_real'] = load_df_from_uploader(uploaded_file_real)


    def load_and_train_model():
        """파일을 로드하고 모델 학습을 실행하는 콜백 함수"""
        
        # 1. 데이터 전처리 및 결합
        df_weld_processed = process_weld_data(st.session_state['df_virtual'], st.session_state['df_real'])
        st.session_state['df_weld'] = df_weld_processed
        
        if st.session_state['df_weld'].empty:
            st.error("🚨 모델 학습 실패: 필수 데이터(3번 파일)가 로드되지 않았습니다.")
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            return

        # 2. 모델 학습
        model, scaler = train_model(st.session_state['df_weld'])
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

        if model is not None:
            st.success("✅ AI 모델 학습 및 로드 완료! UI에 초기 조건이 반영되었습니다.")
            
            # 3. 초기 조건 반영 (있을 경우)
            if st.session_state['df_init'] is not None and not st.session_state['df_init'].empty:
                # 첫 번째 행을 초기 조건으로 사용
                init_row = st.session_state['df_init'].iloc[0]
                for var in PROCESS_VARS:
                    if var in init_row:
                        try:
                            # 값을 float으로 변환하여 안전하게 저장 (데이터 타입 오류 방지)
                            st.session_state[f'input_{var}'] = float(init_row[var])
                        except ValueError:
                            st.warning(f"⚠️ 초기 조건 파일의 '{var}' 값이 유효한 숫자가 아닙니다. 기본값을 유지합니다.")


    st.button("🚀 파일 로드 및 AI 모델 학습 시작", on_click=load_and_train_model)

    st.markdown("---")
    st.header("ℹ️ 시스템 상태 확인")

    # 시스템 상태 표시
    if st.session_state['model'] is not None:
        st.success("모델 상태: 학습 완료")
        
        # 데이터 통계 표시
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

    # -------------------------------------------------------------
    # 🌟 슬라이더 UI 생성 (value= 명시적 사용으로 오류 해결)
    # -------------------------------------------------------------
    input_vars = {}
    
    # 모든 슬라이더에서 value= 인수를 명시적으로 사용합니다. (오류 방지 핵심)
    with col_melt:
        input_vars['T_Melt'] = st.slider(
    '용융 온도 (T_Melt)',
    min_value=230.0,
    max_value=260.0,
    value=240.0,
    step=5.0,
    key='slider_T_Melt'
)
    with col_inj:
        input_vars['V_Inj'] = st.slider(
            '사출 속도 (V_Inj)', 
            1, 
            10, 
            value=st.session_state['input_V_Inj'], # value= 명시
            step=1, 
            key='slider_V_Inj'
        )
    with col_pack:
        input_vars['P_Pack'] = st.slider(
            '보압 (P_Pack)', 
            50, 
            100, 
            value=st.session_state['input_P_Pack'], # value= 명시
            step=5, 
            key='slider_P_Pack'
        )
    with col_mold:
        input_vars['T_Mold'] = st.slider(
            '금형 온도 (T_Mold)', 
            30, 
            80, 
            value=st.session_state['input_T_Mold'], # value= 명시
            step=5, 
            key='slider_T_Mold'
        )
    with col_meter:
        input_vars['Meter'] = st.slider(
            '계량 위치 (Meter)', 
            180, 
            200, 
            value=st.session_state['input_Meter'], # value= 명시
            step=1, 
            key='slider_Meter'
        )
    with col_vp:
        input_vars['VP_Switch_Pos'] = st.slider(
            'VP 전환 위치', 
            10, 
            20, 
            value=st.session_state['input_VP_Switch_Pos'], # value= 명시
            step=1, 
            key='slider_VP_Switch_Pos'
        )

    st.markdown("---")
    st.header("B. 전문가의 정성적 및 정량적 노하우 입력")

    # 노하우 입력 (V_Inj, T_Mold에 대한 가정)
    col_intent_v, col_delta_v, col_intent_t, col_delta_t = st.columns(4)

    with col_intent_v:
        v_inj_intent = st.radio("V_Inj 노하우", ['Keep_Constant', 'Increase', 'Decrease'], horizontal=True)
    with col_delta_v:
        v_inj_delta = st.number_input("V_Inj 변화폭 (±)", min_value=0, max_value=5, value=0)

    with col_intent_t:
        t_mold_intent = st.radio("T_Mold 노하우", ['Keep_Constant', 'Increase', 'Decrease'], horizontal=True)
    with col_delta_t:
        t_mold_delta = st.number_input("T_Mold 변화폭 (±)", min_value=0, max_value=5, value=0)

    st.markdown("---")
    
    # -----------------
    # 진단 실행
    # -----------------
    if st.session_state['model'] is not None:
        
        # 현재 조건 진단
        current_risk = predict_weld_risk(st.session_state['model'], st.session_state['scaler'], input_vars)
        st.info(f"🟢 현재 조건에서의 불량 위험 확률: **{current_risk*100:.2f}%**")
        
        
        def run_diagnosis():
            """진단 버튼 클릭 시 실행"""
            # 이미 위에서 current_risk를 계산했으므로, 여기서는 UI 업데이트만.
            if current_risk >= 0.5:
                st.error("🔴 위험도 높음: 즉시 최적화 조건을 검토하세요.")
            else:
                st.success("🟢 위험도 낮음: 현재 조건을 유지해도 좋습니다.")
                
        # -----------------
        # 최적화 실행
        # -----------------
        def run_optimization():
            """최적 공정 조건 제시 버튼 클릭 시 실행"""
            model = st.session_state['model']
            scaler = st.session_state['scaler']

            # 최적화 목표 함수 (불량 확률 최소화)
            def objective_function(X_array):
                X_df = pd.DataFrame([X_array], columns=PROCESS_VARS)
                return predict_weld_risk(model, scaler, X_df.iloc[0].to_dict())

            # 초기값 설정 (현재 사용자 입력값)
            X0 = np.array([input_vars[var] for var in PROCESS_VARS])

            # 노하우 제약 조건 설정
            constraints = []
            
            # T_Melt, P_Pack, Meter, VP_Switch_Pos는 현재 값으로 고정 (노하우 없음 가정)
            for i, var in enumerate(['T_Melt', 'P_Pack', 'Meter', 'VP_Switch_Pos']):
                constraints.append({'type': 'eq', 
                                    'fun': lambda X, idx=i, val=X0[i]: X[idx] - val})

            # V_Inj 노하우 제약
            v_inj_idx = PROCESS_VARS.index('V_Inj')
            v_min, v_max = 1, 10
            
            # V_Inj_Intent에 따라 경계 조정
            if v_inj_intent == 'Increase':
                v_min = max(v_min, input_vars['V_Inj'] + v_inj_delta)
            elif v_inj_intent == 'Decrease':
                v_max = min(v_max, input_vars['V_Inj'] - v_inj_delta)
            
            # T_Mold 노하우 제약
            t_mold_idx = PROCESS_VARS.index('T_Mold')
            t_min, t_max = 30, 80
            
            # T_Mold_Intent에 따라 경계 조정
            if t_mold_intent == 'Increase':
                t_min = max(t_min, input_vars['T_Mold'] + t_mold_delta)
            elif t_mold_intent == 'Decrease':
                t_max = min(t_max, input_vars['T_Mold'] - t_mold_delta)

            # 변수별 경계 설정 (Bounds) - 순서 중요!
            bounds = [
                (200, 300),  # T_Melt
                (v_min, v_max),  # V_Inj (노하우 반영)
                (50, 100),  # P_Pack
                (t_min, t_max),  # T_Mold (노하우 반영)
                (180, 200),  # Meter
                (10, 20)     # VP_Switch_Pos
            ]

            try:
                # 최적화 실행 (SLSQP는 제약 조건에 적합)
                result = minimize(objective_function, X0, method='SLSQP', bounds=bounds, constraints=constraints)
            
                if result.success:
                    opt_params = {PROCESS_VARS[i]: round(result.x[i], 1) for i in range(len(PROCESS_VARS))}
                    opt_risk = predict_weld_risk(model, scaler, opt_params)
                    
                    st.success("✨ 최적 공정 조건 제시 결과")
                    st.write(f"**최소 불량 위험 확률:** **{opt_risk*100:.2f}%**")
                    
                    opt_table = pd.DataFrame([opt_params])
                    opt_table = opt_table.T.rename(columns={0: '최적 공정 조건'})
                    st.dataframe(opt_table)
                    
                    st.markdown("---")
                    st.markdown("##### 🔍 최적화 요약")
                    
                    # 최적화 결과와 현재 조건 비교
                    summary_data = {}
                    for var in PROCESS_VARS:
                        if round(input_vars[var], 1) != opt_params[var]:
                            change = "↑ 상향" if opt_params[var] > input_vars[var] else "↓ 하향"
                            summary_data[var] = f"{opt_params[var]} ({change})"
                    
                    if summary_data:
                        st.table(pd.DataFrame(summary_data.values(), index=summary_data.keys(), columns=['변화된 조건']))
                    else:
                        st.info("현재 조건이 이미 최적 조건에 가깝습니다.")
                    
                else:
                    st.error(f"⚠️ 최적화 실패: {result.message}")

            except Exception as e:
                st.error(f"최적화 실행 중 치명적인 오류 발생: {e}")

        # 진단 및 최적화 버튼 분리
        col_diag, col_opt = st.columns([1,1])
        with col_diag:
            st.button("🔴 Weld Line 통합 진단 실행", on_click=run_diagnosis, use_container_width=True)
        with col_opt:
            st.button("✨ 최적 공정 조건 제시", on_click=run_optimization, use_container_width=True)

    else:
        st.error("🚨 AI 모델이 학습되지 않았습니다. 사이드바에서 파일을 업로드하고 'AI 모델 학습 시작' 버튼을 눌러주세요.")


with tab2:
    st.header("모델 및 데이터 확인")
    
    if st.session_state['model'] is not None:
        model = st.session_state['model']
        st.subheader("1. 학습된 로지스틱 회귀 모델 계수")
        
        # 모델 계수 표로 표시
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
