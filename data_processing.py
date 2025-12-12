# data_processing.py

import pandas as pd # 데이터 분석 및 조작을 위한 pandas 임포트
from statsmodels.nonparametric.api import lowess # LOWESS (Locally Weighted Scatterplot Smoothing) 함수 임포트
from sklearn.ensemble import IsolationForest # 이상치 탐지 모델인 Isolation Forest 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 수치형 스케일링, 범주형 인코딩 클래스 임포트
from sklearn.compose import ColumnTransformer # 여러 전처리 단계를 병렬로 처리하는 ColumnTransformer 임포트

# ----------------------------------------------------
# 1. 데이터 요약 정보 출력 함수
# ----------------------------------------------------
def describe_data(data: pd.DataFrame, title: str):
    """데이터의 통계적 요약 및 결측치 정보를 출력합니다."""
    # # describe_data 함수 본문 시작
    print(f"\n--- {title} ---") # 제목 출력
    print(data.describe(include='all').T)  # 데이터프레임의 통계적 요약 (수치형/범주형 모두 포함) 출력 및 전치
    print(f"   - 결측치 정보: \n{data.isnull().sum()[data.isnull().sum() > 0]}") # 결측치가 1개 이상인 컬럼과 그 개수를 출력
    print(f"   - 데이터 shape: {data.shape}") # 데이터의 행과 열 크기 (shape) 출력

# ----------------------------------------------------
# 2. 데이터 로드 함수
# ----------------------------------------------------
def load_data(file_path: str):
    """지정된 경로에서 CSV 파일을 읽어와 반환합니다."""
    try:
        raw_data = pd.read_csv(file_path) # CSV 파일을 읽어 raw_data에 저장
        print(f"'{file_path}' 파일 로드 성공. 데이터 shape: {raw_data.shape}")
        return raw_data # 로드된 데이터프레임 반환
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}") # 파일 로드 실패 시 오류 메시지 출력
        return pd.DataFrame() # 빈 데이터프레임 반환

# ----------------------------------------------------
# 3. 데이터 전처리 및 파이프라인 설정 함수 (핵심)
# ----------------------------------------------------
def preprocess_data(raw_data: pd.DataFrame, feature_cols: list, numerical_cols: list, categorical_cols: list):
    """
    요청된 feature_cols에 따라 피처를 유동적으로 선택하고 전처리 파이프라인을 설정합니다.
    """
    data = raw_data.copy() # 원본 데이터를 수정하지 않기 위해 복사본 생성
    
    # 1. 결측치 처리 (수치형: 평균, 범주형: 최빈값)
    for col in numerical_cols: # 수치형 컬럼의 결측치를 평균값(mean)으로 대체
        data[col] = data[col].fillna(data[col].mean())
    for col in categorical_cols: # 범주형 컬럼의 결측치를 최빈값(mode)으로 대체
        if col in data.columns and data[col].dtype == 'object': # 컬럼이 데이터에 존재하고 object 타입인지 확인
            data[col] = data[col].fillna(data[col].mode()[0]) # mode()[0]은 최빈값(Series)의 첫 번째 요소
        
    # 2. LOWESS 기반 Isolation Forest 이상치 제거 (BG와 SG 관계에 의존)
    if 'SG' in data.columns and 'BG' in data.columns: # 핵심 컬럼인 'SG'와 타겟인 'BG'가 모두 존재할 때만 이상치 제거 수행
        print("   - LOWESS 기반 Isolation Forest 이상치 제거 중...")
        
        # LOWESS (Locally Weighted Scatterplot Smoothing) 적용
        # SG를 독립 변수로, BG를 종속 변수로 사용하여 추세선 계산 (frac=0.3: 전체 데이터의 30%를 사용하여 스무딩)
        lowess_result = lowess(data['BG'], data['SG'], frac=0.3)
        data['lowess_residual'] = data['BG'] - lowess_result[:, 1] # 잔차 계산: 실제 BG 값에서 LOWESS 추세선의 예측값 (lowess_result[:, 1])을 뺌
        
        # Isolation Forest에 사용할 피처 정의: 잔차와 다른 모든 수치형 피처를 포함
        feature_for_iforest = ['lowess_residual'] + [col for col in numerical_cols if col in data.columns]
        
        # Isolation Forest 모델 초기화 및 설정 (contamination=0.05: 데이터의 5%를 이상치로 간주)
        if_model = IsolationForest(random_state=42, contamination=0.05) 
        data['outlier'] = if_model.fit_predict(data[feature_for_iforest]) # 모델 학습 및 이상치 예측 (1: 정상 데이터, -1: 이상치)
        
        # 이상치 제거: 'outlier' 값이 1인 행(정상 데이터)만 선택하고, 
        # 탐지용으로 추가했던 'outlier' 및 'lowess_residual' 컬럼을 제거
        preprocessed_data = data[data['outlier'] == 1].drop(columns=['outlier', 'lowess_residual'])
        print(f"   - 이상치 제거 후 N: {len(preprocessed_data)} (제거된 비율: {(len(raw_data) - len(preprocessed_data)) / len(raw_data) * 100:.2f}%)") # 제거된 데이터 비율 출력

    else:
        print("   - BG 또는 SG 컬럼이 없어 LOWESS 기반 이상치 제거를 건너뜀.")
        preprocessed_data = data.copy() # 이상치 제거 없이 데이터 복사


    # 3. 피처 엔지니어링 파이프라인 (Scaling 및 Encoding)
    # 현재 실험에서 사용할 수치형 피처 목록만 필터링
    num_for_pipe = [col for col in numerical_cols if col in feature_cols]
    cat_for_pipe = [col for col in categorical_cols if col in feature_cols] # 현재 실험에서 사용할 범주형 피처 목록만 필터링

    # ColumnTransformer를 사용하여 전처리 파이프라인 정의
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_for_pipe), # 'num' 변환기: 수치형 피처에 StandardScaler (표준화) 적용
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_for_pipe) # 'cat' 변환기: 범주형 피처에 OneHotEncoder 적용 (알 수 없는 값 무시, 희소 행렬 대신 일반 배열 반환)
        ],
        remainder='passthrough' # 위의 두 그룹에 포함되지 않은 나머지 피처는 그대로 통과
    )
    
    # 4. 최종 X, Y 준비
    # 최종 학습 피처 X: 이상치 제거된 데이터에서 현재 실험에 필요한 feature_cols만 선택
    X = preprocessed_data[[col for col in feature_cols if col in preprocessed_data.columns]]
    y = preprocessed_data['BG'] # 최종 타겟 Y: 이상치 제거된 데이터에서 'BG' 컬럼 선택
    
    # ColumnTransformer 객체를 X 데이터에 대해 학습 (fit)
    # 이 객체는 main.py로 반환되어 model_training.py의 Pipeline에 포함됩니다.
    preprocessor.fit(X)
        
    return X, y, preprocessor # 전처리된 피처, 타겟, 전처리 객체 반환