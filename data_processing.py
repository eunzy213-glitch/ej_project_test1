# 데이터를 불러와 기본적인 결측치 처리, LOWESS 기반 잔차 계산을 통한 이상치 탐지 및 제거등 모델 학습을 위한 핵심 전처리 단계

import pandas as pd # 데이터 분석 및 조작을 위한 pandas 라이브러리 임포트
from statsmodels.nonparametric.api import lowess # LOWESS (Locally Weighted Scatterplot Smoothing) 함수를 직접 임포트
from sklearn.ensemble import IsolationForest # 이상치 탐지 모델인 Isolation Forest 임포트

def describe_data(data: pd.DataFrame, title: str):
    """데이터의 통계적 요약 및 결측치 정보를 출력합니다."""
    print(f"\n--- {title} ---")
    print(data.describe(include='all')) # 데이터프레임의 통계적 요약 (수치형/범주형 모두 포함) 출력
    print(f"   - 결측치 정보: \n{data.isnull().sum()[data.isnull().sum() > 0]}") # 결측치가 있는 컬럼과 그 개수를 출력
    print(f"   - 데이터 shape: {data.shape}") # 데이터의 행과 열 크기 (shape) 출력

def load_and_preprocess_data(file_path: str, lowess_feature: str = 'SG'):
    """
    데이터를 로드하고 범주형 변수를 처리하며, LOWESS 기반 Isolation Forest로 이상치를 제거합니다.
    """
    try:
        # 지정된 경로에서 CSV 파일을 읽어와 raw_data 변수에 저장
        raw_data = pd.read_csv(file_path)
        print(f"'dataset.csv' 파일 로드 성공. 데이터 shape: {raw_data.shape}")
    except FileNotFoundError:
        # 파일이 없을 경우 오류 메시지 출력 후 None 반환
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return None, None
    
    # 1. 데이터 복사 및 범주형 변수 처리
    data = raw_data.copy() # 원본 데이터(raw_data)를 복사하여 data 변수에서 작업
    
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist() # object 타입 (범주형) 컬럼 리스트 생성
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist() # float/int 타입 (수치형) 컬럼 리스트 생성

    # 결측치 처리 (여기서는 단순하게 처리)
    for col in ['Family_History', 'Pregnancy']:
        if col in data.columns:
            # 해당 범주형 컬럼에 대해 결측치(NaN)를 'Unknown' 문자열로 채움
            data[col] = data[col].fillna('Unknown') 

    # 2. 전처리 전 데이터 Describe
    describe_data(data, "[1] 전처리 전 데이터 Describe") # 이상치 제거 전 데이터의 통계 요약 출력
    
    # 3. LOWESS 스무딩 기반 잔차 계산 (기준 피처: SG)
    print(f"\n   - LOWESS 스무딩 기반 잔차 계산 중 (기준 피처: {lowess_feature})...")
    
    # LOWESS 계산
    lowess_result = lowess(
        endog=data['BG'], # 종속 변수 (예측 대상): BG
        exog=data[lowess_feature], # 독립 변수 (기준 피처): SG
        frac=0.3 # 스무딩 정도를 결정하는 파라미터 (0.3은 전체 데이터의 30%를 사용하여 스무딩)
    )
    
    # 잔차 추가
    # lowess_result[:, 1]은 LOWESS 추세선의 예측 값(y_fitted)을 의미
    # 실제 BG 값과 추세선의 예측 값의 차이(잔차)를 새로운 컬럼에 저장
    data['lowess_residual'] = data['BG'] - lowess_result[:, 1]
    
    # 4. Isolation Forest를 이용한 이상치 제거
    print("   - Isolation Forest를 이용한 이상치 제거 중...")
    
    # 이상치 탐지에 사용할 피처 (BG, SG 포함하여 관계 기반 이상치 탐지)
    feature_for_iforest = ['lowess_residual'] + numerical_cols
    
    # Isolation Forest 모델 초기화 및 설정 (contamination=0.05는 전체 데이터의 5%를 이상치로 간주)
    if_model = IsolationForest(random_state=42, contamination=0.05) 
    
    # 모델 학습 및 이상치 예측 (1: 정상, -1: 이상치)
    data['outlier'] = if_model.fit_predict(data[feature_for_iforest])
    
    # 이상치 제거
    # 'outlier' 컬럼 값이 1인 행만 선택 (정상 데이터)하고, 이상치 탐지용으로 추가했던 두 컬럼을 제거
    preprocessed_data = data[data['outlier'] == 1].drop(columns=['lowess_residual', 'outlier'])
    
    # 제거된 이상치의 개수 계산
    removed_count = data.shape[0] - preprocessed_data.shape[0]
    print(f"   - 전처리 후 데이터 shape: {preprocessed_data.shape} ({removed_count}개 이상치 제거)")

    # 5. 전처리 후 데이터 Describe
    describe_data(preprocessed_data, "[2] 전처리 후 데이터 Describe") # 이상치 제거 후 데이터의 통계 요약 출력
    
    # 전처리 전 데이터 (단, 임시로 추가했던 'lowess_residual', 'outlier' 컬럼은 제거)
    data_raw_clean = data.drop(columns=['lowess_residual', 'outlier'])

    # 이상치가 제거되지 않은 원본 데이터와 이상치가 제거된 최종 전처리 데이터를 반환
    return data_raw_clean, preprocessed_data