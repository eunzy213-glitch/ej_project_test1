import pandas as pd # 데이터 조작을 위한 pandas 임포트
from sklearn.model_selection import train_test_split # 훈련/테스트 데이터 분할 함수 임포트
from sklearn.preprocessing import PolynomialFeatures # 다항식 피처 생성을 위한 클래스 임포트
from sklearn.linear_model import LinearRegression, HuberRegressor # 선형 회귀 및 강건한 회귀(Huber) 모델 임포트
from catboost import CatBoostRegressor # CatBoost 모델 임포트 (경량 부스팅 트리)
from lightgbm import LGBMRegressor # LightGBM 모델 임포트 (경량 Gradient Boosting Machine)
from sklearn.pipeline import Pipeline # 여러 변환 단계를 묶어주는 Pipeline 클래스 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트

def train_all_models(data: pd.DataFrame, X: pd.DataFrame, y: pd.Series, prefix: str):
    """
    주어진 데이터셋(X, y)에 대해 5가지 모델을 학습시키고 결과를 반환합니다.
    (요청에 따라 Target_R과 SG는 학습 피처에서 제외됩니다.)
    
    Args:
        data: 원본 또는 전처리된 데이터프레임 (학습 과정에 직접 사용되지는 않지만 함수 정의상 받음)
        X: 피처 데이터프레임 (BG가 제외된 상태)
        y: 타겟 변수 (BG) 시리즈
        prefix: 모델 이름에 붙일 접두사 ('Raw' 또는 'Processed')
    
    Returns:
        results: 테스트 결과 (예측값, 실제값) 리스트
        trained_models: 학습된 모델 객체와 이름 리스트 (피처 중요도 시각화용)
        feature_names: 최종 인코딩된 피처 이름 목록
    """
    
    # 1. 피처 인코딩 및 분할
    
    # ✅ 핵심 수정 반영: Target_R과 SG를 학습 피처에서 제외
    features_to_exclude = ['Target_R', 'SG'] # 제외할 피처 목록 정의
    # X에서 제외 목록에 있는 컬럼을 제거. errors='ignore'는 해당 컬럼이 없어도 오류를 발생시키지 않음.
    X_filtered = X.drop(columns=[col for col in features_to_exclude if col in X.columns], errors='ignore')
    
    # 제외 후 남은 피처에 대해 One-Hot Encoding 적용
    # 범주형 피처를 0과 1의 이진 피처로 변환 (모델 학습 가능하도록)
    # drop_first=True는 다중공선성(Multicollinearity) 방지를 위해 첫 번째 범주를 제거함
    X_encoded = pd.get_dummies(X_filtered, drop_first=True)
    
    # 모델 학습에 사용할 최종 피처 이름 목록
    feature_names = X_encoded.columns.tolist()
    
    # 데이터 분할 (Train: 80%, Test: 20%)
    # 학습 데이터와 테스트 데이터를 무작위로 분할 (random_state=42로 결과 재현성 확보)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 2. 모델 정의 및 학습 파이프라인
    
    models = [
        # 1. 일반 선형 회귀 모델
        (f'{prefix}_Linear_Model', LinearRegression()),
        
        # 2. 3차 다항 회귀 모델 (선형 모델에 다항식 변환을 추가하는 Pipeline 사용)
        (f'{prefix}_Poly_3_Model', Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)), # 3차 다항식 피처 생성
            ('linear', LinearRegression()) # 생성된 피처로 선형 회귀 학습
        ])),
        
        # 3. Huber Regressor (이상치에 강건한 회귀 모델)
        # epsilon=1.35 (기본값) 이하의 잔차는 제곱 오차, 초과는 선형 오차로 처리
        (f'{prefix}_Robust_Huber_Model', HuberRegressor(epsilon=1.35)), 
        
        # 4. CatBoost Regressor (범주형 데이터 처리에 강점)
        (f'{prefix}_CatBoost_Model', CatBoostRegressor(
            verbose=0, random_state=42, allow_writing_files=False, # 학습 로그 출력 끄기, 재현성, 임시 파일 생성 방지
            n_estimators=500 # 트리의 개수 설정
        )),
        
        # 5. LightGBM Regressor (빠른 학습 속도)
        (f'{prefix}_LightGBM_Model', LGBMRegressor(
            random_state=42, verbose=-1, n_estimators=500 # 학습 로그 출력 끄기, 재현성, 트리의 개수 설정
        ))
    ]

    results = [] # 모델의 예측 결과(y_test, Prediction)를 저장할 리스트
    trained_models = [] # 학습된 모델 객체 정보를 저장할 리스트
    
    print(f"   - {prefix} 데이터셋 학습 중 (Target_R, SG 제외)...")
    
    for model_name, model in models:
        print(f"     > {model_name} 학습 중...")
        
        # 모델 학습 (정의된 모델 객체에 X_train과 y_train을 사용하여 학습)
        model.fit(X_train, y_train)
        
        # 예측
        # 학습된 모델을 사용하여 X_test에 대한 예측값 생성
        prediction = model.predict(X_test)
        
        # 결과 저장
        results.append({
            'Model': model_name,
            'y_test': y_test,
            'Prediction': pd.Series(prediction, index=y_test.index)
        })
        
        # 모델 객체와 피처 이름 저장 (피처 중요도 시각화에 사용)
        # 튜플 형태로 (모델 이름, 모델 객체, 최종 피처 이름 목록) 저장
        trained_models.append((model_name, model, feature_names))

    # 학습 결과(results), 모델 객체(trained_models), 최종 피처 이름 목록(feature_names) 반환
    return results, trained_models, feature_names