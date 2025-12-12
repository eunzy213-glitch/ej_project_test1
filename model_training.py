# model_training.py

import pandas as pd # 데이터 조작을 위한 pandas 임포트
from sklearn.model_selection import train_test_split # 훈련/테스트 데이터 분할 함수 임포트
from sklearn.preprocessing import PolynomialFeatures # 다항식 피처 생성을 위한 클래스 임포트
from sklearn.linear_model import LinearRegression, HuberRegressor # 선형 회귀 및 강건한 회귀(Huber) 모델 임포트
from catboost import CatBoostRegressor # CatBoost 모델 임포트
from lightgbm import LGBMRegressor # LightGBM 모델 임포트
from sklearn.pipeline import Pipeline # 여러 변환 단계를 묶어주는 Pipeline 클래스 임포트
from sklearn.compose import ColumnTransformer # 전처리기 객체 타입 힌트용 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트

# ----------------------------------------------------
# 1. 모델 학습 메인 함수
# ----------------------------------------------------
def train_all_models(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, feature_names: list, prefix: str):
    """
    주어진 데이터셋에 대해 5가지 모델을 학습시키고 결과를 반환합니다.
    
    Args:
        X: 피처 데이터프레임
        y: 타겟 변수 시리즈
        preprocessor: data_processing.py에서 생성된 ColumnTransformer 객체
        feature_names: X의 원래 피처 이름 목록 (여기서는 사용되지 않지만, 함수 정의상 남겨둠)
        prefix: 모델 이름에 붙일 실험 조건 접두사 (예: 'T1-1A_Aux_Only')
    """
    # 데이터 분할: 훈련 세트(80%)와 테스트 세트(20%)로 분할 (random_state=42로 재현성 확보)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 모델 목록 정의
    models = [
        ('Linear_Regression_Model', LinearRegression()), # 1. 일반 선형 회귀 모델
        ('Poly3_Model', Pipeline(steps=[('poly', PolynomialFeatures(3, include_bias=False)), ('linear', LinearRegression())])), # 2. 3차 다항식 회귀 모델 (Pipeline을 사용하여 다항 변환 후 선형 회귀 적용), 선형 회귀 적용
        ('Robust_Huber_Model', HuberRegressor(epsilon=1.35)), # 3. Huber 회귀 모델 (이상치에 강건한(Robust) 회귀, epsilon=1.35는 기본값 사용)
        ('CatBoost_Model', CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False, n_estimators=500)), # 4. CatBoost Regressor (부스팅 모델, 범주형 처리에 강점)
        ('LightGBM_Model', LGBMRegressor(random_state=42, verbose=-1, n_estimators=500)) # 5. LightGBM Regressor (부스팅 모델, 빠른 학습 속도)
    ]

    results = [] # 모델의 예측 결과(y_test, Prediction 등)를 저장할 리스트
    trained_models = [] # 학습된 모델 객체 정보(피처 중요도 시각화용)를 저장할 리스트
    
    print(f"   - {prefix} 데이터셋 학습 중 (피처 N={len(feature_names)})...")
    
    # 모델 목록을 순회하며 학습 및 평가 진행
    for model_name_suffix, model in models:
        full_model_name = f'{prefix}_{model_name_suffix}' # 실험 조건 접두사를 붙인 최종 모델 이름
        print(f"     > {full_model_name} 학습 중...")
        
        # 2. 파이프라인 구성: 전처리 + 모델
        # data_processing.py에서 생성된 전처리기(preprocessor)와 모델을 결합
        final_model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

        try:
            final_model.fit(X_train, y_train) # 파이프라인 학습: X_train이 preprocessor를 거친 후 regressor(model)에 의해 학습됨
            prediction = final_model.predict(X_test) # 예측: X_test가 preprocessor를 거친 후 예측값 생성
            
            # ⭐⭐⭐ 수정된 부분: 전처리 후의 최종 피처 이름 추출 ⭐⭐⭐
            # Pipeline 내부의 'preprocessor' 단계(ColumnTransformer)에 접근하여
            # 최종 변환된 피처 이름(예: cat__BMI_Class_Normal, num__SG)을 추출
            transformed_feature_names = list(final_model.named_steps['preprocessor'].get_feature_names_out())
            
            # 결과 저장 (평가 지표 계산에 사용됨)
            results.append({
                'Model': full_model_name,
                'y_test': y_test,
                'Prediction': prediction,
                'Feature_Names': transformed_feature_names # 변환된 피처 이름 저장
            })
            
            # 피처 중요도 분석을 위한 학습된 모델 객체 저장 (CatBoost, LightGBM 한정)
            if 'CatBoost' in full_model_name or 'LightGBM' in full_model_name:
                 model_object_for_viz = final_model.named_steps.get('regressor', final_model) # Pipeline 내부의 'regressor' 단계(모델 객체)를 추출
                 trained_models.append({
                    'Model_Name': full_model_name, 
                    'Model_Object': model_object_for_viz, # 학습이 완료된 CatBoost/LightGBM 객체 
                    'Feature_Names': transformed_feature_names # 변환된 피처 이름 저장
                })
            
        except Exception as e:
            print(f"       !!! {full_model_name} 학습 오류: {e}") # 학습 중 오류 발생 시 메시지 출력
            
    return results, trained_models # 모든 모델의 결과 리스트와 학습된 부스팅 모델 객체 리스트 반환