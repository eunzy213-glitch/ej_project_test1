# main.py
import pandas as pd  # 데이터 분석 및 조작을 위한 pandas 라이브러리 임포트
import numpy as np  # 수치 계산을 위한 numpy 라이브러리 임포트
import os # 파일 시스템 작업을 위한 os 모듈 임포트 (폴더 생성, 경로 설정 등)

# 모듈 임포트
from data_processing import load_data, preprocess_data, describe_data  # 데이터 처리 관련 함수 임포트
from model_training import train_all_models # 모델 학습 함수 임포트
from evaluation import calculate_metrics, calculate_clark_error_grid # 모델 평가 및 CEG 분석 함수 임포트
from visualization import (
    plot_residuals, # 잔차 분석 플롯 함수 (추세선 포함)
    plot_clark_error_grid, # Clark Error Grid 플롯 함수 (임상적 정확도)
    plot_confusion_matrix, # BG 카테고리 Confusion Matrix 플롯 함수
    plot_feature_importance, # 부스팅 모델 피처 중요도 플롯 함수
    plot_metrics_comparison_summary # R2 성능 통합 비교 플롯 함수 (3개 실험 비교)
)

# 파일 경로 및 설정 상수 정의
DATA_FILE = 'dataset.csv' # 원본 데이터 파일 이름
OUTPUT_DIR = 'model_outputs2' # 모든 결과물(CSV, PNG)을 저장할 폴더 이름

# ----------------------------------------------------
# 메인 실행 함수 정의
# ----------------------------------------------------
def main():
    # 결과 폴더 생성
    if not os.path.exists(OUTPUT_DIR): # 출력 폴더가 존재하지 않으면
        os.makedirs(OUTPUT_DIR) # 폴더를 생성

    # 0. 데이터 로드
    raw_data = load_data(DATA_FILE) # data_processing.py의 load_data 함수를 호출하여 원본 데이터 로드

    if raw_data.empty: # 데이터프레임이 비어있는지 확인 (로드 실패 시)
        print("데이터 로드 실패. 스크립트를 종료합니다.")
        return # 함수 종료

    # 1. 실험 조건 정의 및 데이터 준비
    all_cols = raw_data.columns.tolist() # 원본 데이터의 모든 컬럼 이름 리스트
    base_exclude = ['BG', 'Target_R', 'SG'] # 모든 실험에서 기본적으로 제외할 컬럼 리스트 (BG는 타겟)
    auxiliary_features = [col for col in all_cols if col not in base_exclude] # SG, BG, Target_R을 제외한 나머지 모든 피처를 보조 피처(auxiliary_features)로 정의

    numerical_cols = [col for col in raw_data.columns if raw_data[col].dtype in ['int64', 'float64', 'float32']] # 수치형 컬럼
    categorical_cols = [col for col in raw_data.columns if raw_data[col].dtype == 'object'] # 범주형 컬럼

    # 세 가지 실험 조건 정의 딕셔너리
    EXPERIMENT_SETUPS = {
        'T1-1A_Aux_Only': { # Case 1: 보조 피처(Aux) 단독 사용
            'features': auxiliary_features,
            'description': 'Case 1: 기존 제외 피처 (SG, Target_R 제외)'
        },
        'T1-1B_SG_Only': { # Case 2: SG 피처 단독 사용
            'features': ['SG'],
            'description': 'Case 2: SG만 단독 학습'
        },
        'T1-1C_SG_Aux': { # Case 3: SG와 보조 피처 결합 사용
            'features': ['SG'] + auxiliary_features,
            'description': 'Case 3: SG와 나머지 피처 같이 (Target_R 제외)'
        }
    }

    all_combined_results = [] # 모든 모델의 평가 결과(y_test, Prediction 등)를 모을 리스트
    all_combined_trained_models = [] #부스팅 모델의 학습된 객체 정보를 모을 리스트 (피처 중요도 시각화용)

    # 시작 메시지 출력
    print("\n=========================================================")
    print(f"  Tset 1-1: 세 가지 피처셋 통합 비교 분석 시작 (저장 폴더: {OUTPUT_DIR})")
    print("=========================================================")

    # 정의된 실험 조건 딕셔너리를 순회하며 학습 및 평가 진행
    for prefix, setup in EXPERIMENT_SETUPS.items():
        feature_cols = setup['features'] # 현재 실험에서 사용할 피처 리스트
        print(f"\n################ {setup['description']} 실험 시작 ################")
        
        if not feature_cols: # 사용할 피처 리스트가 비어있으면 경고 출력 후 건너뛰기
            print("   - 경고: 사용할 피처 목록이 비어있습니다. 이 실험을 건너뜀.")
            continue

        print(f"   - 사용 피처: {feature_cols}")

        # 1. 데이터 전처리 (이상치 제거 및 전처리 파이프라인 구축)
        X_processed, y_processed, preprocessor = preprocess_data(
            raw_data.copy(),  # 원본 데이터를 복사하여 전달 (LOWESS/Isolation Forest로 이상치 제거)
            feature_cols=feature_cols, # 이 실험에 사용할 최종 피처 목록
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols
        )
        # 전처리 후 데이터의 통계 정보 출력
        describe_data(X_processed.join(y_processed), f"[{prefix}] 전처리 후 데이터 통계 (N={len(X_processed)})")

        # 3. 모델 학습 및 결과 취합
        results, trained_models = train_all_models(
            X=X_processed, # 이상치가 제거된 피처 데이터
            y=y_processed, # 이상치가 제거된 타겟 데이터
            preprocessor=preprocessor,  # 전처리 정의 객체 (파이프라인에 사용)
            feature_names=feature_cols, # 원본 피처 이름 리스트
            prefix=prefix # 모델 이름 접두사
        )
        # 각 실험의 결과를 전체 결과 리스트에 합치기
        all_combined_results.extend(results)
        all_combined_trained_models.extend(trained_models)
        
        print(f"################ {setup['description']} 실험 완료 ################")
        print("---------------------------------------------------------")


    # ----------------------------------------------------
    # 4. 최종 결과 분석 및 시각화 (통합)
    # ----------------------------------------------------
    print("\n--- [4] 최종 결과 분석 및 시각화 시작 ---")

    # 4-1. 성능 지표 계산 및 출력
    performance_metrics = calculate_metrics(all_combined_results)
    print("\n--- [4-1] 모델별 최종 성능 지표 (R, R2, RMSE, MAE) 정량 비교 ---")
    print(performance_metrics.sort_values(by='R2', ascending=False).to_markdown(index=False)) # R2 기준으로 내림차순 정렬하여 마크다운 표 형태로 콘솔에 출력

    # ⭐⭐⭐ 추가된 부분: 성능 지표 CSV 파일 저장 ⭐⭐⭐
    results_file_path = os.path.join(OUTPUT_DIR, 'model_performance_summary.csv') # 저장할 파일 경로 설정
    performance_metrics.to_csv(results_file_path, index=False) # 데이터프레임을 CSV 파일로 저장 (인덱스 제외)
    print(f"\n   - 모델별 최종 성능 지표가 다음 위치에 저장되었습니다: {results_file_path}")
    
    # 4-2. CEG 분석 및 시각화 파일 생성
    for result in all_combined_results: # 모든 모델 결과에 대해 반복
        calculate_clark_error_grid(result) # CEG 분석 수행 (결과 딕셔너리에 CEG 비율 추가)
        plot_residuals(result, OUTPUT_DIR) # 잔차 분석 플롯 이미지 저장 (빨간 추세선 포함)
        plot_clark_error_grid(result, OUTPUT_DIR) # CEG Zone 분포 플롯 이미지 저장
        plot_confusion_matrix(result, OUTPUT_DIR) # BG 카테고리 Confusion Matrix 이미지 저장

    # 5. 최종 시각화 요약
    plot_metrics_comparison_summary(performance_metrics, OUTPUT_DIR) # 세 가지 실험 조건의 R2 통합 비교 플롯 이미지 저장

    # 5-2. 부스팅 모델 피처 중요도 플롯
    for model_info in all_combined_trained_models: # 학습된 부스팅 모델 정보 리스트 순회
        if 'CatBoost' in model_info['Model_Name'] or 'LightGBM' in model_info['Model_Name']: # CatBoost 또는 LightGBM 모델인지 확인
            final_model_object = model_info['Model_Object'] # 학습된 모델 객체 추출
            plot_feature_importance(
                final_model_object, 
                model_info['Model_Name'], 
                model_info['Feature_Names'], # model_training에서 변환된 피처 이름 리스트 (cat__, num__ 접두사 포함)
                OUTPUT_DIR
            )
    # 최종 완료 메시지
    print("\n=========================================================")
    print(f"  Tset 1-1 통합 실험 및 분석 완료! 결과는 {OUTPUT_DIR} 폴더에 저장되었습니다.")
    print("=========================================================")
# 스크립트가 직접 실행될 때 main 함수 호출
if __name__ == "__main__":
    main()