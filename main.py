# 파이프라인의 메인 스크립트

import pandas as pd # 데이터 분석 및 조작을 위한 pandas 라이브러리 임포트
import numpy as np # 수치 계산을 위한 numpy 라이브러리 임포트
import os # 파일 시스템 작업을 위한 os 모듈 임포트(폴더 생성, 경로 설정 등)

# 모든 모듈 임포트: 별도로 정의된 함수들을 가져옴
from data_processing import load_and_preprocess_data # 데이터 로드 및 전처리(이상치 제거 포함) 함수
from model_training import train_all_models # 모든 회귀 모델 학습 함수
from evaluation import calculate_metrics, calculate_clark_error_grid # 모델 평가 지표 및 CEG 분석 함수
from visualization import (
    plot_model_comparison, # R2 점수 비교 막대 그래프
    plot_residuals, # 잔차 플롯
    plot_clark_error_grid, # Clark Error Grid 플롯
    plot_confusion_matrix, # BG 카테고리 기반 Confusion Matrix
    plot_feature_importance, # 부스팅 모델 피처 중요도 플롯
    plot_outlier_removal_comparison, # 이상치 제거 전후 비교 산점도 (제거된 점 표시)
    plot_raw_vs_processed_scatterplot # 전처리 전후 데이터 분포 비교 산점도
)

# 파일 경로 및 설정 상수 정의
DATA_FILE = 'dataset.csv' # 원본 데이터 파일 이름
OUTPUT_DIR = 'model_outputs' # 모든 결과물(CSV, PNG)을 저장할 폴더 이름
PROCESSED_DATA_FILE = os.path.join(OUTPUT_DIR, 'preprocessed_data.csv') # 전처리 완료된 데이터가 저장될 최종 경로

def main():
    # 1. 데이터 로딩 및 전처리 (LOWESS 기준 피처: 'SG')
    print("✨ 데이터 로딩 및 전처리 시작...")
    # data_processing 모듈의 함수 호출: 원본 데이터와 전처리된 데이터를 모두 반환
    raw_data, preprocessed_data = load_and_preprocess_data(DATA_FILE, lowess_feature='SG')
    
    # 데이터 로딩에 실패했거나 전처리할 데이터가 없는 경우 함수 종료
    if raw_data is None:
        return

    # -----------------------------------------------------------------
    # 전처리 완료된 데이터를 CSV 파일로 저장
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 데이터 저장 시도 (오류 방지 try-except 블록 사용)
    try:
        # preprocessed_data를 CSV 파일로 저장 (index=False는 데이터프레임의 기본 인덱스를 저장하지 않음)
        preprocessed_data.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"\n✅ 전처리 완료 데이터가 '{PROCESSED_DATA_FILE}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 전처리 데이터 저장 중 오류 발생: {e}")
    # -----------------------------------------------------------------


    # -----------------------------------------------------------------
    # 1. 전처리 전/후 데이터 산점도 비교 시각화 (00_a번)
    # raw_data와 preprocessed_data를 산점도로 비교하여 이상치 제거 효과 시각화
    plot_raw_vs_processed_scatterplot(raw_data, preprocessed_data, OUTPUT_DIR, feature='SG')
    # -----------------------------------------------------------------

    # 1-1. 이상치 제거 전후 시각화 (기존 00번 - 제거된 이상치를 X표시)
    plot_outlier_removal_comparison(raw_data, preprocessed_data, OUTPUT_DIR, feature='SG')

    # 2. 데이터 분할 및 피처/타겟 정의 (BG가 타겟 변수)
    # 원본 데이터에서 'BG' (Blood Glucose)를 제외하고 피처(X)와 타겟(y) 분리
    X_raw = raw_data.drop(columns=['BG'])
    y_raw = raw_data['BG']
    # 전처리된 데이터에서 'BG'를 제외하고 피처(X)와 타겟(y) 분리
    X_processed = preprocessed_data.drop(columns=['BG'])
    y_processed = preprocessed_data['BG']
    
    # 3. 모델 학습
    print("\n✨ 모델 학습 시작 (Linear, Poly-3, CatBoost, LightGBM, Huber)...")
    
    # 전처리 전 데이터로 모델 학습 (Raw 데이터셋)
    raw_results, raw_models, raw_feature_names = train_all_models(raw_data, X_raw, y_raw, 'Raw')
    
    # 전처리 후 데이터로 모델 학습 (Processed 데이터셋)
    # 이 과정에서 'SG'와 'Target_R' 피처가 제외된 최종 X 데이터셋이 생성됨
    processed_results, processed_models, processed_feature_names = train_all_models(preprocessed_data, X_processed, y_processed, 'Processed')
    
    # 모든 학습 결과를 하나의 리스트로 통합
    all_results = raw_results + processed_results
    # 모든 학습된 모델 객체를 하나의 리스트로 통합
    all_models = raw_models + processed_models
    
    # 4. 모델 성능 평가
    print("\n✨ 모델 적합도 및 성능 평가 시작...")
    
    # 4-1. 모든 모델의 성능 지표 계산 (R2, RMSE, MAE)
    performance_metrics = calculate_metrics(all_results)
    print("\n--- [3] 모델별 성능 지표 정량 비교 (R2, RMSE, MAE) ---")
    # R2 기준 내림차순으로 정렬하여 마크다운 표 형태로 콘솔에 출력
    print(performance_metrics.sort_values(by='R2', ascending=False).to_markdown(index=False))
    
    # 4-2. Clark Error Grid 및 기타 시각화 준비
    # CEG 분석 결과(Area A, B, C, D, E 비율)를 각 결과 딕셔너리에 추가
    for result in all_results:
        calculate_clark_error_grid(result) 

    # 5. 시각화
    print("\n✨ 시각화 생성 및 저장 시작...")
    
    # R2 비교 (통합 플롯) 시각화 저장
    plot_model_comparison(performance_metrics, OUTPUT_DIR)
    
    # 모델별 진단 플롯 시각화 저장
    for result in all_results:
        plot_residuals(result, OUTPUT_DIR) # 잔차 분석
        plot_clark_error_grid(result, OUTPUT_DIR) # Clark Error Grid
        plot_confusion_matrix(result, OUTPUT_DIR) # BG 카테고리 Confusion Matrix

    # 부스팅 모델 피처 중요도 플롯 (SG, Target_R 제외된 피처만 사용) 시각화 저장
    for model_name, model, feature_names in all_models:
        # CatBoost와 LightGBM 모델만 피처 중요도 분석 실행
        if 'CatBoost' in model_name or 'LightGBM' in model_name:
            # 해당 모델이 Raw 데이터로 학습되었는지 Processed 데이터로 학습되었는지에 따라 피처 이름 목록 선택
            current_feature_names = raw_feature_names if 'Raw' in model_name else processed_feature_names
            # 피처 중요도 시각화 저장
            plot_feature_importance(model, model_name, current_feature_names, OUTPUT_DIR)

    print("\n✅ 모든 시각화 및 분석이 완료되었습니다. 'model_outputs' 폴더를 확인해 주세요.")

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main 함수 호출
    main()