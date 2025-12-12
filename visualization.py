# visualization.py

import pandas as pd # 데이터프레임 조작을 위한 pandas 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트
import matplotlib.pyplot as plt # 기본 그래프 및 시각화를 위한 matplotlib 임포트
import seaborn as sns # Matplotlib 기반의 통계 데이터 시각화 라이브러리 임포트 (더 예쁜 그래프)
from sklearn.metrics import confusion_matrix # Confusion Matrix 계산 함수 임포트
import os # 파일 경로 및 폴더 생성을 위한 os 모듈 임포트
import matplotlib.font_manager as fm # Matplotlib의 폰트 설정을 위한 모듈 임포트

# 폰트 설정 (한글 처리를 위한 표준 로직)
font_name = 'sans-serif' # 기본(대체) 폰트 이름 설정
korean_font_found = False # 한글 폰트 발견 여부 플래그 초기화

# 시스템 폰트 목록을 순회하며 사용 가능한 한글 폰트를 탐색
for font in fm.fontManager.ttflist:
    if 'Nanum' in font.name or 'Malgun Gothic' in font.name or 'Noto Sans CJK' in font.name: # 'Nanum' (나눔), 'Malgun Gothic' (맑은 고딕), 'Noto Sans CJK' 중 하나라도 포함된 폰트를 찾으면
        font_name = font.name # 해당 폰트 이름으로 설정
        korean_font_found = True # 발견 플래그를 True로 변경
        break # 폰트를 찾았으므로 루프 종료
        
plt.rcParams['font.family'] = font_name  # Matplotlib의 기본 폰트를 찾은 한글 폰트로 설정
plt.rcParams['axes.unicode_minus'] = False # 유니코드 마이너스 기호 사용 설정 (마이너스 부호 깨짐 방지)

# ------------------------------------------------------------------
# 1. R2 비교 (통합 플롯)
# ------------------------------------------------------------------
def plot_metrics_comparison_summary(performance_df: pd.DataFrame, output_dir: str):
    """세 가지 피처셋 간의 R2 성능 비교 시각화"""
    df = performance_df.copy() # 원본 데이터프레임 복사
    
    df['Feature_Set'] = df['Model'].apply(lambda x: '_'.join(x.split('_')[:3])) # 'Model' 이름(예: 'T1-1A_Aux_Only_Linear_Regression_Model')에서 'Feature_Set' 추출 (T1-1A_Aux_Only)
    df['Model_Type'] = df['Model'].apply(lambda x: '_'.join(x.split('_')[3:])) # 'Model' 이름에서 'Model_Type' 추출 (Linear_Regression_Model)
    
    mapping = { # Feature_Set의 약어를 한글 라벨로 매핑
        'T1-1A_Aux_Only': 'Case 1: Aux 단독',
        'T1-1B_SG_Only': 'Case 2: SG 단독',
        'T1-1C_SG_Aux': 'Case 3: SG + Aux'
    }
    df['Feature_Set_Label'] = df['Feature_Set'].map(mapping).fillna(df['Feature_Set']) # 매핑 적용. 매핑에 없는 값은 기존 Feature_Set을 사용

    fig, ax = plt.subplots(figsize=(15, 8)) # Matplotlib figure와 axes 객체 생성
    sns.barplot(x='Model_Type', y='R2', hue='Feature_Set_Label', data=df, ax=ax, palette='viridis') # 바 플롯 생성: x축=모델 종류, y축=R2, hue(범례)=피처셋 라벨
    
    ax.set_title('Tset 1-1 통합: 피처셋별 R2 성능 비교', fontsize=16) # 제목 설정
    ax.set_xlabel('모델 종류') # X축 라벨 설정
    ax.set_ylabel('R2 Score') # Y축 라벨 설정
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8) # R2=0 기준선 추가
    ax.legend(title='Feature Set') # 범례 제목 설정
    plt.xticks(rotation=45, ha='right') # X축 눈금 라벨 45도 회전 및 오른쪽 정렬
    plt.tight_layout() # 플롯 요소가 잘리지 않도록 조정
    fig.savefig(os.path.join(output_dir, 'T1_Comparison_R2_Summary.png')) # 파일을 지정된 출력 디렉토리에 PNG 형식으로 저장
    plt.close(fig) # 메모리 절약을 위해 figure 닫기

# ------------------------------------------------------------------
# 2. 잔차 분석 플롯 (LOWESS 추세선 추가)
# ------------------------------------------------------------------
def plot_residuals(result: dict, output_dir: str):
    """잔차 분석 플롯 (예측값 vs 잔차) - LOWESS 추세선 추가"""
    y_test = result['y_test'] # 실제 타겟 값 추출
    y_pred = result['Prediction'] # 예측 값 추출
    residuals = y_test - y_pred # 잔차 계산 (실제값 - 예측값)
    model_name = result['Model'] # 모델 이름 추출
    
    fig, ax = plt.subplots(figsize=(8, 5)) # Matplotlib figure와 axes 객체 생성
    
    # 1. 산점도 플롯: 예측값(x축)과 잔차(y축)를 점으로 표시
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6, label='잔차')
    
    # 2. LOWESS 추세선 추가 (빨간 선)
    # lowess=True를 설정하여 Locally Weighted Scatterplot Smoothing (LOWESS) 추세선을 그립니다.
    sns.regplot(x=y_pred, y=residuals, scatter=False, lowess=True, 
                line_kws={'color': 'red', 'lw': 2, 'label': 'LOWESS 추세선'}, ax=ax)
    
    # 3. 기준선 (0) 추가 (파란 점선)
    ax.axhline(0, color='blue', linestyle='--', linewidth=1, label='잔차 기준선 (0)')
    
    ax.set_title(f'{model_name} - 잔차 분석') # 제목 설정
    ax.set_xlabel('예측값') # X축 라벨 설정
    ax.set_ylabel('잔차 (실제 - 예측)') # Y축 라벨 설정
    ax.legend() # 범례 표시
    plt.tight_layout() # 플롯 요소가 잘리지 않도록 조정
    fig.savefig(os.path.join(output_dir, f'Residuals_{model_name}.png')) # 파일을 지정된 출력 디렉토리에 저장
    plt.close(fig) # 메모리 절약을 위해 figure 닫기

# ------------------------------------------------------------------
# 3. Clark Error Grid (CEG) 플롯
# ------------------------------------------------------------------
def plot_clark_error_grid(result: dict, output_dir: str):
    """CEG 플롯 (CEG Zone 비율 막대 그래프)"""
    if 'CEG' not in result: # CEG 분석 결과가 없으면 함수 종료
        return
        
    model_name = result['Model'] # 모델 이름 추출
    ceg_counts = pd.Series(result['CEG']) # 결과 딕셔너리에서 CEG Zone별 비율(%) 추출
    ceg_counts = ceg_counts.reindex(['A', 'B', 'C', 'D', 'E'], fill_value=0) # A, B, C, D, E 순서대로 인덱스를 재정렬하고, 값이 없는 Zone은 0으로 채움
    
    fig, ax = plt.subplots(figsize=(6, 5)) # Matplotlib figure와 axes 객체 생
    ceg_counts.plot(kind='bar', ax=ax, color=['green', 'blue', 'orange', 'red', 'darkred']) # 막대 그래프 생성: CEG Zone별 비율 시각화
    
    ax.set_title(f'{model_name} - Clark Error Grid Zone 분포', fontsize=12) # 제목 설정
    ax.set_ylabel('데이터 비율 (%)') # Y축 라벨 설정
    ax.set_xlabel('CEG Zone') # X축 라벨 설정
    plt.xticks(rotation=0) # X축 눈금 라벨 회전 방지
    
    for container in ax.containers: # 막대 위에 실제 비율(%) 텍스트 표시
        ax.bar_label(container, fmt='%.1f%%') # 소수점 첫째 자리까지 표시
        
    plt.tight_layout() # 플롯 요소가 잘리지 않도록 조정
    fig.savefig(os.path.join(output_dir, f'CEG_{model_name}.png')) # 파일을 지정된 출력 디렉토리에 저장
    plt.close(fig) # 메모리 절약을 위해 figure 닫기

# ------------------------------------------------------------------
# 4. Confusion Matrix (BG 카테고리화 필요)
# ------------------------------------------------------------------
def plot_confusion_matrix(result: dict, output_dir: str):
    """BG 값(저혈당, 정상, 고혈당) 기반 Confusion Matrix"""
    y_test = result['y_test'] # 실제 타겟 값 추출
    y_pred = result['Prediction'] # 예측 값 추출
    model_name = result['Model'] # 모델 이름 추출
    
    def categorize_bg(bg):
        """혈당 값을 임상적으로 카테고리화"""
        if bg < 70:
            return 'Hypoglycemia (<70)' # 저혈당
        elif bg > 180:
            return 'Hyperglycemia (>180)' # 고혈당
        else:
            return 'Normal (70-180)' # 정상 혈당

    y_true_cat = y_test.apply(categorize_bg) # 실제 값과 예측 값을 카테고리화 함수를 사용하여 변환
    y_pred_cat = pd.Series(y_pred).apply(categorize_bg)
    
    labels = ['Hypoglycemia (<70)', 'Normal (70-180)', 'Hyperglycemia (>180)'] # Confusion Matrix의 순서와 라벨 정의
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels) # 카테고리화된 값으로 Confusion Matrix 계산
    cm_df = pd.DataFrame(cm, index=labels, columns=labels) # 시각화를 위해 계산된 매트릭스를 데이터프레임으로 변환 (인덱스/컬럼에 라벨 지정)
    
    fig, ax = plt.subplots(figsize=(8, 7)) # Matplotlib figure와 axes 객체 생성
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax) # Heatmap 생성: Confusion Matrix를 시각화 (annot=True: 셀에 값 표시, fmt='d': 정수 형식)
    ax.set_title(f'{model_name} - BG 카테고리 Confusion Matrix') # 제목 설정
    ax.set_ylabel('실제 BG 카테고리') # Y축 라벨 설정
    ax.set_xlabel('예측 BG 카테고리') # X축 라벨 설정
    plt.tight_layout() # 플롯 요소가 잘리지 않도록 조정
    fig.savefig(os.path.join(output_dir, f'ConfMatrix_{model_name}.png')) # 파일을 지정된 출력 디렉토리에 저장
    plt.close(fig) # 메모리 절약을 위해 figure 닫기

# ------------------------------------------------------------------
# 5. 부스팅 모델 피처 중요도 플롯
# ------------------------------------------------------------------
def plot_feature_importance(model, model_name: str, feature_names: list, output_dir: str):
    """LightGBM 또는 CatBoost 모델의 피처 중요도를 시각화하고 저장합니다."""
    
    if 'CatBoost' in model_name: # CatBoost 모델 객체에서 피처 중요도 추출
        feature_importances = model.get_feature_importance()
    elif 'LightGBM' in model_name: # LightGBM 모델 객체에서 피처 중요도 추출 (속성 사용)
        feature_importances = model.feature_importances_
    else: # 지원하지 않는 모델의 경우 함수 종료
        return

    # 중요도 값 리스트와 피처 이름 리스트의 길이가 일치하는지 확인 (전처리 오류 방지)
    if len(feature_importances) != len(feature_names):
        print(f"   - [시각화] {model_name}의 중요도 개수({len(feature_importances)})와 입력 피처명 개수({len(feature_names)}) 불일치. 시각화 건너뜀.")
        return

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})  # 피처 이름과 중요도 값을 DataFrame으로 결합
    importance_df = importance_df.sort_values(by='Importance', ascending=False) # 중요도 기준 내림차순 정렬
    importance_df = importance_df[importance_df['Importance'] > 0] # 중요도 값이 0보다 큰 피처만 선택 (유의미한 피처만 시각화)
    
    if importance_df.empty: 
        print(f"   - [시각화] {model_name}은 유의미한 피처 중요도를 생성하지 못했습니다.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6)) # Matplotlib figure와 axes 객체 생성
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax) # 바 플롯 생성: 중요도(x축)와 피처 이름(y축)을 막대로 표시
    ax.set_title(f'{model_name} - 피처 중요도') # 제목 설정
    plt.tight_layout() # 플롯 요소가 잘리지 않도록 조정
    fig.savefig(os.path.join(output_dir, f'FeatureImportance_{model_name}.png')) # 파일을 지정된 출력 디렉토리에 저장
    plt.close(fig) # 메모리 절약을 위해 figure 닫기