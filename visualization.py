import pandas as pd # 데이터 조작 및 관리를 위한 pandas 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트
import matplotlib.pyplot as plt # 기본 그래프 및 시각화를 위한 matplotlib 임포트
import seaborn as sns # Matplotlib 기반의 통계 데이터 시각화 라이브러리 (더 예쁜 그래프)
from sklearn.metrics import confusion_matrix # Confusion Matrix 계산 함수 임포트
import os # 파일 경로 및 폴더 생성을 위한 os 모듈 임포트
import matplotlib.font_manager as fm # Matplotlib의 폰트 설정을 위한 모듈 임포트

# =======================================================
# 폰트 설정
# =======================================================
font_name = 'sans-serif' # 기본 폰트 이름 설정
korean_font_found = False # 한글 폰트 발견 여부 플래그

# 시스템 폰트 목록을 순회하며 한글 폰트를 찾음
for font in fm.fontManager.ttflist:
    # 나눔, 맑은 고딕, Noto Sans CJK 중 하나라도 포함된 폰트를 찾으면
    if 'Nanum' in font.name or 'Malgun' in font.name or 'Noto Sans CJK' in font.name:
        font_name = font.name # 해당 폰트 이름으로 설정
        korean_font_found = True
        break # 폰트를 찾았으므로 루프 종료
        
plt.rcParams['font.family'] = font_name # Matplotlib의 기본 폰트를 찾은 한글 폰트로 설정
plt.rcParams['axes.unicode_minus'] = False # 유니코드 마이너스 기호 사용 설정 (마이너스 부호 깨짐 방지)
# =======================================================


def save_plot(plt, filename: str, output_dir: str):
    """matplotlib 그림을 지정된 경로에 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 출력 디렉토리가 없으면 생성
    full_path = os.path.join(output_dir, filename) # 전체 파일 저장 경로 생성
    plt.savefig(full_path) # 현재 활성화된 Matplotlib 그림을 PNG 파일로 저장
    plt.close() # 메모리 관리를 위해 현재 그림 닫기

# -------------------------------------------------------
# 전처리 전후 데이터 비교 산점도 (00_a번)
# -------------------------------------------------------
def plot_raw_vs_processed_scatterplot(raw_data: pd.DataFrame, processed_data: pd.DataFrame, output_dir: str, feature: str = 'SG'):
    """
    주요 피처(SG)와 BG를 기준으로 전처리 전/후 데이터의 분포를 산점도로 비교합니다.
    """
    plt.figure(figsize=(12, 6)) # 플롯 크기 설정

    # Raw 데이터 (밝은 회색): 원본 데이터 전체를 투명도 0.2로 표시
    sns.scatterplot(x=raw_data[feature], y=raw_data['BG'], 
                    alpha=0.2, label='Raw Data (Original)', color='gray')

    # Processed 데이터 (파란색): 이상치가 제거된 데이터를 투명도 0.6으로 표시 (이상치 제거 효과 시각화)
    sns.scatterplot(x=processed_data[feature], y=processed_data['BG'], 
                    alpha=0.6, label='Processed Data (Outliers Removed)', color='blue')
    
    plt.title(f'📈 BG vs {feature} - 전처리 전/후 데이터 분포 비교') # 제목 설정
    plt.xlabel(f'{feature} 값 (Salivary Glucose)') # X축 레이블 설정
    plt.ylabel('BG 값 (Blood Glucose)') # Y축 레이블 설정
    plt.grid(True, linestyle=':', alpha=0.5) # 그리드 라인 추가
    plt.legend() # 범례 표시

    # 그림 저장 함수 호출
    save_plot(plt, f'00_a_Raw_vs_Processed_Scatterplot_{feature}.png', output_dir)
    print(f"   - [시각화] 전처리 전/후 데이터 분포 비교 플롯 저장 완료 (기준 피처: {feature}).")


# -------------------------------------------------------
# 00. 이상치 제거 전후 비교 시각화 (기준 피처: SG)
# -------------------------------------------------------
def plot_outlier_removal_comparison(raw_data: pd.DataFrame, processed_data: pd.DataFrame, output_dir: str, feature: str = 'SG'):
    """주요 피처(SG)와 BG를 기준으로 이상치 제거 전후의 데이터를 시각적으로 비교합니다."""
    plt.figure(figsize=(12, 6))

    # Raw 데이터와 Processed 데이터의 인덱스를 비교하여 제거된 데이터의 인덱스를 찾음
    removed_indices = raw_data.index.difference(processed_data.index)
    removed_data = raw_data.loc[removed_indices] # 제거된 데이터만 추출

    # 1. Raw 데이터 전체 (배경)
    sns.scatterplot(x=raw_data[feature], y=raw_data['BG'], 
                    alpha=0.3, label='Raw Data (All Points)', color='gray')

    # 2. Processed 데이터 (유지된 점)
    sns.scatterplot(x=processed_data[feature], y=processed_data['BG'], 
                    alpha=0.6, label='Processed Data (Kept Points)', color='blue')
    
    # 3. 제거된 이상치 (X 마커로 강조)
    sns.scatterplot(x=removed_data[feature], y=removed_data['BG'], 
                    marker='X', s=100, color='red', label='Removed Outliers', linewidth=1) # 마커를 X로, 크기를 크게, 빨간색으로 표시
    
    plt.title(f'🧪 {feature} vs BG - 이상치 제거 전후 비교 (LOWESS+Isolation Forest)')
    plt.xlabel(f'{feature} 값 (Salivary Glucose)')
    plt.ylabel('BG 값 (Blood Glucose)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    save_plot(plt, f'00_Outlier_Removal_Comparison_{feature}.png', output_dir)
    print(f"   - [시각화] 이상치 제거 전후 비교 플롯 저장 완료 (기준 피처: {feature}).")

# -------------------------------------------------------
# 01. 모델 비교 막대 그래프
# -------------------------------------------------------
def plot_model_comparison(performance_df: pd.DataFrame, output_dir: str):
    """모델별 R2 점수를 막대 그래프로 비교하고 저장합니다. (모든 모델 명확히 표시)"""
    # R2 점수 기준 내림차순으로 데이터 정렬 (가장 좋은 모델이 왼쪽)
    performance_df = performance_df.sort_values(by='R2', ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Raw/Processed 여부를 구분하는 'Type' 컬럼 생성 (색상 구분을 위해)
    performance_df['Type'] = performance_df['Model'].apply(lambda x: 'Processed' if 'Processed' in x else 'Raw')
    
    # 막대 그래프 생성 (Type에 따라 색상 구분, dodge=False로 그룹을 묶지 않고 개별 막대로 표시)
    sns.barplot(x='Model', y='R2', data=performance_df, 
                hue='Type', palette={'Processed': '#1f77b4', 'Raw': '#ff7f0e'}, 
                dodge=False)
    
    # R2 = 0 인 지점에 빨간색 점선 기준선 추가 (0 미만은 모델이 평균 예측보다 나쁨을 의미)
    plt.axhline(0, color='red', linestyle='--', linewidth=1) 
    
    plt.title('🥇 모델별 $R^2$ 점수 비교 (Raw vs Processed)', fontsize=16)
    plt.xlabel('모델')
    plt.ylabel('$R^2$ Score')
    plt.xticks(rotation=45, ha='right', fontsize=9) # X축 레이블 45도 회전 및 글꼴 크기 조정
    plt.tight_layout() # 그래프 요소가 잘리지 않도록 레이아웃 자동 조정
    save_plot(plt, '01_R2_Model_Comparison.png', output_dir)
    print("   - [시각화] R2 비교 플롯 저장 완료.")

# -------------------------------------------------------
# 02. 잔차 분석
# -------------------------------------------------------
def plot_residuals(result: dict, output_dir: str):
    """잔차 분석 (Residual Plot)을 시각화하고 잔차 추세선(LOWESS)을 추가하여 저장합니다."""
    model_name = result['Model']
    y_test = result['y_test']
    prediction = result['Prediction']
    
    plt.figure(figsize=(10, 6))
    residuals = y_test - prediction # 잔차 계산: 실제값 - 예측값
    
    # 잔차 플롯 생성 (x축: 예측값, y축: 잔차)
    # lowess=True: 잔차의 추세선(LOWESS 스무딩)을 빨간색으로 추가
    sns.residplot(x=prediction, y=residuals, 
                  lowess=True, 
                  scatter_kws={'alpha': 0.6}, # 산점도 점의 투명도 설정
                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}) # 추세선 스타일 설정
    
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1) # 잔차 0 기준선 추가
    plt.title(f'📉 {model_name} - 잔차 분석 (Residual Plot with LOWESS Trend)')
    plt.xlabel('예측값 (Predicted BG)')
    plt.ylabel('잔차 (Residuals: Actual - Predicted)')
    plt.grid(True, linestyle=':', alpha=0.6)
    save_plot(plt, f'02_Residual_Plot_{model_name}.png', output_dir)
    print(f"   - [시각화] {model_name} 잔차 플롯 저장 완료 (LOWESS 추세선 포함).")

# -------------------------------------------------------
# 03. Clark Error Grid (CEG) 영역 표시 로직 추가
# -------------------------------------------------------
def plot_clark_error_grid(result: dict, output_dir: str):
    """Clark Error Grid (CEG) 플롯에 영역 경계선을 추가하여 시각화합니다."""
    model_name = result['Model']
    y_true = result['y_test'].values # 실제값 (X축)
    y_pred = result['Prediction'] # 예측값 (Y축)
    
    plt.figure(figsize=(8, 8))
    
    # 1. 45도 기준선 (y=x, 이상적인 예측)
    plt.plot([0, 400], [0, 400], 'k-', lw=1, alpha=0.5) 
    
    x_range = np.arange(0, 401) # 0부터 400까지의 X축 범위 생성
    
    # 2. Area A/B 경계선 (임상적으로 허용 가능한 영역)
    # x <= 70: y = x +/- 20 (저혈당 범위 오차 허용)
    # x > 70: y = x * 1.2 또는 x * 0.8 (나머지 범위 오차 허용)
    y_ab_upper = np.where(x_range <= 70, x_range + 20, x_range * 1.2) 
    y_ab_lower = np.where(x_range <= 70, x_range - 20, x_range * 0.8)
    
    plt.plot(x_range, y_ab_upper, 'g--', lw=1.5, label='Area A/B Boundary') # 상한선 (녹색 점선)
    plt.plot(x_range, y_ab_lower, 'g--', lw=1.5) # 하한선 (녹색 점선)

    # 3. Area C/D 경계선 (±50% 오차선)
    plt.plot(x_range, x_range * 0.5, 'y:', lw=1) # 50% 하한선 (노란색 점선)
    plt.plot(x_range, x_range * 1.5, 'y:', lw=1) # 150% 상한선 (노란색 점선)
    
    # 실제 예측 데이터 포인트
    plt.scatter(y_true, y_pred, alpha=0.7, s=15, label='Predictions')
    
    plt.title(f'🧪 {model_name} - Clark Error Grid (CEG) 분석')
    plt.xlabel('실제 BG (mg/dL)')
    plt.ylabel('예측 BG (mg/dL)')
    plt.xlim(0, 350) # X축 범위 설정
    plt.ylim(0, 350) # Y축 범위 설정
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_plot(plt, f'03_Clark_Error_Grid_{model_name}.png', output_dir)
    print(f"   - [시각화] {model_name} CEG 플롯 저장 완료 (경계선 추가).")

# -------------------------------------------------------
# 04. Confusion Matrix
# -------------------------------------------------------
def plot_confusion_matrix(result: dict, output_dir: str):
    """BG 카테고리 기반 Confusion Matrix를 시각화하고 저장합니다."""
    model_name = result['Model']
    y_true = result['y_test']
    y_pred = result['Prediction']
    
    def categorize_bg(bg):
        """BG 값을 임상 카테고리로 분류하는 헬퍼 함수"""
        if bg < 70: return 'Hypo' # 저혈당
        elif bg <= 180: return 'Normal' # 정상 (70 ~ 180)
        else: return 'Hyper' # 고혈당 (180 초과)

    # 실제값과 예측값을 카테고리화
    y_true_cat = y_true.apply(categorize_bg)
    y_pred_cat = y_pred.apply(categorize_bg)
    
    labels = ['Hypo', 'Normal', 'Hyper'] # Confusion Matrix의 순서 정의
    # Confusion Matrix 계산
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    # 계산된 행렬을 pandas DataFrame으로 변환 (시각화 용이)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(8, 6))
    # 히트맵 시각화
    # annot=True: 셀에 값 표시, fmt='d': 정수 형식, cmap='Blues': 파란색 계열 색상 사용
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'📊 {model_name} - BG 카테고리 Confusion Matrix')
    plt.xlabel('예측 카테고리')
    plt.ylabel('실제 카테고리')
    plt.tight_layout()
    
    save_plot(plt, f'04_Confusion_Matrix_{model_name}.png', output_dir)
    print(f"   - [시각화] {model_name} Confusion Matrix 저장 완료.")

# -------------------------------------------------------
# 05. 피처 중요도 분석 (Target_R, SG 제외된 피처만 사용)
# -------------------------------------------------------
def plot_feature_importance(model, model_name: str, feature_names: list, output_dir: str):
    """LightGBM 또는 CatBoost 모델의 피처 중요도를 시각화하고 저장합니다."""
    # CatBoost 모델의 피처 중요도 추출
    if 'CatBoost' in model_name:
        feature_importances = model.get_feature_importance()
    # LightGBM 모델의 피처 중요도 추출
    elif 'LightGBM' in model_name:
        feature_importances = model.feature_importances_
    else:
        # 지원하지 않는 모델의 경우 메시지 출력 후 종료
        print(f"   - [시각화] {model_name}은 피처 중요도를 지원하지 않아 건너뜁니다.")
        return

    # 피처 이름과 중요도 값을 DataFrame으로 변환 및 중요도 기준 내림차순 정렬
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # 중요도 값이 0보다 큰 피처만 남겨 유의미한 피처만 시각화
    importance_df = importance_df[importance_df['Importance'] > 0]

    if importance_df.empty:
        # 중요도 0 초과 피처가 없는 경우 메시지 출력 후 종료
        print(f"   - [시각화] {model_name}은 Target_R, SG 제외 후 유의미한 피처 중요도를 생성하지 못했습니다.")
        return

    # 그래프 크기 설정 (피처 개수에 따라 Y축 높이 동적으로 조정)
    plt.figure(figsize=(10, max(5, len(importance_df) * 0.5)))
    # 막대 그래프 생성 (X축: 중요도, Y축: 피처 이름)
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'✨ {model_name} - 피처 중요도 분석 (SG, Target_R 제외)')
    plt.tight_layout()
    
    save_plot(plt, f'05_Feature_Importance_{model_name}.png', output_dir)
    print(f"   - [시각화] {model_name} 피처 중요도 저장 완료.")