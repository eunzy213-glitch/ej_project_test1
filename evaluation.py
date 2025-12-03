import pandas as pd # 데이터 분석 및 데이터프레임 구조를 위한 pandas 임포트
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # R2, MSE, MAE 평가 지표 함수 임포트
import numpy as np # 수치 계산 및 제곱근 함수(np.sqrt)를 사용하기 위해 numpy 임포트

def calculate_metrics(all_results: list) -> pd.DataFrame:
    """
    모든 모델의 R2, RMSE, MAE를 계산하고 데이터프레임으로 반환합니다.
    
    Args:
        all_results: 각 모델의 결과 딕셔너리 리스트. 각 딕셔너리는 'Model', 'y_test', 'Prediction' 키를 포함함.
    
    Returns:
        pd.DataFrame: 모델 이름과 성능 지표(R2, RMSE, MAE)를 포함하는 데이터프레임.
    """
    performance_list = [] # 계산된 성능 지표를 딕셔너리 형태로 임시 저장할 리스트
    
    for result in all_results:
        # 각 결과 딕셔너리에서 실제값(y_test)과 예측값(Prediction)을 추출
        y_test = result['y_test']
        prediction = result['Prediction']
        
        # 결정 계수 (R-squared, R2) 계산: 모델이 분산을 얼마나 잘 설명하는지 나타냄
        r2 = r2_score(y_test, prediction)
        
        # 평균 제곱근 오차 (Root Mean Squared Error, RMSE) 계산: 오차의 평균 크기를 실제값과 같은 단위로 나타냄
        # MSE를 계산한 후 제곱근을 취함
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        
        # 평균 절대 오차 (Mean Absolute Error, MAE) 계산: 오차의 절대값 평균
        mae = mean_absolute_error(y_test, prediction)
        
        # 계산된 지표를 딕셔너리 형태로 리스트에 추가
        performance_list.append({
            'Model': result['Model'],
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })
    
    # 최종적으로 리스트에 모인 딕셔너리들을 데이터프레임으로 변환하여 반환    
    return pd.DataFrame(performance_list)


def calculate_clark_error_grid(result: dict):
    """
    Clark Error Grid (CEG) 분석을 수행하고 그 결과를 result 딕셔너리에 추가합니다.
    CEG는 혈당 예측 모델의 임상적 정확도를 평가하는 표준 방법입니다.
    """
    y_true = result['y_test'] # 실제 혈당 값 (True BG)
    y_pred = result['Prediction'] # 모델 예측 혈당 값 (Predicted BG)
    
    # [실제 CEG 계산 로직이 들어갈 자리]
    # 실제 CEG 분석은 (A, B 영역의 비율)을 계산하는 복잡한 로직을 요구하며,
    # 해당 로직은 외부 라이브러리 또는 별도로 구현된 함수에 의존합니다.
    # 여기서는 단순 템플릿만 유지하며, CEG 시각화는 visualization.py에서 담당합니다.
    
    # 분석 결과를 저장할 'CEG_Analysis' 딕셔너리 키를 result에 추가
    result['CEG_Analysis'] = {
        'Total_Count': len(y_true) # 총 데이터 포인트 수 저장
        # 실제 프로젝트에서는 Area A (임상적으로 정확함), B (오차 허용), C, D, E의 비율이 여기에 저장됨
        # (예: 'Area_A_Pct': 0.90, 'Area_B_Pct': 0.08 등)
    }
    
    # 업데이트된 result 딕셔너리를 반환
    return result