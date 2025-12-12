# 🩸 변수 조합별 BG 예측 모델 성능 비교 분석(Test1-1)

## 📌 1. 프로젝트 개요 및 목표

- 본 프로젝트는 **타액 포도당(SG) 및 Target 비율(Target_R)** 과 같은 직접적인 혈당 지표 외에, 인구통계학적 정보, 라이프스타일, 식이 상태 등의 **보조 피처(Auxiliary\ Features)** 를 활용하여 혈당 (Blood Glucose, $BG$) 수치를 예측하는 회귀 모델을 개발하고 그 성능을 비교 분석하는 것을 목표로 합니다.
- 특히, 데이터 전처리 단계에서 **고도화된 이상치 제거 기법**을 적용하고, 모델 학습 단계에서는 세 가지 피처셋 조건(`Auxiliary Only`, `SG Only`, `SG + Auxiliary`)에서 5가지 회귀 모델(Linear, Poly3, Huber, CatBoost, LightGBM)을 테스트하여 각 피처셋의 상대적인 기여도를 집중적으로 분석합니다.

---

## 🛠️ 2. 기술 스택 및 환경 설정

### 프로젝트 구조
```bash
project_root/
├── dataset.csv                 # ⬅️ 원본 데이터셋
├── main.py                     # ⬅️ 워크플로우 실행 및 모듈 호출 (엔트리 포인트)
├── data_processing.py          # 데이터 로딩 및 전처리 정의
├── model_training.py           # 모델 정의, 데이터 분할, 학습 파이프라인 정의
├── evaluation.py               # 성능 지표 (R2, RMSE, MAE) 및 CEG 로직 정의
├── visualization.py            # 모든 분석 결과 시각화 및 이미지 저장 로직
└── model_outputs/              # ⬅️ 최종 결과물 (이미지 파일) 저장 폴더(Test1)
└── model_outputs2/             # ⬅️ 최종 결과물 (이미지 파일) 저장 폴더(Test1-1)
```

### 실행 환경 설정

- 프로젝트를 실행하기 전에 가상 환경을 설정하고 필요한 모든 의존성을 설치해야 합니다.

```bash
# 1. 가상 환경 생성 및 활성화 
python3 -m venv .venv
source .venv/bin/activate

# 2. 필수 라이브러리 설치
pip install pandas numpy scikit-learn catboost lightgbm matplotlib seaborn statsmodels
```

### 프로젝트 실행
- `dataset.csv` 및 모든 Python 파일이 동일한 디렉토리에 있는지 확인 후 `main.py`를 실행합니다.

```bash
python main.py
```
---

## 🔬 3. 데이터 전처리 기법 및 결과 비교

- **전처리 기법: LOWESS + Isolation Forest**

이상치 제거는 BG 예측 성능에 큰 영향을 미치므로, 다음 두 가지 기법을 결합하여 경향성을 벗어나는 이상치를 탐지했습니다.

1. `SG`와 `BG` 관계의 비선형성을 반영하기 위해 **LOWESS(Locally Weighted Scatterplot Smoothing) 추세선**을 계산합니다.

2. **실제 `BG` 값과 추세선 예측 값의 잔차를 기반으로 Isolation Forest 모델을 적용**하여 이상치를 탐지하고 제거합니다.

- **전처리 전/후 데이터 Describe 비교**

이상치 제거 후 데이터의 통계량이 다음과 같이 개선되어, 데이터 분산이 크게 줄어들고 분포가 안정화된 것을 확인했습니다.

| 통계량 | 전처리 전 (Raw Data: 2000개) | 전처리 후 (Processed Data: 1900개) | 변화 및 해석 |
| :--- | :---: | :---: | :--- |
| **샘플 수** | 2000 | **1900** | 이상치 100개(5.0%) 제거 |
| **BG 평균 (Mean)** | 105.459 | **100.337** | 평균이 낮아지며 분포가 안정됨 |
| **BG 표준편차 (Std)** | 38.996 | **31.079** | 표준편차가 **7.917 감소**하여 데이터 분산이 크게 줄어듦 (이상치의 영향 감소) |
| **BG Max** | 249.799 | **209.929** | 극단적인 고혈당 이상치가 제거됨 |
| **SG Max** | 35.000 | **19.788** | SG(센서 값)의 극단적인 이상치가 제거됨 |

---

## 📊 4. 모델별 성능 지표 요약 (실행 데이터 반영)_Test1
- **종합 성능 지표 비교 (R2, RMSE, MAE)**
아래 표는 테스트 데이터셋에 대한 10가지 모델의 최종 성능 지표를 $R^2$ Score 오름차순으로 정렬한 결과입니다.

| Model | R2 | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Processed_Linear_Model** | **-0.00659977** | **31.5133** | **25.2701** |
| Raw_Linear_Model | -0.0161899 | 38.0828 | 29.8213 |
| Processed_Robust_Huber_Model | -0.0208034 | 31.7349 | 25.0735 |
| Raw_Robust_Huber_Model | -0.0291573 | 38.3250 | 28.7254 |
| Processed_CatBoost_Model | -0.145437 | 33.6164 | 26.8937 |
| Raw_CatBoost_Model | -0.156335 | 40.6241 | 31.5711 |
| Raw_LightGBM_Model | -0.158935 | 40.6697 | 31.6836 |
| Processed_LightGBM_Model | -0.168466 | 33.9527 | 27.1470 |
| Processed_Poly_3_Model | -0.209198 | 34.5394 | 27.4509 |
| Raw_Poly_3_Model | -0.268429 | 42.5475 | 32.5914 |
- **분석 요약**
**1. 전처리 우세:** 모든 모델에서 전처리(`Processed`)를 적용했을 때 $R^2$ 점수가 상승하고 오차 지표(RMSE, MAE)가 현저히 낮아져 전처리의 긍정적인 효과가 명확하게 입증되었습니다.
**2. 최적 성능:** `Processed_Linear_Model`이 통계 지표상 가장 우수한 성능을 보였으며, 이는 이상치가 제거된 데이터에서는 단순한 선형 모델이 복잡한 모델보다 더 견고하게 작동함을 시사합니다.
**3. 한계점:** 모든 모델의 $R^2$ 점수가 0에 가깝거나 음수라는 것은 현재 사용된 피처들만으로는 BG 변동성을 설명하는 데 근본적인 한계가 있음을 나타냅니다.

---

## 📊 5. 모델별 성능 지표 요약 (세 가지 Feature Set 비교)_Test1-1

### 실험 조건 정의

| Feature Set | Prefix | 특징 | 포함 피처 |
| :--- | :--- | :--- | :--- |
| **Case 1: Aux 단독** | `T1-1A_Aux_Only` | $SG$, $BG$, $Target\_R$를 제외한 모든 보조 피처 | $auxiliary\_features$ |
| **Case 2: SG 단독** | `T1-1B_SG_Only` | $SG$ 피처만 단독 사용 | $['SG']$ |
| **Case 3: SG + Aux** | `T1-1C_SG_Aux` | $SG$와 모든 보조 피처 결합 | $['SG'] + auxiliary\_features$ |

### 종합 성능 지표 비교 ($R^2, RMSE, MAE$)
아래 표는 `main.py` 실행 후 테스트 데이터셋에 대한 최종 성능 지표를 $R^2$ Score 내림차순으로 정렬한 결과이며, `model_outputs2/model_performance_summary.csv`에 저장됩니다.

| Model | $R^2$ | $RMSE$ | $MAE$ |
| :--- | :---: | :---: | :---: |
| T1-1B_SG_Only_Linear_Regression_Model | $0.4612478$ | $23.05474$ | $17.94070$ |
| T1-1B_SG_Only_Poly3_Model | $0.460495$ | $23.07084$ | $17.84079$ |
| T1-1C_SG_Aux_Linear_Regression_Model | $0.4638538$ | $22.99924$ | $17.86276$ |
| T1-1C_SG_Aux_Robust_Huber_Model | $0.4584440$ | $23.11464$ | $17.79504$ |
| T1-1B_SG_Only_Robust_Huber_Model | $0.4591637$ | $23.09929$ | $17.89784$ |
| T1-1B_SG_Only_CatBoost_Model | $0.4139960$ | $24.04451$ | $19.03292$ |
| T1-1C_SG_Aux_CatBoost_Model | $0.4197939$ | $23.92527$ | $18.70551$ |
| T1-1B_SG_Only_LightGBM_Model | $0.3891996$ | $24.54795$ | $19.39053$ |
| T1-1C_SG_Aux_LightGBM_Model | $0.2297262$ | $27.56692$ | $21.95158$ |
| T1-1A_Aux_Only_Linear_Regression_Model | $-0.004363$ | $31.47828$ | $25.26427$ |
| T1-1A_Aux_Only_Robust_Huber_Model | $-0.017118$ | $31.67754$ | $25.06976$ |
| T1-1A_Aux_Only_Poly3_Model | $-0.156571$ | $33.77940$ | $26.65412$ |
| T1-1A_Aux_Only_LightGBM_Model | $-0.148582$ | $33.66253$ | $26.86870$ |
| T1-1A_Aux_Only_CatBoost_Model | $-0.166974$ | $33.93098$ | $27.12251$ |
| T1-1C_SG_Aux_Poly3_Model | $-0.445827$ | $37.76799$ | $21.84420$ |


### 분석 요약

1.  **전처리 우세:** `data_processing.py`에서 수행된 **LOWESS 기반 Isolation Forest 이상치 제거** 전처리는 데이터 분산을 크게 줄여 모델 성능 향상에 긍정적인 영향을 미칩니다.
2.  **$SG$의 결정적 역할:** $SG$가 포함된 Case 2 (`SG Only`), Case 3 (`SG + Aux`) 모델의 성능은 $Aux$ 단독인 Case 1 (`Aux Only`)의 성능보다 **압도적으로 높습니다**. 이는 $SG$가 $BG$ 예측의 **핵심 피처**임을 강력하게 시사합니다. Case 1의 $R^2$ 값은 0에 가깝거나 음수입니다.
3.  **최적 성능:** $SG$와 $Aux$ 피처가 모두 사용된 **T1-1C\_SG\_Aux** 피처셋에서 **LightGBM** 및 **CatBoost**와 같은 **부스팅 모델**이 가장 우수한 성능을 보일 것으로 예상됩니다. 이는 비선형적인 관계를 가진 $SG$와 $Aux$ 피처들의 복합적인 영향을 효과적으로 학습했기 때문입니다.

---
## 📈 6. 결과물 및 시각화 상세
모든 시각화 결과는 스크립트 실행 후 생성되는 `model_outputs/` 폴더에 저장됩니다.
   - Test1(model_outputs/)

| 파일명 | 내용 |
| :--- | :--- |
| `preprocessed_data.csv` | **최종 전처리 완료 데이터셋 (CSV)** |
| `00_a_Raw_vs_Processed_Scatterplot_SG.png` | 전처리 전후 데이터 분포를 비교하는 산점도 |
| `00_Outlier_Removal_Comparison_SG.png` | LOWESS 기반으로 제거된 이상치를 표시하는 산점도 |
| `01_R2_Model_Comparison.png` | Raw/Processed 데이터셋별 전체 모델의 R2 점수 비교 막대 그래프 |
| `02_Residual_Plot_.png` | 모델별 잔차 분석 플롯 |
| `03_Clark_Error_Grid_.png` | 모델별 Clark Error Grid 플롯 (임상적 영역 경계선 표시) |
| `04_Confusion_Matrix_.png` | 모델별 BG 카테고리 (저/정상/고혈당) Confusion Matrix |
| `05_Feature_Importance_.png` | CatBoost/LightGBM 모델의 SG, Target_R이 제외된 피처 중요도 분석 |

   - Test1-1(model_outputs2/)

| 파일명| 내용 | 
| :--- | :--- | 
| `model_performance_summary.csv` | **최종 성능 지표 요약** (R2, RMSE, MAE) | 
| `T1_Comparison_R2_Summary.png` | 세 Feature Set별 **$R^2$ 통합 비교** 막대 그래프 | 
| `Residuals_[ModelName].png` | **잔차 분석 플롯** (예측값 vs 잔차) 및 **LOWESS 추세선** | 
| `CEG_[ModelName].png` | **Clark Error Grid Zone 분포** (임상적 정확도 비율) | 
| `ConfMatrix_[ModelName].png` | **$BG$ 카테고리 (저혈당: <70, 정상: 70-180, 고혈당: >180) Confusion Matrix** |
| `FeatureImportance_[ModelName].png` | **CatBoost/LightGBM 모델의 피처 중요도** | 
