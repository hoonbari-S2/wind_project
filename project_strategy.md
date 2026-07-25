# 🌪️ 제3회 풍력발전량 예측 AI 경진대회 (BARAM 2026) - 프로젝트 전략 및 명세서(v6 ver. 2026. 07. 25)

---

## 1. 대회 개요 및 평가 체계 분석

### 1.1 대회 정보
* **대회명**: 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026
* **주최 / 주관**: 한국동서발전, GS E&R, 태백가덕산풍력발전 / 데이콘
* **대회 기간**: 2026년 7월 6일 ~ 2026년 8월 14일 (1차 마감) / 2026년 8월 17일 (2차 산출물 마감)

### 1.2 평가지표 구조
$$\text{Total Score} = 0.5 \times (1 - \text{NMAE}) + 0.5 \times \text{FICR}$$

* **평가 대상 구간**: 실제 발전량이 그룹별 설비용량의 **10% 이상인 시간대만 평가**에 반영 (`actual >= capacity * 0.10`).
* **설비용량 환산 ($kWh$)**:
  * `kpx_group_1`: $21.6 \text{ MW} \rightarrow 21,600 \text{ kWh}$
  * `kpx_group_2`: $21.6 \text{ MW} \rightarrow 21,600 \text{ kWh}$
  * `kpx_group_3`: $21.0 \text{ MW} \rightarrow 21,000 \text{ kWh}$

### 1.3 지표별 특성 및 전략적 시사점
1. **$1 - \text{NMAE}$ (평균 절대 오차율)**:
   * $L_1 \text{ Loss}(\text{MAE})$ 기반의 전반적인 예측 정밀도를 평가함.
2. **$\text{FICR}$ (정산금 획득률)**:
   * **계단형 불연속 단가 구획**: 오차율 $\le 6\%$ ($4.0 \text{원/kWh}$), $6\% <$ 오차율 $\le 8\%$ ($3.0 \text{원/kWh}$), 오차율 $> 8\%$ ($0.0 \text{원/kWh}$).
   * **핵심 인사이트**: OOF 기반 Nelder-Mead 최적 후처리를 통해 예측 오차를 $6\%$ 및 $8\%$ 이내 구간으로 인위적으로 밀어 넣는 후처리 기법 적용 필수.

---

## 2. 데이터 및 도메인 구조 분석

### 2.1 그룹별 터빈 스펙 및 입지 지형 메타 정보 (`info.xlsx`)
| KPX 그룹 | 단계 | 단지명 | 제작사 | 모델명 | 대상 호기 | 설비용량 | Hub Height | Rotor Dia. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`kpx_group_1`** | 1단계 | 태백가덕산 | VESTAS | V126 | WTG 01 ~ 06 (6기) | **21.6 MW** | 117 m | 126 m |
| **`kpx_group_2`** | 1단계 | 태백가덕산 | VESTAS | V126 | WTG 07 ~ 12 (6기) | **21.6 MW** | 117 m | 126 m |
| **`kpx_group_3`** | 2단계 | 태백가덕산(1호) / 태백원동(2~5호) | UNISON | U136 | WTG 01 ~ 05 (5기) | **21.0 MW** | 117 m | 136 m |

---

## 3. 프로젝트 파이프라인 및 검증 체계 (v6 개편)

### 3.1 파일 및 모듈 역할 구조

```
project_root/
├── run.py                 # [엔트리포인트] CLI 실행 컨트롤러 (`python run.py train / inference`)
├── configs/               # yaml 하이퍼파라미터 및 경로 설정 (config_v1~v6)
├── information/           # 데이터 명세서(data_description.md) 및 메타 데이터(info.xlsx)
├── main/
│   ├── train.py           # [학습] 6대 피처 파이프라인 + Pruning + GroupKFold + Nelder-Mead
│   └── inference.py       # [추론] feature_cols 재정렬 + Nelder-Mead 최적 후처리 적용
├── notebooks/             # EDA 및 TreeSHAP 검증 (EDA.ipynb)
├── saved_models/          # 버전별/Fold별 학습 모델 및 feature_cols.pkl, post_params.pkl 저장
├── src/
│   ├── features.py        # 6대 도메인 파생 피처 생성 및 다중공선성/TreeSHAP 35개 Pruning
│   ├── postprocessing.py  # Nelder-Mead (alpha, beta, zero_th) 비구배 최적화 후처리
│   ├── logger.py          # execution_time_sec 및 로컬 하드웨어 스펙 자동 로깅
│   └── utils.py           # 평가 지표(calculate_metric) 및 시드 고정
└── experiment_log.xlsx    # 정량적 실험 결과 및 실행 소요 시간 자동 기록

```

### 3.2 EDA & Feature Selection (v6 적용 완료)
1. **SCADA 이상치 정제 & Power Curve 분석**: VESTAS $600\text{ kW10m}$, UNISON $700\text{ kW10m}$ 물리 정격 규명 및 고풍속 De-rating(감발) 동작 포착.
2. **다중공선성 정제**: $r \ge 0.95$ 이상 중복 거시 기압 격자 및 물리 중복 강수 컬럼 1차 단순화.
3. **TreeSHAP OOF 검정**: Gain=0 & SHAP=0 완전 미사용 정적 변수 35개 최종 가지치기(Pruning).

---

## 4. 실험 이력 및 점수 추이 (Experiment Log)

| 제출 버전 | OOF / Train Total | Public Score | 1 - NMAE (오차율) | FICR | 주요 변경 및 파이프라인 특징 |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | - | 0.588346 | 0.863681 (13.63%) | 0.313010 | 기본 샘플 제출 양식 베이스라인 |
| **v1** | 0.7137 (Train) | 0.593206 | 0.861465 (13.85%) | 0.324947 | 달력 피처 + GFS 80m 합성 풍속 (RandomForest) |
| **v2** | 0.7993 (Train) | 0.602090 | 0.863945 (13.61%) | 0.340235 | LightGBM 전환 + 100m 풍속/풍향 피처 |
| **v3** | 0.6864 (OOF) | 0.605691 | 0.866023 (13.40%) | 0.345359 | 5-Fold CV 도입 및 OOF 앙상블 적용 |
| **v4** | 0.6886 (OOF) | 0.613708 | 0.870777 (12.92%) | 0.356640 | 격자 Pivot + 117m 멱법칙 보정 + 풍향 sin/cos 변환 |
| **v5** | 0.6882 (OOF) | 0.615710 | 0.870822 (12.91%) | 0.359520 | 푄현상/열돔 지수 + 시계열 차분/이동평균 피처 |
| **v6** | **고도화 완료** | **검증 진행 중** | - | - | **6대 물리/도메인 피처 + 35개 Pruning + XGBoost CUDA + Nelder-Mead (3파라미터) 후처리 적용** |

---

## 5. 도메인/물리 가설 기반 피처 엔지니어링 명세 (`src/features.py`)

### 5.1 v6 통합 6대 파생 피처 세트
1. **예보 모델 간 불확실성 (LDAPS vs GFS)**: 10m U/V 풍속 편차, 지표/해면 기압 편차, 2m 기온/이슬점/건조도 예보 격차.
2. **상층 대기 연직 유입 & 돌풍 (GFS 고유)**: 850hPa 바람 유입 비율, 연직 기온 감률, 돌풍 지수(Gust Ratio), 117m 멱법칙 보정 풍속, Ramp Event(1/2차 차분 및 3시간 Rolling).
3. **미세 지형 난류 & 산곡풍 (LDAPS 고유)**: 50m U/V 난류 변동폭, 경계층 높이(BLH) 역수 및 117m 상호작용, 지표 순 단파복사 및 직달일사 비율.
4. **열역학 & 정밀 물리 수식**: 이상기체 상태방정식 공기 밀도 ($\rho = P / (R \cdot T)$), 물리적 풍력 에너지 밀도 ($WPD = \frac{1}{2} \rho v^3$).
5. **터빈 제어 & 도메인 직교 분해**: Cut-in/out, Soft Cut-out De-rating factor, Cut-out 히스테리시스, 태백산맥 능선 축(170도) 수직/평행 풍속 분해, 착빙 위험도, 커테일먼트 시간대.
6. **지역 특수 기상 지수**: 푄현상(높새바람) 지수 (동풍 $\times$ 건조도 $\times$ 계절/시간 가중치), 열돔 고온 무풍 정체 지수.

---

## 6. Nelder-Mead 최적화 후처리 전략 (`src/postprocessing.py`)

### 6.1 알고리즘 메커니즘 및 최적화 수식
$$\text{Post-processed Prediction} = \text{Clip}\Big(\text{Where}(\alpha \cdot \hat{y} + \beta < \text{zero\_th},\ 0.0,\ \alpha \cdot \hat{y} + \beta),\ 0,\ \text{Capacity}\Big)$$

* **$\alpha$ (Slope)**: 모델 전반의 과대/과소 예측 스케일 교정.
* **$\beta$ (Intercept)**: 센서 편향 및 대기 마찰에 따른 절편 이동.
* **$\text{zero\_th}$ (Zero Threshold)**: Cut-in 미달 저풍속 구간의 미세 잔여 노이즈를 $0\text{ kWh}$로 강제 절단.
* **Nelder-Mead 채택 이유**: 대회 평가 산식(FICR 계단식 단가) 및 $0\text{ kWh}$ 절단 조건이 미분 불가능한 불연속 함수이므로, 미분을 사용하지 않는 비구배(Non-gradient) 다면체 탐색 기법으로 Total Score를 직접 극대화함.

## 7. 향후 프로젝트 실행 로드맵 (v7 ~ Final)

[Phase 1. v6 Benchmark] ──► [Phase 2. Hyperparameter Optimization] ──► [Phase 3. Multi-Model Diversification]
│
[Phase 5. Final Submissions] ◄── [Phase 4. Weighted Stacking & Post-Processing] ◄┘

### 📍 Phase 1. v6 Baseline 및 검증 체계 안착
* **목표**: v6 파이프라인(XGBoost CUDA + 6대 도메인 피처 + Pruning + Nelder-Mead) 실행 및 성능 Benchmark 수립
* **세부 실행 과제**:
  * `python main_train.py` 실행 및 OOF CV Total Score 산출
  * Nelder-Mead 후처리 적용 전/후 score 개선폭($\Delta \text{Total Score}$) 확인 및 `experiment_log.xlsx` 로깅 자동화 점검
  * `submit_v6.csv` 제출을 통한 Public Leaderboard 점수 변화 및 CV-Public 간 Correlation 검증

---

### 📍 Phase 2. Optuna 기반 초매개변수(Hyperparameter) 최적화
* **목표**: XGBoost CUDA 모델의 표현력 향상 및 과적합(Overfitting) 방어
* **세부 실행 과제**:
  * **Tweedie Variance Power ($\rho$) 튜닝**: $1.1 \sim 1.9$ 범위 탐색을 통해 $0\text{ kWh}$ 점질량 및 오른쪽 꼬리 왜도(Skewness) 피팅 최적점 탐색
  * **트리 구조 하이퍼파라미터 튜닝**: `max_depth` ($4 \sim 10$), `learning_rate` ($0.01 \sim 0.05$), `subsample` ($0.6 \sim 0.9$), `colsample_bytree` ($0.5 \sim 0.9$) 탐색
  * **정규화 및 노이즈 방어**: `min_child_weight` 및 `reg_alpha` / `reg_lambda` 세밀 조율로 산악 지형 극단 예보 구간 오버피팅 제어

---

### 📍 Phase 3. 이종(Diverse) 트리 알고리즘 확장
* **목표**: XGBoost 단일 모델의 한계를 극복하기 위해 다각화된 이종 트리 모델 라인업 구축
* **세부 실행 과제**:
  * **CatBoost Regressor**: Ordered Boosting 특성을 활용하여 연속형 기상 변수 간 복합 비선형 상호작용 학습
  * **LightGBM Regressor**: Leaf-wise 수풍 구간 분할 성능을 활용한 빠른 피팅 및 다양성 확보
  * **Huber / MAE Loss 대조 모델**: Tweedie Loss 모델 외에 $L_1$ 오차 직접 최적화 모델을 추가하여 $1 - \text{NMAE}$ 방어선 구축

---

### 📍 Phase 4. OOF 기반 가중 앙상블 & Nelder-Mead 재적용
* **목표**: 서로 다른 모델의 예측 편향(Bias) 상쇄 및 정산금 획득률($\text{FICR}$) 극대화
* **세부 실행 과제**:
  * **그룹별 독립 가중치 앙상블**: KPX Group 1, 2, 3 각각에 대해 OOF 점수를 극대화하는 모델별 최적 가중치($w_{\text{xgb}}, w_{\text{cat}}, w_{\text{lgb}}$) 탐색 (`scipy.optimize` 활용)
  * **앙상블 후 Nelder-Mead 최종 재적용**: 앙상블을 통해 노이즈가 평탄화(Smoothing)된 예측값 상에 $(\alpha, \beta, \text{zero\_th})$ 후처리를 다시 태워 $6\%$ 정산금 구간 점유율 쥐어짜기

---

### 📍 Phase 5. 제출 파일 최종 선정 및 코드 재현성(Reproducibility) 확보
* **목표**: Public/Private Shake-down 방어 및 2차 발표평가/코드 제출 완벽 대비
* **세부 실행 과제**:
  * OOF CV 점수와 Public 점수가 동반 상승한 **단일 최적 모델(Best Individual)** 및 **최종 앙상블 모델(Best Ensemble)** 2종 선택
  * 전역 난수 시드 고정(`seed_everything`), 불필요 주석 정리, 가용 하드웨어 실행 시간(Execution Time) 최종 확인으로 산출물 재현성 확보