# 🌪️ 제3회 풍력발전량 예측 AI 경진대회 (BARAM 2026) - 프로젝트 전략 및 명세서

---

## 1. 대회 개요 및 평가 체계 분석

### 1.1 대회 정보

* **대회명**: 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026


* **주최 / 주관**: 한국동서발전, GS E&R, 태백가덕산풍력발전 / 데이콘


* **대회 기간**: 2026년 7월 6일 ~ 2026년 8월 14일 (1차 마감) / 2026년 8월 17일 (2차 산출물 마감)



### 1.2 평가지표 구조

$$\text{Total Score} = 0.5 \times (1 - \text{NMAE}) + 0.5 \times \text{FICR}$$

* **평가 대상 구간**: 실제 발전량이 그룹별 설비용량의 **$10\%$ 이상인 시간대만 평가**에 반영 (`actual >= capacity * 0.10`).


* **설비용량 환산 ($kWh$)**:
* `kpx_group_1`: $21.6 \text{ MW} \rightarrow 21,600 \text{ kWh}$

* `kpx_group_2`: $21.6 \text{ MW} \rightarrow 21,600 \text{ kWh}$

* `kpx_group_3`: $21.0 \text{ MW} \rightarrow 21,000 \text{ kWh}$




### 1.3 지표별 특성 및 전략적 시사점

1. **$1 - \text{NMAE}$ (평균 절대 오차율)**:
* $\text{NMAE} = \frac{1}{N} \sum \frac{\vert{}\text{예측 발전량} - \text{실제 발전량}\vert{}}{\text{그룹 설비용량}}$

* $L_1 \text{ Loss}(\text{MAE})$ 기반의 전반적인 예측 정밀도를 평가함.




2. **$\text{FICR}$ (정산금 획득률)**:
* $\text{FICR} = \frac{\sum (\text{실제 발전량} \times \text{구간별 정산단가})}{\sum (\text{실제 발전량} \times 4.0)}$

* **계단형 불연속 단가 구획**:
* 오차율 $\le 6\%$: **$4.0 \text{ 원/kWh}$** ($100\%$ 수령)


* $6\% <$ 오차율 $\le 8\%$: **$3.0 \text{ 원/kWh}$** ($75\%$ 수령)


* 오차율 $> 8\%$: **$0.0 \text{ 원/kWh}$** (미수령)




* **핵심 인사이트**: 단순 MAE 최소화 모델은 오차율 경계선($6\%$, $8\%$) 부근에서 FICR 점수가 급격히 하락함. OOF 기반 최적 후처리($\alpha, \beta$ 스케일 변환)로 예측 오차를 $6\%$ 및 $8\%$ 이내 구간으로 인위적으로 밀어 넣는 후처리 기법 적용 필수.



---

## 2. 데이터 및 도메인 구조 분석

### 2.1 그룹별 터빈 스펙 및 입지 지형 메타 정보 (`info.xlsx`)

| KPX 그룹 | 단계 | 단지명 | 제작사 | 모델명 | 대상 호기 | 설비용량 | Hub Height | Rotor Dia. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`kpx_group_1`** | 1단계 | 태백가덕산 | VESTAS | V126 | WTG 01 ~ 06 (6기) | **21.6 MW** | 117 m | 126 m |
| **`kpx_group_2`** | 1단계 | 태백가덕산 | VESTAS | V126 | WTG 07 ~ 12 (6기) | **21.6 MW** | 117 m | 126 m |
| **`kpx_group_3`** | 2단계 | 태백가덕산(1호) / 태백원동(2~5호) | UNISON | U136 | WTG 01 ~ 05 (5기) | **21.0 MW** | 117 m | 136 m |

> **도메인 핵심 특성**:
> * **수풍 면적 차이**: VESTAS(로터 $126\text{m}$)와 UNISON(로터 $136\text{m}$)은 동일 풍속 대비 수풍 면적이 달라 Power Curve 경사도가 다름.
> 
> 
> * **Group 3 지형 및 후류(Wake Effect) 이슈**:
> * Group 3의 1호기(가덕산)는 2~5호기(원동, 약 $2.5\text{km}$ 이격)와 떨어져 있으며, Group 1, 2(VESTAS 12기) 바로 후류 측에 위치함.
> 
> 
> * 서풍 형성 시 Group 1, 2가 바람 에너지를 먼저 흡수한 후 발생하는 와류(Turbulence) 및 풍속 감쇄 직격탄을 받으므로, Group 3 모델 구축 시 가덕산과 원동 기상 격자 예보값을 조합(Pivot 및 공간 가중 평균)해야 함.
> 
> 
> 
> 
> * **라벨 데이터 결측**: `kpx_group_3`는 상업운전 개시 시점이 늦어 **2022년 전체 라벨이 Null**로 처리되어 있어 2023~2024년 데이터로만 학습.
> 
> 
> 
> 

### 2.2 기상 예보 데이터 이원화 비교 (LDAPS vs GFS)

| 구분 | LDAPS (국지예보) | GFS (전지구예보) |
| --- | --- | --- |
| **공간 해상도** | **~1.5 km** (16개 격자 ID)

 | **~25 km (0.25도)** (9개 격자 ID)

 |
| **바람 고도 변수** | 지상 10m, 50m (최대/최소), 5m 경계층 바람

 | 지상 10m, **80m, 100m**, PBL, **850/700/500hPa**<br> |
| **고유 기상 변수** | 순 하향 단파/장파복사, 직달/산란 단파복사, 경계층 높이(`blh`)

 | 돌풍(`gust`), 행성경계층 수직속도, 상층 등압면 기온/바람

 |
| **특성 및 활용** | 산악 국지 지형 분해능 우수, 난류 및 산곡풍 추정

 | **117m 허브 고도에 근접**, 상공 바람 유입 및 대기 불안정도 추정

 |

---

## 3. 프로젝트 파이프라인 및 검증 체계 (v6 개편)

### 3.1 파일 및 모듈 역할 구조

```
project_root/
├── configs/               # yaml 하이퍼파라미터 및 경로 설정 (config_v1~v5)
├── information/           # 명세서(data_description.md) 및 메타 데이터(info.xlsx)
├── main/
│   ├── main_train.py      # [학습] GroupKFold OOF 학습, feature_cols.pkl & post_params.pkl 저장
│   └── main_inference.py  # [추론] feature_cols 재정렬 및 최적 후처리(apply_postprocessing) 적용
├── notebooks/             # 가설 검증 EDA 및 Pydeck 3D 시각화 (EDA.ipynb)
├── saved_models/          # 버전별/Fold별 학습 모델 저장 (.pkl)
├── src/
│   ├── features.py        # 달력/풍속/도메인/시계열 파생 변수 생성 (Past-only 연산)
│   ├── postprocessing.py  # OOF 기반 Total Score 극대화 수치 최적화 (scipy.optimize)
│   ├── logger.py          # experiment_log.xlsx 자동 로깅
│   └── utils.py           # 평가 지표(calculate_metric) 및 시드 고정
└── experiment_log.xlsx    # 정량적 실험 결과 및 실행 소요 시간 자동 기록

```

### 3.2 검증 체계 (Validation Strategy)

* **Year-Month `GroupKFold` 도입**:
* 기존 무작위 `KFold(shuffle=True)` 적용 시 시계열 파생 변수(`diff`, `rolling`)로 인한 **Validation Data Leakage (미래 정보 참조)** 발생.


* 연-월(`Year-Month`) 단위로 그룹화하여 특정 월 전체가 완전 독립된 Validation Set으로 들어가도록 검증 체계 개편.


* **Inference Pipeline Safety**:
* 학습 시 사용된 컬럼 순서 및 목록을 `feature_cols.pkl`로 저장 후 추론 시 `reindex`하여 **피처 인덱스 밀림 현상 방지**.


* Dataframe 형태 입력 유지로 트리 기반 모델(LightGBM, CatBoost, XGBoost)의 **컬럼명 정합성 확보**.



---

## 4. 실험 이력 및 점수 추이 (Experiment Log)

| 제출 버전 | OOF / Train Total | Public Score | 1 - NMAE (오차율) | FICR | 주요 변경 및 파이프라인 특징 |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | - | 0.588346 | 0.863681 (13.63%) | 0.313010 | 기본 샘플 제출 양식 베이스라인

 |
| **v1** | 0.7137 (Train) | 0.593206 | 0.861465 (13.85%) | 0.324947 | 달력 피처 + GFS 80m 합성 풍속 (RandomForest)

 |
| **v2** | 0.7993 (Train) | 0.602090 | 0.863945 (13.61%) | 0.340235 | LightGBM 전환 + 100m 풍속/풍향 피처

 |
| **v3** | 0.6864 (OOF) | 0.605691 | 0.866023 (13.40%) | 0.345359 | 5-Fold CV 도입 및 OOF 앙상블 적용

 |
| **v4** | 0.6886 (OOF) | 0.613708 | 0.870777 (12.92%) | 0.356640 | 격자 Pivot + 117m 멱법칙 보정 + 풍향 sin/cos 변환

 |
| **v5** | 0.6882 (OOF) | 0.615710 | 0.870822 (12.91%) | 0.359520 | 푄현상/열돔 지수 + 시계열 차분/이동평균 피처

 |
| **v6** | **재구축 완료** | - | - | - | **Data Leakage 완전 수정 + GroupKFold + FICR 후처리 연동**<br> |

---

## 5. 도메인/물리 가설 기반 피처 엔지니어링 명세

### 5.1 기적용 핵심 피처 (`src/features.py`)

1. **117m Hub Height 멱법칙(Power Law) 보정 풍속**:
* GFS 80m, 100m 풍속 비율로 대기 연돌 지수 $\alpha$ 계산 후 117m 고도 풍속 외삽:

$$\alpha = \frac{\ln(v_{100} / v_{80})}{\ln(100 / 80)}, \quad v_{117} = v_{100} \cdot \left(\frac{117}{100}\right)^\alpha$$


* LDAPS 10m 풍속은 산악 지형 표준 $\alpha = 0.20$ 적용하여 117m 수풍 고도로 보정.


2. **풍향 삼각함수 변환**:
* $360^\circ \leftrightarrow 1^\circ$ 경계 불연속성 방지를 위해 $\sin(\text{rad}), \cos(\text{rad})$ 벡터 변환.


3. **태백 특수 기상 지수**:
* **높새바람(푄현상) 지수**: 동풍 강도 $\times$ 건조도(기온 - 이슬점) $\times$ 계절/시간 가중치.
* **열돔/대기정체 지수**: 여름철(7~8월) 고온 대비 저풍속 비율 ($T / (v_{117} + 0.1)$).


4. **시계열 변동성 (Past-only 연산)**:
* 1시간/2시간 과거 차분 (`ws.diff(1)`, `ws.diff(2)`) 및 과거 3시간 이동평균/표준편차 (`center=False`).



### 5.2 v6+ 신규 추가 가설 피처 로드맵

```
                            [신규 피처 확장 플랜]
                                      │
 ┌───────────────────┬────────────────┴───────────────────┬───────────────────┐
 │                   │                                    │                   │
 ▼                   ▼                                    ▼                   ▼
[1. 모델 간 불확실성]  [2. GFS 상공 기류 & 돌풍]             [3. LDAPS 난류 & 복사]  [4. 물리/도메인 수식]
- U/V 풍속 예보 격차  - 850hPa 상공 바람 유입 비율            - 50m 난류 변동 폭     - 공기 밀도 (ρ)
- 지표/해면 기압 차  - 연직 기온 감률 (대기 불안정도)          - 경계층 높이 (blh)    - 풍력 에너지 밀도 (WPD)
- 2m 기온/이슬점 차   - 돌풍 지수 (Gust / WS)             - 일사량 산곡풍 지수   - Group 1->2->3 Cross-Lag

```

1. **기상 모델 간 불확실성 (Discrepancy)**:
* GFS와 LDAPS 간 10m U/V 풍속, 기압, 기온 차이 피처 생성 $\rightarrow$ 예보 불확실성이 큰 구간 오차 방어.




2. **GFS 상공 기류 및 돌풍 지수**:
* **850hPa 유입 비율**: 상공 바람이 지표 하강 기류로 유입되는 강도 추정.


* **돌풍 비율 (Gust Ratio)**: $\text{Gust} / v_{117}$ 지수가 높을 경우 터빈 피칭 제어(Pitching)에 따른 출력 감소 현상 모사.




3. **LDAPS 난류 및 미세 지형 피처**:
* **50m 난류 변동폭**: $\text{U/V}_{\text{max}} - \text{U/V}_{\text{min}} \rightarrow$ 순간 난류에 의한 발전 효율 감소 반영.


* **일사량 산곡풍 지수**: (직달 + 산란) 단파복사로 낮 시간대 골짜기 대류(Valley Wind) 유도 지수화.




4. **열역학 및 풍력 공학 물리 수식**:
* **이상기체 상태방정식 공기 밀도 ($\rho$)**:

$$\rho = \frac{P}{R \cdot T} \quad \left(P: \text{기압}, \, T: \text{절대온도(K)}, \, R = 287.05 \text{ J/(kg·K)}\right)$$


* **물리적 풍력 에너지 밀도 ($WPD$)**:

$$WPD = \frac{1}{2} \cdot \rho \cdot v_{117}^3$$


* **단지 간 후류(Wake Effect) Cross-Lag**:
* 상류 단지(`group_1`)의 과거 $t-1, t-2$ 풍속/발전량을 하류 단지(`group_2`, `group_3`)의 피처로 전달 모사.




---

## 6. FICR 최적화 후처리 전략 (`src/postprocessing.py`)

### 6.1 알고리즘 동작 메커니즘

* **목적**: MAE/MSE 기반 회귀 모델의 보수적 예측값을 FICR 인센티브 구간($6\%$ 이하 $4\text{원}$, $8\%$ 이하 $3\text{원}$) 내부로 이동시켜 $Total Score$ 극대화.


* **수식 구조**:

$$\hat{y}_{\text{adjusted}} = \text{clip}\left(\hat{y} \cdot \alpha + \beta, \, 0, \, \text{Capacity}\right)$$


* **자동 파라미터 탐색 (`scipy.optimize.minimize`)**:
* `main_train.py` 실행 시 OOF 예측값과 실제 정답을 바탕으로 Nelder-Mead 알고리즘이 $Total Score$를 최대화하는 그룹별 $(\alpha, \beta)$ 수치를 자동으로 최적 수렴 후 `post_params.pkl` 저장.


* `main_inference.py` 실행 시 저장된 파라미터를 불러와 테스트 예측값에 즉시 일괄 적용.



---

## 7. 향후 액션 플랜 및 v6+ 진행 로드맵

```
  Step 1. 기초 데이터 EDA 정밀화 (EDA.ipynb 실행)
   └── 그룹별 Target 분포, 결측치 패턴, 풍속-발전량 Power Curve 이상치 재검증[cite: 15, 16]

   [EDA 확장 파이프라인]
 ├── 1. 기초 통계 & 분포 정밀 점검 (Target & Weather Features)
 │     ├── Target(Group 1~3) 평균, 표준편차, 왜도(Skewness), 첨도(Kurtosis), 0 kWh 및 10% 미만 비율
 │     └── 기상 예보 변수(풍속, 기온, 기압)의 이상치(Outlier) 및 불연속 구간 식별
 ├── 2. Feature Importance 분석 (LightGBM Gain 기준)
 │     ├── v5 피처 세트 기반 Quick OOF 모델 재학습
 │     ├── Top 20 핵심 피처 및 Bottom 20 노이즈 피처(Pruning 후보) 도출
 └── 3. 상관관계 & 다중공선성(Multicollinearity) 분석
       ├── Target - Feature 간 상관계수 (Pearson / Spearman)
       └── 피처 간 높은 상관관계(|r| > 0.95)를 갖는 다중공선성 변수 그룹 추출

  Step 2. 기존 v2~v5 피처 유효성 검증 및 가지치기 (Feature Pruning)
   └── SHAP / Feature Importance 분석으로 다중공선성 및 불필요 피처 정리[cite: 15]

  Step 3. v6+ 물리/도메인 피처 단계적 반영 및 OOF 점수 검증
   └── 공기밀도(ρ), WPD, LDAPS 난류 변동폭, GFS 상공 바람, Cross-Lag 피처 생성 및 단독/합성 실험[cite: 15, 17]

  Step 4. 모델 다변화 및 앙상블 파이프라인 구축
   └── LightGBM, CatBoost, XGBoost 개별 최적화 후 가중 평균(Weighted Ensemble) 적용[cite: 12, 13]

  Step 5. 2차 발표평가 대비 가용성 검증
   └── experiment_log.xlsx에 학습 및 추론 소요 시간(Execution Time) 로깅 자동화[cite: 4, 15]

```

---