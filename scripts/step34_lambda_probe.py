"""
[도구] scripts/step34_lambda_probe.py
==============================================================================
후처리 강도 λ 프로브 — 학습 0회, 저장된 제출 파일 두 장의 산술.

    pred(λ) = raw + λ·(post − raw)

    λ=0 → v13-nopost (LB 0.620418 실측)
    λ=1 → v13        (LB 0.641458 실측)
    λ>1 → 미시험 구간. 이 스크립트가 만드는 것.

왜 이것을 묻는가 (§3.30 point 2)
  λ=1 은 2022~24 OOF 로 적합한 값임. 그런데 §3.30 이 실측한 것은
  "우리 OOF 가 '문턱 과적합' 이라 부른 날카로운 조정이 2025 public 에서는
  오히려 벌고 있음" 임 — bag25(후처리 곡선을 **뭉갠** 변형)가 LB −0.0020 이었음.
  뭉개서 졌다면 **더 세우면** 벌 수 있다는 방향이 한 번도 시험되지 않았음.
  §3.14/§3.15 가 닫은 것은 **OOF 기준의 후처리 함수공간**이고, λ 축은
  **LB 를 계측기로 쓰는 것**(§3.30 규칙 2(b) · §8.7)이라 별개임.

사전 등록 판정 (결과 보기 전에 고정. 07-27 KST 밤)
  λ=1.15 > 0.641458  -> 후처리가 2025 에서 **과소적용**. λ=1.20 으로 1회 더.
                        §3.14 의 '포화' 는 OOF 기준이었음을 정본에 정정.
  λ=1.15 ≤ 0.641458  -> λ=1 이 2025 에서도 최적 근처. **후처리 축 OOF·LB 양쪽 종결.**
                        λ<1 은 시험하지 않음 (λ=0 이 이미 −0.0210 이므로 방향이 자명).
  선형 외삽 상한(여지가 λ 에 선형일 때) = 0.644614. 포화가 있으면 그보다 작음.
  제출 후 premium_track.py 로 프리미엄까지 기록함 (§8.7).

λ 선택 근거 (안전 구간)
  raw 순서 대비 예측 순서 역전 비율 / 최저 예측:
    λ=1.05  0.7% / 15.6%cap      λ=1.15  0.7% / 17.1%cap
    λ=1.10  0.7% / 16.3%cap      λ=1.20  0.7% / 17.8%cap
    λ=1.30 26.3% / 18.6%cap  <- 급증. 1.25 이상은 쓰지 않음
  클리핑 0행, 결측 0.

실행
    python scripts/step34_lambda_probe.py --lam 1.15
    python scripts/step34_lambda_probe.py --lam 1.20 --out submit_v13_lam120.csv
==============================================================================
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True, help="후처리 강도. 1.0 = 현행 v13")
    ap.add_argument("--post", default="./submissions/submit_v13.csv")
    ap.add_argument("--raw", default="./submissions/submit_v13_no_post.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    post = pd.read_csv(args.post)
    raw = pd.read_csv(args.raw)
    if not (post["forecast_id"] == raw["forecast_id"]).all():
        raise ValueError("forecast_id 순서가 다름 — 정렬 후 다시 실행할 것")

    out = post.copy()
    n_clip = 0
    for c, cap in CAP.items():
        v = raw[c] + args.lam * (post[c] - raw[c])
        n_clip += int(((v < 0) | (v > cap)).sum())
        out[c] = v.clip(0, cap)

    # 진단: raw 순서 대비 역전 (λ>1 은 원리적으로 단조성을 깎으므로 크기를 확인함)
    inv = 0
    for c in CAP:
        o = np.argsort(raw[c].to_numpy())
        inv += int((np.diff(out[c].to_numpy()[o]) < -1e-9).sum())
    inv_pct = inv / (3 * (len(out) - 1)) * 100

    dst = Path(args.out or f"./submissions/submit_v13_lam{int(round(args.lam*100))}.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    print(f"λ = {args.lam}  ->  {dst}")
    print(f"  클리핑 {n_clip}행 / 결측 {int(out.isna().sum().sum())} / 순서역전 {inv_pct:.1f}%")
    for c, cap in CAP.items():
        d = (out[c].mean() - post[c].mean()) / post[c].mean() * 100
        print(f"  {c}: 평균 {out[c].mean():8.1f} ({out[c].mean()/cap*100:5.2f}%cap)  "
              f"최소 {out[c].min():7.1f}  v13 대비 {d:+.2f}%")
    if inv_pct > 5:
        print("  ⚠ 순서역전 5% 초과 — 단조 사상이 깨졌음. λ 를 낮출 것")


if __name__ == "__main__":
    main()
