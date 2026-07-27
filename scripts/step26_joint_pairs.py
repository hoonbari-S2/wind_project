"""
scripts/step26_joint_pairs.py
==============================================================================
통합(long-format) 조합 재설계 — 성공한 변화는 전부 '학습 문제 구조' 였음

왜 이 축인가
  §3.8 따름정리가 **5번 확인됨** (물리 피처 / SCADA 스태킹 / lead·블록 / 격자 산포 / 안정도).
  입력 표현을 바꾸는 시도는 원리적으로 실패함. 반면 성공한 4개는 전부 학습 문제 구조였음.
      정제 타깃(타깃 재정의) · 통합 학습(표본 구조) · 가지치기(피처 집합) · piecewise 후처리(출력)

  그리고 `info.xlsx` 가 물리적 그룹 구조를 알려줌.
      G1 = 가덕산 V126 1~6호     ┐ 같은 단지 · 같은 기종 · 인접 배치
      G2 = 가덕산 V126 7~12호    ┘
      G3 = 가덕산 U136 1호 + **원동 U136 2~5호**   ← 다른 기종 · 다른 단지 · 다른 연계점

  **물리적으로 가장 자연스러운 통합은 G1+G2 인데, 현행 v13 은 G1+G3 를 쓰고 G2 를 독립으로 둠.**
  step5b 가 시험한 조합(A/B/C/D/G)에 **G1+G2 만의 통합이 없음.**

두 개의 축이 섞여 있었음 — 이 스크립트가 그것을 분리함
  `run_cv_joint(df, feats, targets, ...)` 는 `targets` 로 long pool 을 만듦.
  v13 은 **pool = {G1,G2,G3} 전체**로 학습하고 **예측만 {G1,G3}** 에서 가져다 씀.
  즉 지금까지 조절한 것은 '예측 출처' 뿐이었고 **'학습 표본 pool' 은 한 번도 안 바꿨음.**

      축 1  pool  : 통합 모델을 어느 그룹의 행으로 학습하는가
      축 2  use   : 어느 그룹이 통합 모델의 예측을 쓰는가

조건 (pool -> use, 나머지는 독립 모델)
      A  전부독립                                        기준
      B  pool{1,2,3} -> use{1,3}                        **v13 현행**
      C  pool{1,2,3} -> use{1,2,3}                      전체 통합 (§3.7 의 조건 B)
      D  pool{1,2}   -> use{1,2}                        **신규.** 같은 단지·기종
      E  pool{1,3}   -> use{1,3}                        G2 행을 pool 에서 뺀 것 (B 와 비교)
      F  pool{1,2}->use{1,2} + pool{1,2,3}->use{3}      혼합. 두 기전을 각각 최적으로

사전 예측 (결과 보기 전 고정 — 이것이 판정의 절반임)
  §3.7  은 통합의 이득을 **borrowed strength** 하나로 설명함 — 라벨이 33% 적은 G3 만 이득,
        라벨이 많은 G2 는 손해(3/3 음수). 이 기전만 작동한다면
        **D(G1+G2)는 이득이 없어야 함.** 둘 다 라벨이 충분하기 때문임.
  §3.21 은 두 번째 기전을 추가함 — 통합 모델이 **격자 간 공간 차이를 암묵 학습**함.
        이 기전이 작동한다면 **인접·동일기종인 G1+G2 에서 이득이 나야 함.**

  => **D 에서 G1·G2 가 B 대비 개선되면 §3.21 의 공간 기전이 지배적임.**
     개선이 없으면 borrowed strength 가 지배적이며 §3.7 원문이 옳음.
     어느 쪽이든 통합 학습의 기전이 확정되므로 결과와 무관하게 정보가 남음.

  E vs B 는 'G2 의 행을 pool 에 넣는 것' 만의 효과를 분리함 (순수 표본 크기 효과).

판정 (§3.6 규칙 2·6)
  raw OOF 연도짝비교 vs B(v13). 부호 3/3 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.
  ⚠ 총점만 보면 상쇄로 신호가 지워짐 (§3.7 의 부수 발견). **반드시 그룹별로 읽을 것.**

실행
    python scripts/step26_joint_pairs.py --config configs/config_v13.yaml
==============================================================================
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.utils import seed_everything, CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, run_cv, run_cv_joint,
                            total_score, group_score, score_by_year, is_difference_real)

quiet_warnings()
BAR = "=" * 92
G1, G2, G3 = "kpx_group_1", "kpx_group_2", "kpx_group_3"

# 조건 정의: (라벨, [(pool 튜플, use 리스트), ...])  · 비어 있으면 전부 독립
ARMS = [
    ("A 전부독립",      []),
    ("B v13 현행",      [((G1, G2, G3), [G1, G3])]),
    ("C 전체통합",      [((G1, G2, G3), [G1, G2, G3])]),
    ("D G1+G2",        [((G1, G2), [G1, G2])]),
    ("E G1+G3 pool",   [((G1, G3), [G1, G3])]),
    ("F 혼합",          [((G1, G2), [G1, G2]), ((G1, G2, G3), [G3])]),
]


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    idc = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    vc = [c for c in df.columns if c not in (idc | {"latitude", "longitude"})]
    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=vc)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    agg = df.groupby("forecast_kst_dtm")[vc].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.reset_index().merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if "data_available_kst_dtm" in df.columns:
        av = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step26")
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]; seed = cfg["seed"]
    mt, mp = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    esr = cfg.get("early_stopping_rounds", 30)
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + 피처 + 정제 타깃 (v13 구성 그대로)")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))

    answer = df[targets].copy()
    ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
    df = df.merge(ct, on="forecast_kst_dtm", how="left")
    scale = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        tgt = df[f"{g}_cf"].to_numpy(float) * cap
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * cap)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * cap)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"  행 {len(df):,}  피처 {len(feats)}  top-k {args.top_k}")
    print(f"  라벨 수: " + "  ".join(f"{g.replace('kpx_','')} {int(df[g].notna().sum()):,}" for g in targets))

    # ---------------- 독립 모델은 1회만 학습해 전 조건이 공유 ----------------
    print(BAR); print("STEP 2  독립 모델 1회 학습 (전 조건 공유)")
    indep, _, _, _ = run_cv(df, feats, targets, mt, mp, scheme="loyo", es_rounds=esr,
                            es_mode="refit", seed=seed, verbose=False, top_k=args.top_k)
    print(f"  완료 ({time.time()-t0:.0f}s)")

    # ---------------- 필요한 pool 만 학습 ----------------
    pools = sorted({p for _, specs in ARMS for p, _ in specs}, key=len)
    print(BAR); print(f"STEP 3  통합 모델 {len(pools)}개 학습 (pool 별 1회)")
    JOINT = {}
    for p in pools:
        tl = list(p)
        oj, _, _ = run_cv_joint(df, feats, tl, mt, mp, scheme="loyo", es_rounds=esr,
                                es_mode="refit", seed=seed, verbose=False, top_k=args.top_k)
        JOINT[p] = oj
        n = int(sum(df[g].notna().sum() for g in tl))
        print(f"  pool{{{', '.join(g.replace('kpx_group_','G') for g in tl)}}}  "
              f"학습가능 {n:,}행   ({time.time()-t0:.0f}s)")

    # ---------------- 조건별 OOF 조립 ----------------
    print(BAR); print("STEP 4  조건별 raw OOF")
    oofs = {}
    for name, specs in ARMS:
        o = indep.copy()
        for p, use in specs:
            for g in use:
                o[g] = JOINT[p][g]
        for g in targets:
            o[g] = np.clip(o[g] * scale[g], 0, CAPACITY_KWH[g])
        oofs[name] = o
        o.to_csv(odir / f"oof_{name.split()[0]}.csv")
        s = total_score(answer, o, targets)
        gs = [group_score(answer[g].to_numpy(float), o[g].to_numpy(float), CAPACITY_KWH[g])[0]
              for g in targets]
        print(f"  {name:14s} raw {s[0]:.4f}  |  " +
              "  ".join(f"{g.replace('kpx_group_','G')} {v:.4f}" for g, v in zip(targets, gs)))

    # ---------------- 판정 ----------------
    base = "B v13 현행"
    print(BAR); print(f"STEP 5  판정 — {base} 기준 연도짝비교 (§3.6 규칙 2·6)")
    print("  ⚠ §3.7 부수 발견: 총점만 보면 상쇄로 신호가 지워짐. **그룹별로 읽을 것.**")
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)
    for name in [n for n, _ in ARMS if n != base]:
        print(f"\n  [{name}  vs  {base}]")
        res = is_difference_real(df, answer, oofs[name], oofs[base], targets,
                                 name_a=name, name_b=base)
        gpos, line = 0, []
        for g in targets:
            cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
            sb = group_score(a, oofs[base][g].to_numpy(float), cap)[0]
            sa = group_score(a, oofs[name][g].to_numpy(float), cap)[0]
            gpos += int(np.isfinite(sa - sb) and sa > sb)
            line.append(f"{g.replace('kpx_group_','G')} {sa-sb:+.4f}")
        ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
        print("   그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")
        print(f"   => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}")

    # ---------------- 기전 판별 ----------------
    print(BAR); print("STEP 6  기전 판별 — borrowed strength 인가 공간 정보인가")
    cap1, cap2 = CAPACITY_KWH[G1], CAPACITY_KWH[G2]
    a1, a2 = answer[G1].to_numpy(float), answer[G2].to_numpy(float)
    d1 = (group_score(a1, oofs["D G1+G2"][G1].to_numpy(float), cap1)[0]
          - group_score(a1, oofs[base][G1].to_numpy(float), cap1)[0])
    d2 = (group_score(a2, oofs["D G1+G2"][G2].to_numpy(float), cap2)[0]
          - group_score(a2, oofs[base][G2].to_numpy(float), cap2)[0])
    print(f"  D(G1+G2) − B  :  G1 {d1:+.4f}   G2 {d2:+.4f}")
    if d1 > 0.001 or d2 > 0.001:
        print("  => **§3.21 의 공간 기전이 지배적임.** 라벨이 충분한 두 그룹의 통합이 이득을 냄.")
        print("     통합 학습은 borrowed strength 만이 아니라 격자 간 공간 차이를 배우고 있음.")
    else:
        print("  => **borrowed strength 가 지배적임.** §3.7 원문이 옳음.")
        print("     라벨이 충분한 그룹끼리의 통합에는 이득이 없음. 공간 기전은 부차적임.")
    print("\n  [E vs B] = 'G2 의 행을 pool 에 넣는 효과' 만의 분리 (순수 표본 크기 효과)")
    for g in (G1, G3):
        cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
        de = (group_score(a, oofs["E G1+G3 pool"][g].to_numpy(float), cap)[0]
              - group_score(a, oofs[base][g].to_numpy(float), cap)[0])
        print(f"    {g.replace('kpx_group_','G')}: {de:+.4f}   "
              f"({'G2 행이 도움이 됨' if de < 0 else 'G2 행이 오히려 방해'})")

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("통과 시: config_v16.yaml 로 본학습(--baseline-oof 게이트 포함) 후 LB 1회.")


if __name__ == "__main__":
    main()