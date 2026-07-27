"""
scripts/step11_blend_sweep.py
==============================================================================
Tweedie(v13) x MAE+W(v14) 혼합 — 후처리 여지를 남기면서 raw 정확도를 가져온다

step10 이 확정한 구조
    후처리 여지   A(Tweedie) +0.0236 (3/3 양수, std 0.0062)
                  D(MAE+W)   +0.0017 (2022 는 -0.0023, std 0.0042)
    LB 역산       raw14 - raw13 = +0.0090   (MAE 모델이 2025 에서도 raw 로는 더 좋았다)
    결론          밴드 몰아넣기는 총량이 정해진 상금(+0.024)이고, 후처리가 모델보다
                  그걸 더 잘 캔다. MAE 는 그 상금을 +0.0090 만 캐고 나머지를 막아버렸다.

가설
    p = (1-w)*Tweedie + w*MAE 를 섞으면
      - raw 정확도는 Tweedie 단독보다 좋고 (MAE 쪽 이득을 일부 가져옴)
      - 밴드에 완전히 붙지는 않아 후처리 여지가 남는다
    두 항의 합이 최대가 되는 w 가 0 이나 1 이 아닌 내부점이면 혼합이 이긴다.
    step10 PART 1 이 '평균은 이득' 을 이미 보였으므로 방향은 맞는다.

사후선택 방지 (이게 이 스크립트의 핵심 설계)
    w 를 3년 전체에서 고르고 3년 전체에서 평가하면 당연히 좋아 보인다.
    그래서 두 값을 따로 보고한다.
      oracle w  : 그 해를 포함해 고른 w        <- 낙관 상한. 절대 이 값으로 판단하지 말 것
      nested w  : 그 해를 빼고 나머지 2년에서 고른 w -> 그 해에 적용  <- 정직한 값
    후처리 파라미터도 같은 방식으로 그 해를 빼고 적합한다 (이중 중첩).

학습 없음. 저장된 OOF 만 읽는다. 수십 초면 끝난다.

실행:
    python scripts/step11_blend_sweep.py --config configs/config_v13.yaml
    python scripts/step11_blend_sweep.py --config configs/config_v13.yaml --make-submission
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
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, total_score, group_score
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
WGRID = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]


def load_oof(path, index, targets):
    d = pd.read_csv(path, index_col=0)
    return d.reindex(index=index, columns=targets).astype(float)


def blend(a, b, w, targets):
    out = a.copy()
    for g in targets:
        v = (1.0 - w) * a[g].to_numpy(float) + w * b[g].to_numpy(float)
        out[g] = np.clip(v, 0, CAPACITY_KWH[g])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="train_dir / targets 만 읽는다")
    ap.add_argument("--dir-a", default="./saved_models/v13", help="Tweedie 쪽")
    ap.add_argument("--dir-b", default="./saved_models/v14", help="MAE+W 쪽")
    ap.add_argument("--make-submission", action="store_true",
                    help="raw_test_preds.csv 두 개를 섞고 후처리해서 제출 파일까지 만든다")
    ap.add_argument("--out-name", default="submit_v15_blend.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    da, db = Path(args.dir_a), Path(args.dir_b)

    # ---- 정답 라벨. 피처 파이프라인 없이 train_labels.csv 만 읽는다.
    #      train.py 가 base=labels 에 좌결합(left merge)만 하므로 행 순서가 동일하다.
    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]

    oa = load_oof(da / "oof_preds.csv", df.index, targets)
    ob = load_oof(db / "oof_preds.csv", df.index, targets)
    print(BAR)
    print(f"A(Tweedie) {da}   B(MAE+W) {db}")
    print(f"  행 {len(df):,}  연도 {years}")
    sa, sb = total_score(answer, oa, targets), total_score(answer, ob, targets)
    print(f"  raw OOF   A {sa[0]:.4f} (1-NMAE {sa[1]:.4f} / FICR {sa[2]:.4f})")
    print(f"            B {sb[0]:.4f} (1-NMAE {sb[1]:.4f} / FICR {sb[2]:.4f})")

    # ---------------------------------------------------------------- 사전 계산
    # (w, 평가연도) -> raw 점수 / 후처리 후 점수.  후처리는 그 해를 빼고 적합한다.
    raw_ys, post_ys = {}, {}
    for w in WGRID:
        bl = blend(oa, ob, w, targets)
        for y in years:
            te = (yrs == y)
            tr = ~te & answer.notna().any(axis=1).to_numpy()
            pp = optimize_postprocessing(answer.loc[tr], bl.loc[tr], mode="piecewise", verbose=False)
            post = apply_postprocessing(bl.loc[te].copy(), pp)
            raw_ys[(w, y)] = total_score(answer.loc[te], bl.loc[te], targets)[0]
            post_ys[(w, y)] = total_score(answer.loc[te], post, targets)[0]
        print(f"  w={w:.2f} 계산 완료 ({time.time()-t0:.0f}s)", end="\r")

    print(BAR)
    print("혼합비 w 별 (w=0 은 Tweedie 단독, w=1 은 MAE+W 단독)")
    print(f"  {'w':>6s}{'raw':>10s}{'후처리':>10s}{'후처리여지':>12s}   연도별 후처리 후")
    for w in WGRID:
        r = np.mean([raw_ys[(w, y)] for y in years])
        p = np.mean([post_ys[(w, y)] for y in years])
        det = " ".join(f"{y}:{post_ys[(w, y)]:.4f}" for y in years)
        mark = ""
        if w == 0.0:
            mark = "  <- v13"
        elif w == 1.0:
            mark = "  <- v14"
        print(f"  {w:6.2f}{r:10.4f}{p:10.4f}{p-r:12.4f}   {det}{mark}")

    # ---------------------------------------------------------------- 판정
    print(BAR)
    print("판정")
    orc_w = max(WGRID, key=lambda w: np.mean([post_ys[(w, y)] for y in years]))
    orc_s = np.mean([post_ys[(orc_w, y)] for y in years])
    print(f"  [oracle]  최적 w={orc_w:.2f}  점수 {orc_s:.4f}   <- 낙관 상한. 판단 근거로 쓰지 말 것")

    # nested: 평가 연도를 빼고 나머지 연도에서 w 를 고른다
    nested, picks = [], []
    for y in years:
        others = [z for z in years if z != y]
        wsel = max(WGRID, key=lambda w: np.mean([post_ys[(w, z)] for z in others]))
        nested.append(post_ys[(wsel, y)])
        picks.append((y, wsel, post_ys[(wsel, y)]))
    nv = np.array(nested, float)
    base = np.array([post_ys[(0.0, y)] for y in years], float)     # v13 = Tweedie 단독
    print(f"  [nested]  연도별 선택 " + ", ".join(f"{y}->w={w:.2f}({s:.4f})" for y, w, s in picks))
    print(f"            평균 {nv.mean():.4f}   vs  Tweedie 단독 {base.mean():.4f}   "
          f"차이 {nv.mean()-base.mean():+.4f}")

    d = nv - base
    d_nz = d[np.abs(d) > 1e-9]
    ok = len(d_nz) >= 2 and (np.all(d_nz > 0) or np.all(d_nz < 0)) and \
        abs(d.mean()) > (d.std(ddof=1) if len(d) > 1 else np.inf)
    print(f"            연도별 차이 [" + " ".join(f"{v:+.4f}" for v in d) + "]  "
          f"부호일치={len(d_nz) >= 2 and (np.all(d_nz > 0) or np.all(d_nz < 0))}  "
          f"표준편차={d.std(ddof=1):.4f}")
    print(f"  => {'✅ 신호. 혼합이 v13 을 이긴다.' if ok else '❌ 노이즈. 제출 1회를 쓸 근거가 안 된다.'}")
    if ok:
        # 실제 제출에 쓸 w 는 3년 전체에서 고른다 (nested 는 '그 절차가 통하는가' 의 검정용)
        print(f"     제출용 w 는 3년 전체 기준 {orc_w:.2f} 를 쓴다 "
              f"(nested 는 절차의 정직한 평가치이고, 실제 적합은 가진 데이터를 다 쓴다)")

    print(BAR)
    print("그룹별 (후처리 후, 3년 평균)")
    print(f"  {'w':>6s}" + "".join(f"{g.replace('kpx_',''):>14s}" for g in targets))
    for w in [0.0, orc_w, 1.0] if orc_w not in (0.0, 1.0) else [0.0, 1.0]:
        bl = blend(oa, ob, w, targets)
        ss = []
        for g in targets:
            vals = []
            for y in years:
                te = (yrs == y)
                tr = ~te & answer.notna().any(axis=1).to_numpy()
                pp = optimize_postprocessing(answer.loc[tr], bl.loc[tr], mode="piecewise", verbose=False)
                post = apply_postprocessing(bl.loc[te].copy(), pp)
                v = group_score(answer.loc[te, g].to_numpy(float),
                                post[g].to_numpy(float), CAPACITY_KWH[g])[0]
                if np.isfinite(v):
                    vals.append(v)
            ss.append(np.mean(vals) if vals else np.nan)
        print(f"  {w:6.2f}" + "".join(f"{v:14.4f}" for v in ss))

    # ---------------------------------------------------------------- 제출
    if args.make_submission:
        print(BAR)
        ta, tb = da / "raw_test_preds.csv", db / "raw_test_preds.csv"
        if not (ta.exists() and tb.exists()):
            print(f"  ⚠ raw_test_preds.csv 가 없다 ({ta} / {tb}). "
                  f"main/inference.py 를 각 config 로 한 번씩 돌려서 만들 것.")
        else:
            pa, pb = pd.read_csv(ta), pd.read_csv(tb)
            assert len(pa) == len(pb), "두 예측 행 수가 다르다"
            sub = pa.copy()
            for g in targets:
                sub[g] = np.clip((1 - orc_w) * pa[g].to_numpy(float)
                                 + orc_w * pb[g].to_numpy(float), 0, CAPACITY_KWH[g])
            # 후처리 파라미터는 3년 OOF 전체(혼합본)로 적합한다 = train.py 와 같은 절차
            bl_full = blend(oa, ob, orc_w, targets)
            pp = optimize_postprocessing(answer, bl_full, mode="piecewise", verbose=False)
            sub = apply_postprocessing(sub, pp)
            sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / args.out_name
            sub.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  🚀 제출 파일 저장: {out}   (w={orc_w:.2f}, 후처리 piecewise)")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()