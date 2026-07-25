"""
==============================================================================
SCADA 1단계 점검 스크립트.

지금은 학습 파이프라인을 전혀 건드리지 않는다. 그냥 SCADA 를 읽어서
  - 시각 정렬이 맞는지
  - 라벨이 재구성되는지
  - 정지/출력제한이 얼마나 있는지
만 확인하고 결과 파일 2개를 저장한다.

실행:
    python scripts/step1_scada_check.py --train-dir ./data/train

끝나면 화면에 나온 리포트를 그대로 복사해서 알려주면 된다.
==============================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scada import (ALIGN_MIN, load_scada, verify_alignment, to_hourly, fit_reference_curve,
                       availability_mask, refine_outage_cause, availability_from_status,
                       group_available_capacity, fit_loss_factor, clean_target,
                       detect_wd_offset, GROUP_TURBINES, CAPACITY_KWH, _cols)

BAR = "=" * 74


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="./train", help="train_labels.csv 등이 있는 폴더")
    ap.add_argument("--out-dir", default="./data/scada_derived")
    args = ap.parse_args()

    tdir = Path(args.train_dir)
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)

    need = ["train_labels.csv", "scada_vestas_train.csv", "scada_unison_train.csv"]
    for f in need:
        if not (tdir / f).exists():
            print(f"❌ {tdir/f} 가 없다. --train-dir 경로를 확인할 것.")
            print(f"   현재 폴더 내용: {[p.name for p in tdir.glob('*.csv')][:10]}")
            return

    print(BAR); print("STEP 1/6  파일 읽기")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ves = load_scada(str(tdir / "scada_vestas_train.csv"), "vestas")
    uni = load_scada(str(tdir / "scada_unison_train.csv"), "unison")
    print(f"  labels {labels.shape}  {labels.kst_dtm.min()} ~ {labels.kst_dtm.max()}")
    print(f"  vestas {ves.shape}  {ves.kst_dtm.min()} ~ {ves.kst_dtm.max()}")
    print(f"  unison {uni.shape}  {uni.kst_dtm.min()} ~ {uni.kst_dtm.max()}")
    for g in GROUP_TURBINES:
        print(f"  라벨 {g}: 유효 {labels[g].notna().sum():,} / 결측 {labels[g].isna().sum():,}")

    print(BAR); print("STEP 2/6  시각 정렬 검증  (기대: vestas +50분 / unison +60분)")
    offsets = {}
    for maker, sc, grp in [("vestas", ves, "kpx_group_1"), ("unison", uni, "kpx_group_3")]:
        try:
            r = verify_alignment(sc, labels, maker, grp)
            print(f"  [{maker} / {grp}] 상위 3개")
            print(r.head(3).to_string(index=False))
            best = int(r.iloc[0].offset_min)
            offsets[maker] = best
            flag = "✅" if r.iloc[0].r2 >= 0.99 else "⚠️"
            print(f"    {flag} 채택 오프셋 {best:+d}분, R²={r.iloc[0].r2:.5f}"
                  f"{'' if best == ALIGN_MIN[maker] else f'  (기본값 {ALIGN_MIN[maker]}분과 다름 -> 알려줄 것)'}")
        except Exception as e:
            offsets[maker] = ALIGN_MIN[maker]
            print(f"    ❌ {maker} 정렬 검증 실패({e}) -> 기본값 {ALIGN_MIN[maker]}분 사용")

    print(BAR); print("STEP 3/6  10분 -> 시간 집계")
    hv = to_hourly(ves, "vestas", align_min=offsets.get("vestas"))
    hu = to_hourly(uni, "unison", align_min=offsets.get("unison"))
    print(f"  vestas 시간행 {len(hv):,}   unison 시간행 {len(hu):,}")
    print(f"  vestas wtg01 시간 kWh: 중앙값 {hv['vestas_wtg01_power_kw10m'].median():.0f} / "
          f"최대 {hv['vestas_wtg01_power_kw10m'].max():.0f}  (정격 3600)")
    print(f"  unison wtg01 시간 kWh: 중앙값 {hu['unison_wtg01_power_kw10m'].median():.0f} / "
          f"최대 {hu['unison_wtg01_power_kw10m'].max():.0f}  (정격 4200)")

    print(BAR); print("STEP 4/6  기준 파워커브  (유효 bin 이 40개 이상이어야 정상)")
    cv = fit_reference_curve(hv, "vestas")
    cu = fit_reference_curve(hu, "unison")

    print(BAR); print("STEP 5/6  가동률 판정")
    stv = refine_outage_cause(availability_mask(hv, "vestas", curve=cv, verbose=False)[0], hv, "vestas")
    stu = refine_outage_cause(availability_mask(hu, "unison", curve=cu, verbose=False)[0], hu, "unison")
    for name, st in [("vestas", stv), ("unison", stu)]:
        vc = pd.Series(st.to_numpy().ravel()).value_counts()
        tot = vc.sum()
        print(f"  [{name}] " + "  ".join(f"{k}:{v:,}({v/tot*100:.1f}%)" for k, v in vc.items()))
        d = vc.get("derated", 0) / tot * 100
        if d > 8:
            print(f"    ⚠️ derated 비율 {d:.1f}% 로 높다 (5% 이하 기대). derate_frac 조정 필요할 수 있음.")

    avail = group_available_capacity({"vestas": availability_from_status(stv),
                                      "unison": availability_from_status(stu)})
    print("\n  그룹별 평균 가용률:")
    for g in GROUP_TURBINES:
        if g in avail:
            fr = avail[g] / CAPACITY_KWH[g]
            print(f"    {g}: 평균 {fr.mean()*100:.2f}%   100%인 시간 {(fr>=0.999).mean()*100:.1f}%   "
                  f"50% 미만 {(fr<0.5).mean()*100:.2f}%")

    print(BAR); print("STEP 6/6  계통 손실계수 + 정제 타깃  (head 에서 k=0.987 / 0.989)")
    sc_sum = {}
    for g, turbs in GROUP_TURBINES.items():
        src = hv if turbs[0][0] == "vestas" else hu
        cols = [_cols(m, i)[0] for m, i in turbs if _cols(m, i)[0] in src.columns]
        if cols:
            sc_sum[g] = pd.Series(src[cols].sum(axis=1).to_numpy(), index=pd.to_datetime(src["kst_dtm"]))
    sc_sum = pd.DataFrame(sc_sum)
    loss = fit_loss_factor(labels.set_index("kst_dtm"), sc_sum)
    print(pd.DataFrame(loss).T.round(4).to_string())

    # --- SCADA 재구성값을 그대로 '예측' 이라 치면 대회 점수가 얼마나 나오나 ---
    # RMSE 는 데이터 해독이 맞았나 보는 지표라 직관이 안 온다. 대회 언어로 번역한다.
    # k 를 같은 데이터에 맞추고 같은 데이터로 평가하면 자기충족이므로,
    # 나머지 연도에서 k 를 적합해 해당 연도에 적용하는 out-of-sample 방식으로 낸다.
    # (파라미터 1개 / 26,000행 이라 편향은 O(1/n) 이지만, 숫자는 방어 가능해야 한다.)
    try:
        from src.validation import group_score
        print("\n  [SCADA 재구성 품질을 대회 산식으로 환산]")
        print("   k 를 '다른 연도' 에서 적합해 해당 연도에 적용 (out-of-sample). 1등 = 0.67069")
        tot_in, tot_out = [], []
        for g in GROUP_TURBINES:
            if g not in sc_sum or g not in loss:
                continue
            j = labels.set_index("kst_dtm")[[g]].join(
                sc_sum[[g]].rename(columns={g: "_sc"}), how="inner").dropna()
            yrs = j.index.year.to_numpy()
            k_in = loss[g]["k"]
            pred_out = np.full(len(j), np.nan)
            ks = {}
            for y in np.unique(yrs):
                te, tr = (yrs == y), (yrs != y)
                if tr.sum() < 100 or te.sum() < 100:
                    continue
                x, yv = j["_sc"].to_numpy()[tr], j[g].to_numpy()[tr]
                ok = x > CAPACITY_KWH[g] * 0.05
                ky = float(np.median(yv[ok] / x[ok])); ks[int(y)] = round(ky, 5)
                pred_out[te] = ky * j["_sc"].to_numpy()[te]
            m = np.isfinite(pred_out)
            s_i = group_score(j[g].to_numpy(), k_in * j["_sc"].to_numpy(), CAPACITY_KWH[g])
            s_o = group_score(j[g].to_numpy()[m], pred_out[m], CAPACITY_KWH[g])
            tot_in.append(s_i); tot_out.append(s_o)
            print(f"     {g}: in-sample {s_i[0]:.5f}  |  out-of-sample {s_o[0]:.5f}  "
                  f"(차이 {s_o[0]-s_i[0]:+.5f})   연도별 k={ks}")
        if tot_out:
            ai, ao = np.array(tot_in), np.array(tot_out)
            print(f"     {'3그룹 평균':<11}: in-sample {ai[:,0].mean():.5f}  |  "
                  f"out-of-sample {ao[:,0].mean():.5f}  (차이 {ao[:,0].mean()-ai[:,0].mean():+.5f})")
            print(f"     1-NMAE {ao[:,1].mean():.5f}   FICR {ao[:,2].mean():.5f}")
            print("     -> 두 값 차이가 0에 가까우면 k 적합에 의한 과적합은 없다는 뜻.")
            print("     ⚠️ 이 점수는 리더보드에서 낼 수 있는 값이 아니다. 2025년엔 SCADA 가 없다.")
            print("        의미는 '라벨 해독이 맞았다' 뿐이고, 모델 성능과 무관하다.")
    except Exception as e:
        print(f"   (환산 생략: {e})")

    tgt = clean_target(labels, avail, loss)
    for g in GROUP_TURBINES:
        c = f"{g}_cf"
        if c in tgt:
            print(f"  {g}_cf: 유효 {tgt[c].notna().sum():,}  평균 {tgt[c].mean():.4f}  "
                  f"제외(가용<50%) {tgt[c].isna().sum() - labels[g].isna().sum():,}행")

    avail.to_csv(odir / "available_capacity.csv", index=False, encoding="utf-8-sig")
    tgt.to_csv(odir / "clean_target.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detect_wd_offset(hv, "vestas")).to_csv(odir / "wd_offset_vestas.csv", encoding="utf-8-sig")
    pd.DataFrame(detect_wd_offset(hu, "unison")).to_csv(odir / "wd_offset_unison.csv", encoding="utf-8-sig")
    print(f"\n💾 저장 완료 -> {odir}/  (available_capacity.csv, clean_target.csv, wd_offset_*.csv)")
    print(BAR); print("여기까지 나온 화면을 그대로 복사해서 보내주면 다음 단계로 간다.")


if __name__ == "__main__":
    main()