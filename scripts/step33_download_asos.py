"""
scripts/step33_download_asos.py
==============================================================================
[도구] 기상청 API허브에서 ASOS 시간자료를 내려받아 data/external/asos_hourly.csv 생성

규칙 4 (외부 데이터) 소명 자료를 겸함 — 이 스크립트 하나로 재현 가능해야 함.
  * 출처: 기상청 API허브 (https://apihub.kma.go.kr) 지상 종관기상관측(ASOS) 시간자료
  * 접근: 무료 회원가입 + API 키 (누구나 접근 가능). 키는 코드에 넣지 않고
    환경변수 KMA_API_KEY 또는 --api-key 로 받음.
  * 라이선스: 공공누리 제1유형 (출처표시, 상업적 이용 가능)
  * 시점: 관측 즉시 준실시간 공개. 피처는 각 행의 예측기준시점(전날 13:00) 보다
    1시간 이상 과거의 관측만 사용함 (features_obs.py 참조).

관측소 선정: 관측지점 메타데이터를 받아 단지 중심(37.283N, 128.963E — info.xlsx
터빈 좌표 평균)에서 가까운 순으로 자동 선정. 거리와 지점번호를 출력하므로
그대로 소명 자료가 됨.

실행
    set KMA_API_KEY=<발급키>          (PowerShell: $env:KMA_API_KEY="<발급키>")
    python scripts/step33_download_asos.py
    python scripts/step33_download_asos.py --n-stations 3 --start 2021-12-01 --end 2025-12-31
==============================================================================
"""
import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://apihub.kma.go.kr/api/typ01/url"
FARM = (37.2830, 128.9630)          # info.xlsx 터빈 좌표 평균 (태백 가덕산·원동)


def fetch(url, retries=3):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("euc-kr", errors="replace")
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def dist_km(lat, lon):
    return float(np.hypot((lat - FARM[0]) * 110.57,
                          (lon - FARM[1]) * 111.32 * np.cos(np.radians(FARM[0]))))


def station_list(key):
    """ASOS 지점 메타데이터 -> (거리, 지점번호, 이름, 위도, 경도) 정렬 리스트."""
    txt = fetch(f"{API}/stn_inf.php?inf=SFC&tm=202401010000&help=0&authKey={key}")
    rows = []
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        try:
            stn, lon, lat = int(p[0]), float(p[1]), float(p[2])
        except (ValueError, IndexError):
            continue
        name = next((t for t in p if any("가" <= c <= "힣" for c in t)), "?")
        rows.append((dist_km(lat, lon), stn, name, lat, lon))
    rows.sort()
    return rows


def parse_hourly(txt):
    """kma_sfctm3 응답 파싱. typ01 고정 컬럼: TM STN WD WS ... PA ... TA TD HM"""
    out = []
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 14 or not p[0].isdigit():
            continue
        try:
            wd36 = float(p[2])                    # typ01 풍향은 36방위 (1=10도)
            ta = float(p[11])
            out.append({
                "kst_dtm": pd.to_datetime(p[0], format="%Y%m%d%H%M"),
                "stn": int(p[1]),
                "wd": (wd36 * 10.0) % 360.0 if wd36 >= 0 else np.nan,
                "ws": float(p[3]),
                "pa": float(p[7]),
                # -9.0 은 결측 코드. 실제 -9.0C 소수 표본을 잃지만 결측 오염보다 안전
                "ta": np.nan if abs(ta + 9.0) < 1e-9 or ta < -45 else ta,
                "hm": float(p[13]),
            })
        except (ValueError, IndexError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("KMA_API_KEY"))
    ap.add_argument("--start", default="2021-12-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--n-stations", type=int, default=2)
    ap.add_argument("--out", default="./data/external/asos_hourly.csv")
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("⛔ API 키가 없다. 환경변수 KMA_API_KEY 또는 --api-key 로 줄 것.")

    print("🔎 관측소 메타데이터 조회...")
    stns = station_list(args.api_key)
    print("   단지 최근접 6개:")
    for d, stn, nm, lat, lon in stns[:6]:
        print(f"     {d:6.1f} km  #{stn:<5} {nm:<8} ({lat:.3f}, {lon:.3f})")
    pick = stns[:args.n_stations]
    print(f"   선정: {', '.join(f'#{s[1]} {s[2]} ({s[0]:.1f}km)' for s in pick)}")

    months = pd.period_range(args.start, args.end, freq="M")
    rows = []
    for _, stn, nm, _, _ in pick:
        for m in months:
            t1 = m.start_time.strftime("%Y%m%d%H%M")
            t2 = min(m.end_time, pd.Timestamp(args.end) + pd.Timedelta(hours=23)) \
                .strftime("%Y%m%d%H") + "59"
            url = (f"{API}/kma_sfctm3.php?tm1={t1}&tm2={t2}&stn={stn}"
                   f"&help=0&authKey={args.api_key}")
            got = parse_hourly(fetch(url))
            rows.extend(got)
            print(f"   #{stn} {m}: {len(got):,}행", end="\r")
            time.sleep(0.25)
        print()

    df = pd.DataFrame(rows).drop_duplicates(["kst_dtm", "stn"]).sort_values(["kst_dtm", "stn"])
    # 결측 코드(-9 계열) 정리
    for c, lo, hi in [("ws", 0, 60), ("wd", 0, 360), ("ta", -45, 45),
                      ("hm", 0, 100), ("pa", 700, 1100)]:
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"💾 {out}  행 {len(df):,}  기간 {df.kst_dtm.min()} ~ {df.kst_dtm.max()}")
    meta = pd.DataFrame(pick, columns=["dist_km", "stn", "name", "lat", "lon"])
    meta.to_csv(out.parent / "asos_stations.csv", index=False, encoding="utf-8-sig")
    print(f"💾 {out.parent/'asos_stations.csv'}  (소명용 관측소 선정 근거)")


if __name__ == "__main__":
    main()
