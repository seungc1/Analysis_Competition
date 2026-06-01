"""
=======================================================
팀원 A | 1~2주차: 데이터 결합 및 기초 정제
2026 국가데이터 활용대회 — 고령층 경제 이질성 진단
=======================================================

핵심 전략:
- 가계금융복지조사를 기반으로 고령층(65+) 추출
- 민간데이터(카드매출, 신용정보)를 가구 단위로 JOIN
- "자산은 많으나 현금이 없는" 패턴을 포착하기 위한 변수 설계
- SDC 우대 데이터 최대 활용 → 심사 'データ활용성' 항목 가점
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 경로 설정 (SDC 환경에 맞게 조정) ────────────────────────────────
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1. 가계금융복지조사 로드 및 고령층 필터링
# (SDC 제공 마이크로데이터 — 심사 우대 데이터)
# ════════════════════════════════════════════════════════════════

def load_hfws(year: int) -> pd.DataFrame:
    """
    가계금융복지조사(Household Finance and Welfare Survey) 연도별 로드.
    
    주요 변수:
      가구주_연령, 순자산, 부동산자산, 금융자산, 부채,
      근로소득, 사업소득, 재산소득, 이전소득, 총소득
    """
    path = DATA_DIR / f"hfws_{year}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["조사연도"] = year
    return df


def prepare_elderly_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    65세 이상 고령 가구 추출 및 연령 그룹 분류.

    연구보고서(제2534호) 기준:
      전기 고령자: 65~74세 (국민연금 성숙기, 소득 개선 빠름)
      후기 고령자: 75세 이상 (빈곤 고착형, OECD 최하위 수준)
    """
    df = df[df["가구주_연령"] >= 65].copy()

    # 연령 그룹 분류 (분석의 핵심 변수)
    df["연령그룹"] = np.where(
        df["가구주_연령"] <= 74,
        "전기고령자(65-74)",
        "후기고령자(75+)"
    )

    # ── 부동산 편중도 계산 ──────────────────────────────────────
    # 연구보고서: 부동산 80% 이상 편중이 '자산-소득 괴리'의 핵심
    df["부동산편중도"] = np.where(
        df["총자산"] > 0,
        df["부동산자산"] / df["총자산"],
        np.nan
    )
    df["부동산편중_고위험"] = df["부동산편중도"] >= 0.80  # 보고서 기준

    # ── 부채비율 ──────────────────────────────────────────────
    df["부채비율"] = np.where(
        df["총자산"] > 0,
        df["부채"] / df["총자산"],
        np.nan
    )

    # ── 이상치 플래그: 단순 삭제 금지 ─────────────────────────
    # "소득=0이면서 부동산 보유" = 자산부유-소득빈곤층의 전형
    # → 삭제하면 분석의 핵심 대상이 사라짐
    df["이상치_플래그"] = (
        (df["총소득"] <= 0) &
        (df["부동산자산"] > 0)
    )
    df["처리방침"] = np.where(
        df["이상치_플래그"],
        "유지_핵심대상",   # 삭제 금지
        "정상"
    )

    return df


# ════════════════════════════════════════════════════════════════
# STEP 2. 시계열 병합 (2012~2024)
# ════════════════════════════════════════════════════════════════

def build_panel(years: list[int]) -> pd.DataFrame:
    """연도별 가계금융복지조사 패널 데이터 구성."""
    frames = []
    for year in years:
        try:
            df = load_hfws(year)
            df = prepare_elderly_cohort(df)
            frames.append(df)
            print(f"✓ {year}년 로드 완료: {len(df):,}가구")
        except FileNotFoundError:
            print(f"⚠ {year}년 파일 없음 — 건너뜀")
    
    panel = pd.concat(frames, ignore_index=True)
    print(f"\n총 패널: {len(panel):,}가구-연도 관측치")
    return panel


# ════════════════════════════════════════════════════════════════
# STEP 3. 민간 데이터 연계 (카드매출 + 신용정보)
# ════════════════════════════════════════════════════════════════

def load_card_sales(year: int) -> pd.DataFrame:
    """
    카드 매출 데이터 로드 (SDC 민간자료 67종 중 해당).
    
    활용 목적:
      실제 소비 수준 파악 → 소득통계와의 괴리 입증
      D-Value(실질소비여력) = 카드매출 / 신고소득 계산
    """
    path = DATA_DIR / f"card_sales_{year}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 필요 컬럼: 가구ID, 연간_카드지출, 업종별_지출(식품/의료/여가 등)
    return df


def load_credit_info(year: int) -> pd.DataFrame:
    """
    신용정보 데이터 로드 (대출 잔액, 연체 여부).
    
    활용 목적:
      유동성 함정 지수 계산 핵심 변수
      고가주택 보유 + 카드 연체 = 자산부유-현금빈곤 직접 증거
    """
    path = DATA_DIR / f"credit_info_{year}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 필요 컬럼: 가구ID, 대출잔액, 연체여부, 연체횟수
    return df


def merge_private_data(hfws_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """가계조사 + 카드매출 + 신용정보 가구 단위 병합."""
    
    try:
        card_df = load_card_sales(year)
        credit_df = load_credit_info(year)

        df = hfws_df.merge(card_df, on="가구ID", how="left")
        df = df.merge(credit_df, on="가구ID", how="left")
        print(f"  → 카드매출 매칭률: {df['연간_카드지출'].notna().mean():.1%}")
        print(f"  → 신용정보 매칭률: {df['연체여부'].notna().mean():.1%}")

    except FileNotFoundError as e:
        print(f"  ⚠ 민간데이터 파일 없음: {e}")
        # 더미 컬럼 생성 (구조 유지용)
        df = hfws_df.copy()
        df["연간_카드지출"] = np.nan
        df["연체여부"] = np.nan

    return df


# ════════════════════════════════════════════════════════════════
# STEP 4. 건축물대장 연계 (자산 노후도 변수)
# ════════════════════════════════════════════════════════════════

def load_building_age(sgis_grid_id: str = None) -> pd.DataFrame:
    """
    건축물대장 데이터 → 격자(Grid) 단위 노후 주택 비율 계산.
    
    활용 목적:
      '자산 재생 핫스팟' 도출 (팀원 C의 SGIS 분석과 연계)
      30년 이상 노후 주택에 묶인 자산 가치 산출
    """
    path = DATA_DIR / "building_registry.csv"
    bldg = pd.read_csv(path, encoding="utf-8-sig")

    bldg["사용승인연도"] = pd.to_numeric(bldg["사용승인연도"], errors="coerce")
    bldg["노후건물"] = bldg["사용승인연도"] <= 1994  # 2024년 기준 30년 이상

    # SGIS 격자 단위 집계
    grid_stats = bldg.groupby("격자ID").agg(
        노후건물수=("노후건물", "sum"),
        전체건물수=("노후건물", "count"),
    ).reset_index()
    grid_stats["노후건물비율"] = grid_stats["노후건물수"] / grid_stats["전체건물수"]

    return grid_stats


# ════════════════════════════════════════════════════════════════
# STEP 5. 자산 분위 분류 (전체 가구 기준 vs 고령층 내부 기준)
# ════════════════════════════════════════════════════════════════

def assign_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    자산·소득 분위 계산.
    
    두 가지 기준 모두 생성:
      1) 전체 가구 기준 분위 → 세대 간 비교
      2) 고령층 내부 기준 분위 → 고령층 내 양극화 분석
         (연구보고서: 후기 고령자 상위 5%, 1% 급증이 핵심 발견)
    """
    for col, new_col in [
        ("순자산", "순자산_분위"),
        ("총소득", "소득_분위"),
    ]:
        if col in df.columns:
            df[new_col] = pd.qcut(
                df[col], q=5,
                labels=["1분위(하위20%)", "2분위", "3분위", "4분위", "5분위(상위20%)"],
                duplicates="drop"
            )

    # 고령층 내부 분위 (같은 연도 내 65+ 기준)
    for year, grp in df.groupby("조사연도"):
        mask = df["조사연도"] == year
        df.loc[mask, "순자산_고령내분위"] = pd.qcut(
            df.loc[mask, "순자산"], q=5, labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )

    return df


# ════════════════════════════════════════════════════════════════
# STEP 6. 이상치 처리 가이드라인
# ════════════════════════════════════════════════════════════════

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    팀원 A의 핵심 역량: 단순 삭제가 아닌 '의미 기반' 이상치 처리.
    
    규칙:
      ① 소득=0, 자산>0 → 유지 (자산부유-소득빈곤층 핵심)
      ② 소득<0 → 이전 연도 값으로 대체 또는 보정
      ③ 자산 극단값(상위 0.5%) → 별도 분석 레이어로 분리
      ④ 카드지출 > 소득 × 5 → 이상 소비 플래그 (단, 자산 기반 소비 가능)
    """
    # 규칙 ①: 자산부유-소득빈곤층 — 절대 삭제 금지
    asset_rich_income_poor = (
        (df["총소득"] < df["총소득"].quantile(0.20)) &
        (df["순자산"] > df["순자산"].quantile(0.60))
    )
    df.loc[asset_rich_income_poor, "페르소나_후보"] = "자산부유-현금빈곤"

    # 규칙 ③: 자산 극단값 → Winsorization (분석에서 제외하지 않고 상한선 캡)
    p995 = df["순자산"].quantile(0.995)
    df["순자산_조정"] = df["순자산"].clip(upper=p995)

    # 규칙 ④: 비정상 소비 플래그
    if "연간_카드지출" in df.columns:
        df["소비이상_플래그"] = (
            df["연간_카드지출"] > df["총소득"] * 5
        )

    print(f"\n자산부유-소득빈곤 후보: {asset_rich_income_poor.sum():,}가구")
    print(f"순자산 상한 조정: {(df['순자산'] > p995).sum():,}가구 캡 처리")

    return df


# ════════════════════════════════════════════════════════════════
# STEP 7. 기초 통계 보고 (스토리라인 검증용)
# ════════════════════════════════════════════════════════════════

def validate_story(df: pd.DataFrame, year: int = 2024):
    """
    연구보고서 주요 수치 재현 → 우리 데이터의 신뢰성 확인.
    
    보고서 기준값:
      - 부동산 보유율: 후기 고령자 71.5% (2024년)
      - 순자산 배율: 전체 평균의 1.038배 (2024년)
      - 소득 1분위 비중: 44.5% (2024년, 2012년 58.7%)
    """
    target = df[df["조사연도"] == year].copy()
    if target.empty:
        print(f"⚠ {year}년 데이터 없음")
        return

    print(f"\n{'='*50}")
    print(f"  데이터 검증 — {year}년 기준 (연구보고서 비교)")
    print(f"{'='*50}")

    # 부동산 보유율
    if "부동산자산" in target.columns:
        late_elderly = target[target["연령그룹"] == "후기고령자(75+)"]
        real_estate_rate = (late_elderly["부동산자산"] > 0).mean()
        print(f"후기 고령자 부동산 보유율: {real_estate_rate:.1%}")
        print(f"  (보고서 기준: 71.5%) → {'✓' if abs(real_estate_rate - 0.715) < 0.05 else '⚠ 괴리 확인 필요'}")

    # 부동산 편중도
    avg_re_share = target["부동산편중도"].mean()
    print(f"평균 부동산 편중도: {avg_re_share:.1%}")
    print(f"  (보고서 기준: ~80%) → {'✓' if avg_re_share >= 0.75 else '⚠ 괴리 확인 필요'}")

    # 소득 하위 20% 비중
    low_income_share = (target["소득_분위"] == "1분위(하위20%)").mean()
    print(f"고령층 소득 1분위 비중: {low_income_share:.1%}")
    print(f"  (보고서 기준: 44.5%) → {'✓' if abs(low_income_share - 0.445) < 0.05 else '⚠ 괴리 확인 필요'}")

    print(f"{'='*50}")


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 55)
    print("  팀원 A | 1~2주차 데이터 파이프라인 실행")
    print("  2026 국가데이터 활용대회")
    print("=" * 55)

    # 1. 가계금융복지조사 패널 구성 (2012~2024)
    TARGET_YEARS = list(range(2012, 2025))
    panel = build_panel(TARGET_YEARS)

    # 2. 민간 데이터 연계 (최신 연도 우선)
    #    실제 SDC에서는 모든 연도 연계 시도
    panel = merge_private_data(panel, year=2024)

    # 3. 분위 분류
    panel = assign_quintiles(panel)

    # 4. 이상치 처리 (핵심: 삭제하지 않고 의미 파악)
    panel = handle_outliers(panel)

    # 5. 검증
    validate_story(panel, year=2024)

    # 6. 저장
    output_path = OUTPUT_DIR / "elderly_panel_v1.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"\n✓ 저장 완료: {output_path}")
    print(f"  컬럼 수: {len(panel.columns)}")
    print(f"  행 수: {len(panel):,}")

    return panel


if __name__ == "__main__":
    panel = run_pipeline()
