"""
=======================================================
팀원 A | 3~4주차: 연관 규칙 분석 및 파생변수 도출
2026 국가데이터 활용대회 — 고령층 경제 이질성 진단
=======================================================

핵심 목표:
  - FP-Growth로 "고가주택 보유 + 카드연체 → 의료비 지출 낮음" 같은
    숨겨진 연관 패턴 발굴
  - D-Value(실질소비여력), 유동성함정지수, 자산노후도지수 생성
  - 팀원 B(군집분석), 팀원 C(SGIS 매핑)에 넘길 Master Table 구성
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

OUTPUT_DIR = Path("./output")

# vdr_pipeline.py 컬럼명 상수 (두 파일 간 호환)
RE_COL  = '자산_실물자산_부동산_거주주택금액'
INC_COL = '처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]'


# ════════════════════════════════════════════════════════════════
# STEP 1. 핵심 파생변수 3종 생성
# ════════════════════════════════════════════════════════════════

def create_d_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    D-Value (실질소비여력 지수)
    
    = 연간_카드지출 / 신고_총소득
    
    해석:
      D-Value > 1.0  → 소득 이상 지출 (자산 기반 소비 추정)
      D-Value < 0.3  → 극도로 낮은 소비 (현금 부족 의심)
      통계상 빈곤 지역인데 D-Value 높음 → '유형 빈곤' 지역 (SGIS 핫스팟)
    
    참조: 분석 레이어 ① — 실질 소비 여력
    """
    eps = 1e-6  # 0으로 나누기 방지
    df["D_Value"] = np.where(
        (df["총소득"] > 0) & (df["연간_카드지출"].notna()),
        df["연간_카드지출"] / (df["총소득"] + eps),
        np.nan
    )
    # 해석 레이블
    df["D_Value_등급"] = pd.cut(
        df["D_Value"],
        bins=[-np.inf, 0.3, 0.7, 1.0, 1.5, np.inf],
        labels=["극빈소비", "저소비", "정상소비", "과잉소비", "자산기반소비"]
    )
    return df


def create_liquidity_trap_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    유동성 함정 지수 (Liquidity Trap Index, LTI)
    
    고자산-저유동성 패턴을 0~1 점수로 정량화.
    
    구성 요소:
      ① 부동산 편중도 (높을수록 ↑)
      ② 금융자산 부족도 (낮을수록 ↑)
      ③ 연체 여부 (연체 있으면 ↑)
      ④ 소득 대비 생활비 여유 부족 (낮을수록 ↑)
    
    LTI = 0.35×부동산편중 + 0.25×금융빈곤 + 0.25×연체리스크 + 0.15×소득여유부족
    
    LTI > 0.7 → '유동성 함정' 핵심 대상 (주택연금 정책 우선 타겟)
    """
    # 구성 요소 정규화 (0~1)
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    # ① 부동산 편중도 (이미 0~1 비율)
    comp1 = df["부동산편중도"].fillna(0)

    # ② 금융자산 빈곤도 = 1 - (금융자산/순자산)
    if "금융자산" in df.columns:
        financial_share = np.where(
            df["순자산"] > 0,
            df["금융자산"] / df["순자산"],
            0
        )
        comp2 = minmax(pd.Series(1 - financial_share))
    else:
        comp2 = pd.Series(0, index=df.index)

    # ③ 연체 리스크 (연체 있으면 1, 없으면 0)
    if "연체여부" in df.columns:
        comp3 = df["연체여부"].fillna(0).astype(float)
    else:
        comp3 = pd.Series(0, index=df.index)

    # ④ 소득 여유 부족 = 1 - (총소득/생활비_추정)
    #    생활비 추정: 가구원수 × 월 100만원 기준
    if "가구원수" in df.columns:
        estimated_living = df["가구원수"] * 1_200_000 * 12
        income_margin = np.where(
            estimated_living > 0,
            1 - (df["총소득"] / estimated_living).clip(0, 1),
            0
        )
        comp4 = minmax(pd.Series(income_margin))
    else:
        comp4 = pd.Series(0, index=df.index)

    # 가중 합산
    df["LTI"] = (
        0.35 * comp1 +
        0.25 * comp2 +
        0.25 * comp3 +
        0.15 * comp4
    ).clip(0, 1)

    # 등급 분류
    df["LTI_등급"] = pd.cut(
        df["LTI"],
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["정상", "주의", "경고", "유동성함정"]
    )
    return df


def create_asset_age_index(df: pd.DataFrame, grid_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    자산 노후도 지수 (Asset Aging Index, AAI)
    
    = 노후건물비율 × 부동산자산가치 → 격자 단위 '동결자산 규모'
    
    팀원 C SGIS 분석과 연계:
      AAI 높은 격자 = '자산 재생' 핫스팟 (주택연금/다운사이징 시급)
    """
    if grid_stats is not None and "격자ID" in df.columns:
        df = df.merge(grid_stats[["격자ID", "노후건물비율"]], on="격자ID", how="left")
    else:
        # 격자 연계 없을 경우: 자산 × 부동산편중도로 대리 계산
        df["노후건물비율"] = df.get("노후건물비율", pd.Series(np.nan, index=df.index))

    df["AAI"] = (
        df["노후건물비율"].fillna(df["부동산편중도"]) *
        df["부동산자산"] / 1e8  # 억 단위 정규화
    )
    return df


# ════════════════════════════════════════════════════════════════
# STEP 2. Association Rule Analysis (FP-Growth)
# ════════════════════════════════════════════════════════════════

def discretize_for_apriori(df: pd.DataFrame) -> pd.DataFrame:
    """
    연속변수를 이진(True/False) 항목으로 변환 → 연관규칙 분석 입력 형태.

    항목 정의:
      고가주택   : 부동산자산 상위 40%
      소득빈곤   : 총소득 하위 30%
      카드연체   : 연체여부 = True
      저의료비   : 의료비지출 하위 25%
      후기고령   : 75세 이상
      부동산집중 : 부동산편중도 >= 0.80
    """
    # vdr_pipeline.py 컬럼명 호환: 두 네이밍 모두 지원
    re_col  = RE_COL  if RE_COL  in df.columns else '부동산자산'
    inc_col = INC_COL if INC_COL in df.columns else '총소득'
    med_col = '지출_소비지출_의료비' if '지출_소비지출_의료비' in df.columns else '의료비지출'
    age_col = '가구주_만연령' if '가구주_만연령' in df.columns else '가구주_연령'

    q60_re   = df[re_col].quantile(0.60)
    q30_in   = df[inc_col].quantile(0.30)
    q25_med  = df[med_col].quantile(0.25) if med_col in df.columns else 0

    items = pd.DataFrame({
        "고가주택"  : df[re_col] >= q60_re,
        "소득빈곤"  : df[inc_col] <= q30_in,
        "카드연체"  : df.get("연체여부",   pd.Series(False, index=df.index)).fillna(False).astype(bool),
        "저의료비"  : df[med_col] <= q25_med if med_col in df.columns
                      else pd.Series(False, index=df.index),
        "후기고령"  : df[age_col] >= 75,
        "부동산집중": df.get('부동산편중도', pd.Series(0, index=df.index)).fillna(0) >= 0.80,
        "유동성함정": df.get("LTI_등급",    pd.Series("정상", index=df.index)) == "유동성함정",
        "자산기반소비": df.get("D_Value_등급", pd.Series("정상소비", index=df.index)) == "자산기반소비",
    })
    return items


def run_association_rules(df: pd.DataFrame,
                          min_support: float = 0.05,
                          min_confidence: float = 0.4) -> pd.DataFrame:
    """
    순수 Pandas 기반 연관 규칙 분석 (VDR 환경 호환 — mlxtend 불필요).
    K-Anonymity(조건절 표본 5건 미만 시 자동 마스킹) 내장.

    분석 규칙 3개:
      Rule_1: {고가주택, B유형} → {저의료비}   (현금없어 병원 못 감)
      Rule_2: {고가주택, B유형, 후기고령} → {저의료비} (후기 노인 의료 포기)
      Rule_3: {고가주택, B유형} → {식비집중}   (식비만 겨우 유지)
    """
    items = discretize_for_apriori(df)
    total_n = len(items)

    def _calc(antecedents: list, consequent: str) -> tuple:
        """
        지지도·신뢰도·향상도·조건절_표본수 산출.
        조건절 또는 동시만족 표본이 5건 미만이면 NaN 반환 (K-Anonymity).
        """
        cond = items.copy()
        for item in antecedents:
            cond = cond[cond[item] == True]
        cond_n = len(cond)

        both_n = len(cond[cond[consequent] == True])

        # K-Anonymity: 5건 미만 → 마스킹
        if cond_n < 5 or both_n < 5:
            return np.nan, np.nan, np.nan, cond_n

        support    = both_n / total_n
        confidence = both_n / cond_n
        base_prob  = items[consequent].mean()
        lift       = confidence / base_prob if base_prob > 0 else 0.0

        return support, confidence, lift, cond_n

    rules_config = [
        {
            'id'  : 'Rule_1',
            'ant' : ['고가주택', 'B유형'],
            'con' : '저의료비',
            'desc': '{고가주택, B유형} → {저의료비}: 현금부족으로 병원 기피',
        },
        {
            'id'  : 'Rule_2',
            'ant' : ['고가주택', 'B유형', '후기고령'],
            'con' : '저의료비',
            'desc': '{고가주택, B유형, 후기고령} → {저의료비}: 후기 고령자 의료 포기',
        },
        {
            'id'  : 'Rule_3',
            'ant' : ['고가주택', 'B유형'],
            'con' : '식비집중',
            'desc': '{고가주택, B유형} → {식비집중}: 식비만 겨우 유지하는 경직 소비',
        },
    ]

    # discretize_for_apriori 결과에 B유형·식비집중 항목 추가
    items['B유형'] = df.get('고령층유형',
                            pd.Series('', index=df.index)) == 'B유형_자산소득불일치'
    items['식비집중'] = df.get(
        '지출_소비지출_식료품(외식비포함)',
        pd.Series(0, index=df.index)
    ) >= df.get(
        '지출_소비지출_식료품(외식비포함)',
        pd.Series(0, index=df.index)
    ).quantile(0.80)

    rows = []
    for r in rules_config:
        sup, conf, lift, cnt = _calc(r['ant'], r['con'])
        rows.append({
            '규칙ID'                  : r['id'],
            '연관규칙_구조'           : r['desc'],
            '조건절_표본수'           : cnt,
            '지지도(Support)'         : round(sup,  4) if not np.isnan(sup)  else '마스킹(5건미만)',
            '신뢰도(Confidence)'      : round(conf, 4) if not np.isnan(conf) else '마스킹(5건미만)',
            '향상도(Lift)'            : round(lift, 4) if not np.isnan(lift) else '마스킹(5건미만)',
        })

    result = pd.DataFrame(rows)

    # min_support·min_confidence 기준 필터 (마스킹 행 제외)
    numeric_mask = pd.to_numeric(result['지지도(Support)'], errors='coerce').notna()
    result_filtered = result[
        numeric_mask &
        (pd.to_numeric(result['지지도(Support)'],    errors='coerce') >= min_support) &
        (pd.to_numeric(result['신뢰도(Confidence)'], errors='coerce') >= min_confidence)
    ].copy()

    print(f"\n연관 규칙 분석 결과 ({len(result_filtered)}개 규칙 통과):")
    print(result_filtered.to_string(index=False))

    return result_filtered


def manual_association_check(items: pd.DataFrame) -> pd.DataFrame:
    """
    mlxtend 없을 때 수동 연관 검증 (SDC 환경 대비).
    주요 가설 4개를 직접 집계로 검증.
    """
    results = []

    # 가설 1: 고가주택 + 카드연체 → 저의료비
    h1_mask = items["고가주택"] & items["카드연체"]
    h1_support = h1_mask.mean()
    h1_conf = (h1_mask & items["저의료비"]).sum() / max(h1_mask.sum(), 1)
    base_med = items["저의료비"].mean()
    results.append({
        "antecedents": "고가주택 + 카드연체",
        "consequents": "저의료비",
        "support": h1_support,
        "confidence": h1_conf,
        "lift": h1_conf / max(base_med, 1e-6)
    })

    # 가설 2: 부동산집중 + 후기고령 → 소득빈곤
    h2_mask = items["부동산집중"] & items["후기고령"]
    h2_conf = (h2_mask & items["소득빈곤"]).sum() / max(h2_mask.sum(), 1)
    base_poor = items["소득빈곤"].mean()
    results.append({
        "antecedents": "부동산집중 + 후기고령",
        "consequents": "소득빈곤",
        "support": h2_mask.mean(),
        "confidence": h2_conf,
        "lift": h2_conf / max(base_poor, 1e-6)
    })

    # 가설 3: 소득빈곤 + 고가주택 → 자산기반소비
    h3_mask = items["소득빈곤"] & items["고가주택"]
    h3_conf = (h3_mask & items["자산기반소비"]).sum() / max(h3_mask.sum(), 1)
    base_abc = items["자산기반소비"].mean()
    results.append({
        "antecedents": "소득빈곤 + 고가주택",
        "consequents": "자산기반소비",
        "support": h3_mask.mean(),
        "confidence": h3_conf,
        "lift": h3_conf / max(base_abc, 1e-6)
    })

    # 가설 4: 유동성함정 → 저의료비 (현금부족 → 의료 포기)
    h4_mask = items["유동성함정"]
    h4_conf = (h4_mask & items["저의료비"]).sum() / max(h4_mask.sum(), 1)
    results.append({
        "antecedents": "유동성함정",
        "consequents": "저의료비",
        "support": h4_mask.mean(),
        "confidence": h4_conf,
        "lift": h4_conf / max(base_med, 1e-6)
    })

    result_df = pd.DataFrame(results)
    print("\n📌 수동 연관 검증 결과:")
    print(result_df.to_string(index=False, float_format="{:.3f}".format))
    return result_df


# ════════════════════════════════════════════════════════════════
# STEP 3. Master Table 구성 (팀 전체의 공통 데이터 기반)
# ════════════════════════════════════════════════════════════════

MASTER_TABLE_COLS = [
    # 식별자
    "가구ID", "조사연도",

    # 기본 속성
    "가구주_연령", "연령그룹", "가구원수",

    # 소득·자산 핵심
    "총소득", "총자산", "순자산", "부동산자산", "금융자산", "부채",
    "근로소득", "사업소득", "재산소득", "이전소득",

    # 분위 변수
    "순자산_분위", "소득_분위", "순자산_고령내분위",

    # 파생변수 3종 (팀원 A 핵심 산출물)
    "D_Value", "D_Value_등급",
    "LTI", "LTI_등급",
    "AAI",

    # 구조 변수
    "부동산편중도", "부동산편중_고위험", "부채비율",

    # 이상치/플래그
    "이상치_플래그", "처리방침", "페르소나_후보",

    # 민간 데이터 파생
    "연간_카드지출", "연체여부",

    # SGIS 연계 (팀원 C용)
    "격자ID", "노후건물비율",
]


def build_master_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    최종 Master Table 생성.
    팀원 B (모델링), 팀원 C (공간분석)에 인도할 공통 데이터셋.
    """
    # 존재하는 컬럼만 선택
    cols = [c for c in MASTER_TABLE_COLS if c in df.columns]
    master = df[cols].copy()

    # 최신 연도 기준 요약 통계
    latest = master[master["조사연도"] == master["조사연도"].max()]
    print(f"\n{'='*55}")
    print(f"  Master Table 요약 ({master['조사연도'].max()}년 기준)")
    print(f"{'='*55}")
    print(f"  전체 행 수 (전 연도): {len(master):,}")
    print(f"  최신연도 고령 가구: {len(latest):,}")

    if "연령그룹" in latest.columns:
        for grp, cnt in latest["연령그룹"].value_counts().items():
            print(f"    {grp}: {cnt:,}가구")

    if "LTI_등급" in latest.columns:
        print(f"\n  유동성 함정 분포:")
        for lv, cnt in latest["LTI_등급"].value_counts().items():
            print(f"    {lv}: {cnt:,}가구 ({cnt/len(latest):.1%})")

    if "D_Value_등급" in latest.columns:
        print(f"\n  D-Value 분포:")
        for lv, cnt in latest["D_Value_등급"].value_counts().items():
            print(f"    {lv}: {cnt:,}가구 ({cnt/len(latest):.1%})")

    return master


# ════════════════════════════════════════════════════════════════
# STEP 4. 팀원별 산출물 내보내기
# ════════════════════════════════════════════════════════════════

def export_for_team(master: pd.DataFrame):
    """팀원 B, C에게 넘길 특화 데이터셋 출력."""

    latest_year = master["조사연도"].max()
    latest = master[master["조사연도"] == latest_year].copy()

    # ── 팀원 B용: 모델 학습 데이터 ─────────────────────────────
    team_b_cols = [
        "가구ID", "연령그룹", "총소득", "순자산",
        "부동산편중도", "D_Value", "LTI",
        "연체여부", "이상치_플래그", "페르소나_후보",
        "순자산_분위", "소득_분위"
    ]
    team_b = latest[[c for c in team_b_cols if c in latest.columns]]
    team_b.to_csv(OUTPUT_DIR / "for_teamB_model_input.csv", index=False, encoding="utf-8-sig")
    print(f"\n✓ 팀원 B 산출물: {len(team_b):,}행 × {len(team_b.columns)}열")

    # ── 팀원 C용: SGIS 격자 매핑 데이터 ─────────────────────────
    team_c_cols = [
        "가구ID", "격자ID", "연령그룹",
        "LTI", "LTI_등급", "AAI",
        "부동산자산", "노후건물비율", "D_Value"
    ]
    team_c = latest[[c for c in team_c_cols if c in latest.columns]]
    
    # 격자 단위 집계 (팀원 C가 지도에 바로 올릴 수 있도록)
    if "격자ID" in team_c.columns:
        grid_agg = team_c.groupby("격자ID").agg(
            고령가구수=("가구ID", "count"),
            평균LTI=("LTI", "mean"),
            평균AAI=("AAI", "mean"),
            유동성함정가구수=("LTI_등급", lambda x: (x == "유동성함정").sum()),
            평균D_Value=("D_Value", "mean"),
        ).reset_index()
        grid_agg["유동성함정비율"] = grid_agg["유동성함정가구수"] / grid_agg["고령가구수"]
        grid_agg.to_csv(OUTPUT_DIR / "for_teamC_sgis_grid.csv", index=False, encoding="utf-8-sig")
        print(f"✓ 팀원 C 산출물: {len(grid_agg):,}개 격자")

    # ── Master Table 전체 저장 ─────────────────────────────────
    master.to_parquet(OUTPUT_DIR / "master_table_final.parquet", index=False)
    print(f"✓ Master Table: {OUTPUT_DIR / 'master_table_final.parquet'}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def run_week3_4(panel: pd.DataFrame):
    """3~4주차 파이프라인 실행."""
    print("\n" + "=" * 55)
    print("  팀원 A | 3~4주차: 파생변수 + 연관규칙 분석")
    print("=" * 55)

    # 파생변수 생성
    panel = create_d_value(panel)
    panel = create_liquidity_trap_index(panel)
    panel = create_asset_age_index(panel)

    print("✓ 파생변수 3종 생성 완료: D_Value, LTI, AAI")

    # 연관규칙 분석 (최신 연도 기준)
    latest_year = panel["조사연도"].max()
    latest = panel[panel["조사연도"] == latest_year].copy()
    print(f"\n연관규칙 분석 대상: {len(latest):,}가구 ({latest_year}년)")
    rules = run_association_rules(latest)

    # Master Table 구성
    master = build_master_table(panel)

    # 팀원별 산출물 내보내기
    export_for_team(master)

    return master, rules


if __name__ == "__main__":
    # week1 파이프라인 결과를 이어받아 실행
    panel_path = OUTPUT_DIR / "elderly_panel_v1.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        master, rules = run_week3_4(panel)
    else:
        print("⚠ 먼저 week1_data_pipeline.py를 실행해주세요.")
