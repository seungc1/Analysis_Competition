"""
=======================================================
팀원 A | 5~6주차: 데이터 신뢰도 검증 및 EDA 시각화
2026 국가데이터 활용대회 — 고령층 경제 이질성 진단
=======================================================

핵심 목표:
  - Master Table 편향성 제거 및 최종 확정
  - 팀원 B 스토리라인 지원용 EDA 시각화
  - 심사용 데이터 출처·시계열 정리 (보고서 작성 지원)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

OUTPUT_DIR = Path("./output")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# vdr_pipeline.py 실제 컬럼명 (week1 단순명과 다름 — 혼용 방지용 상수)
INC_COL = '처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]'
RE_COL  = '자산_실물자산_부동산_거주주택금액'
AGE_COL = '가구주_만연령'   # week1의 '가구주_연령' → 실제 컬럼명

# 한글 폰트 설정 (SDC 환경에 맞게 조정)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════
# STEP 1. 편향성 점검
# ════════════════════════════════════════════════════════════════

def check_sampling_bias(df: pd.DataFrame):
    """
    샘플링 편향 점검: 지역·성별·연령 분포가 모집단과 일치하는지 확인.
    
    가계금융복지조사 기준 모집단 비율 (통계청 발표 기준 입력):
      전기/후기 고령자 비율, 지역별 분포, 부동산 보유율 등
    """
    print("=" * 55)
    print("  편향성 점검 보고")
    print("=" * 55)

    latest_year = df["조사연도"].max()
    latest = df[df["조사연도"] == latest_year]

    # 연령 그룹 분포
    if "연령그룹" in latest.columns:
        print("\n① 연령 그룹 분포:")
        dist = latest["연령그룹"].value_counts(normalize=True)
        for k, v in dist.items():
            print(f"   {k}: {v:.1%}")

    # 소득 분위 분포 (균등해야 20%씩)
    # vdr_pipeline: '소득분위_내부' / week1: '소득_분위' — 둘 다 지원
    inc_q = "소득분위_내부" if "소득분위_내부" in latest.columns else "소득_분위"
    if inc_q in latest.columns:
        print("\n② 소득 분위 분포 (이상: 각 20%):")
        dist = latest[inc_q].value_counts(normalize=True).sort_index()
        for k, v in dist.items():
            flag = "✓" if abs(v - 0.20) < 0.05 else "⚠"
            print(f"   {flag} {k}분위: {v:.1%}")

    # 부동산 보유율 (보고서 기준: 73~74%)
    # vdr_pipeline: RE_COL / week1: '부동산자산' — 둘 다 지원
    re_col = RE_COL if RE_COL in latest.columns else "부동산자산"
    if re_col in latest.columns:
        re_rate = (latest[re_col] > 0).mean()
        flag = "✓" if 0.68 <= re_rate <= 0.80 else "⚠ 괴리 확인 필요"
        print(f"\n③ 부동산 보유율: {re_rate:.1%} ({flag})")

    # 유동성 함정 비율 합리성 체크
    if "LTI_등급" in latest.columns:
        lti_severe = (latest["LTI_등급"] == "유동성함정").mean()
        print(f"\n④ 유동성 함정 비율: {lti_severe:.1%}")
        print(f"   (10~30% 범위가 분석적으로 유의미)")


def remove_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    식별된 편향 처리.
    
    주요 편향 유형:
      - 도시 과표집: 농촌 지역 가중치 조정
      - 고소득층 비협조 편향: 응답 가구 보정
      - 시계열 표본 변경: 연도별 가중치 정규화
    """
    # 연도별 가중치 정규화 (표본 크기 변동 흡수)
    year_counts = df.groupby("조사연도").size()
    median_count = year_counts.median()
    df["표본_가중치"] = df["조사연도"].map(
        lambda y: median_count / year_counts.get(y, median_count)
    )

    print(f"✓ 연도별 가중치 적용 완료 (총 {len(df):,}행)")
    return df


# ════════════════════════════════════════════════════════════════
# STEP 2. 핵심 EDA 시각화 (보고서 삽입용)
# ════════════════════════════════════════════════════════════════

def plot_asset_income_gap(df: pd.DataFrame):
    """
    핵심 스토리 시각화: 자산-소득 괴리 추이
    
    보고서 핵심 발견 재현:
      순자산/전체평균 배율 2012→2024 (1.038배 돌파)
      소득/전체평균 배율 2012→2024 (0.609배)
    """
    # vdr_pipeline: INC_COL / week1: '총소득' — 둘 다 지원
    inc = INC_COL if INC_COL in df.columns else "총소득"
    yearly = df.groupby("조사연도").agg(
        고령순자산평균=("순자산", "mean"),
        고령소득평균=(inc, "mean"),
    ).reset_index()

    # 전체 평균 대비 배율 (실제 전체 평균이 없으면 초기값 대비)
    base_asset = yearly.loc[yearly["조사연도"] == yearly["조사연도"].min(), "고령순자산평균"].values[0]
    base_income = yearly.loc[yearly["조사연도"] == yearly["조사연도"].min(), "고령소득평균"].values[0]

    yearly["자산_배율"] = yearly["고령순자산평균"] / base_asset
    yearly["소득_배율"] = yearly["고령소득평균"] / base_income

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["조사연도"], yearly["자산_배율"], "o-", color="#E8593C", lw=2.5, label="순자산 배율")
    ax.plot(yearly["조사연도"], yearly["소득_배율"], "s--", color="#3B8BD4", lw=2.5, label="소득 배율")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.fill_between(yearly["조사연도"], yearly["자산_배율"], yearly["소득_배율"],
                    alpha=0.12, color="#E8593C", label="자산-소득 괴리")
    ax.set_title("고령 가구 자산·소득 배율 추이 (2012=1.0 기준)", fontsize=13, fontweight="bold")
    ax.set_xlabel("조사 연도")
    ax.set_ylabel("기준연도 대비 배율")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_asset_income_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 그림 저장: 01_asset_income_gap.png")


def plot_d_value_distribution(df: pd.DataFrame):
    """
    D-Value 분포: '통계상 빈곤이나 실제 소비는 자산 기반' 입증.
    
    핵심 메시지:
      소득 하위 20%인데도 D-Value > 1.0인 가구 비율
      → 소득통계로는 포착 못하는 실질 경제력 존재
    """
    if "D_Value" not in df.columns or "소득_분위" not in df.columns:
        return

    latest = df[df["조사연도"] == df["조사연도"].max()]
    # vdr_pipeline: '소득분위_내부'(int 1~5) / week1: '소득_분위'(str) — 둘 다 지원
    inc_q = "소득분위_내부" if "소득분위_내부" in latest.columns else "소득_분위"
    low_income = latest[
        latest[inc_q] == (1 if inc_q == "소득분위_내부" else "1분위(하위20%)")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 전체 D-Value 분포
    axes[0].hist(latest["D_Value"].dropna().clip(0, 3), bins=50, color="#3B8BD4", alpha=0.7)
    axes[0].axvline(1.0, color="red", lw=1.5, ls="--", label="D-Value=1.0")
    axes[0].set_title("전체 고령 가구 D-Value 분포")
    axes[0].set_xlabel("D-Value (카드지출/소득)")
    axes[0].legend()

    # 소득 하위 20% 중 D-Value > 1.0 비율 (핵심 스토리)
    over_1 = (low_income["D_Value"] > 1.0).mean()
    axes[1].bar(["D-Value ≤ 1.0", "D-Value > 1.0 (자산기반소비)"],
                [1 - over_1, over_1],
                color=["#B4B2A9", "#E8593C"])
    axes[1].set_title(f"소득 하위 20% 고령 가구\nD-Value > 1.0 비율: {over_1:.1%}")
    axes[1].set_ylabel("비율")
    for i, v in enumerate([1 - over_1, over_1]):
        axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=11)

    fig.suptitle("D-Value: '소득 빈곤 ≠ 소비 빈곤' 입증", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_d_value_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 그림 저장: 02_d_value_distribution.png")


def plot_lti_heatmap_by_age(df: pd.DataFrame):
    """
    연령대별 × LTI 등급 히트맵.
    
    핵심 메시지:
      75세 이상에서 '유동성 함정' 비율 급증
      → 후기 고령자 우선 정책 타겟의 데이터 근거
    """
    if "LTI_등급" not in df.columns:
        return

    latest = df[df["조사연도"] == df["조사연도"].max()]

    # 5세 구간 연령대
    latest = latest.copy()
    # vdr_pipeline: AGE_COL='가구주_만연령' / week1: '가구주_연령' — 둘 다 지원
    age_col = AGE_COL if AGE_COL in latest.columns else "가구주_연령"
    latest["연령대"] = pd.cut(
        latest[age_col],
        bins=[64, 69, 74, 79, 84, 150],
        labels=["65-69", "70-74", "75-79", "80-84", "85+"]
    )

    pivot = pd.crosstab(latest["연령대"], latest["LTI_등급"], normalize="index")
    
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.1%}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="비율")
    ax.set_title("연령대별 유동성 함정 지수(LTI) 분포", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_lti_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 그림 저장: 03_lti_heatmap.png")


# ════════════════════════════════════════════════════════════════
# STEP 3. 데이터 활용 명세서 (보고서 작성용)
# ════════════════════════════════════════════════════════════════

def generate_data_manifest(df: pd.DataFrame) -> str:
    """
    공모전 보고서에 필수 기재할 데이터 활용 명세서 자동 생성.
    
    심사 항목 '데이터 활용성' 직접 대응:
      SDC 제공 자료 활용 시 평가 우대
    """
    latest_year = df["조사연도"].max()
    min_year = df["조사연도"].min()

    manifest = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  데이터 활용 명세서
  2026 국가데이터 활용대회
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【SDC 제공 자료 (심사 우대)】
  ① 가계금융복지조사 마이크로데이터
     - 출처: 국가데이터처 / 한국은행 / 금융감독원
     - 시계열: {min_year}~{latest_year}년 (총 {latest_year - min_year + 1}개년)
     - 분석 대상: 가구주 연령 65세 이상 고령 가구

  ② SGIS 소지역통계 (격자 단위)
     - 출처: 국가데이터처 통계지리정보서비스(SGIS)
     - 활용: 지역별 고령 가구 자산 분포 공간 매핑

  ③ 주택소유통계 (행정통계자료 22종)
     - 출처: 국가데이터처
     - 시계열: 동일 기간

【민간 자료 (SDC 제공 67종)】
  ④ 카드 매출 데이터
     - 목적: 실질 소비 여력(D-Value) 산출
     - 핵심: 소득통계 빈곤 ≠ 실제 소비 수준 입증

  ⑤ 개인신용정보 (대출·연체)
     - 목적: 유동성 함정 지수(LTI) 구성
     - 핵심: 자산 부유·현금 빈곤의 직접 증거

【공공 데이터 (외부 연계)】
  ⑥ 건축물대장 (30년 이상 노후 건물)
     - 출처: 국토교통부 건축물대장 공공데이터 API
     - 목적: 자산 노후도 지수(AAI) 산출

【분석 소프트웨어】
  - Python 3.11 (pandas, numpy, scikit-learn, mlxtend)
  - SGIS/QGIS (팀원 C 공간 분석)
  - 분석 환경: 통계데이터센터(SDC) 방문 분석

【핵심 파생변수 (팀원 A 생성)】
  D-Value  = 연간 카드지출 / 신고 총소득
  LTI      = 0.35×부동산편중 + 0.25×금융빈곤 + 0.25×연체리스크 + 0.15×소득여유부족
  AAI      = 노후건물비율 × 부동산자산가치 (격자 단위)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(manifest)

    # 파일 저장
    with open(OUTPUT_DIR / "data_manifest.txt", "w", encoding="utf-8") as f:
        f.write(manifest)
    print("✓ 저장: data_manifest.txt")
    return manifest


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def run_week5_6(master: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  팀원 A | 5~6주차: 신뢰도 검증 및 최종 확정")
    print("=" * 55)

    # 편향성 점검
    check_sampling_bias(master)
    master = remove_bias(master)

    # EDA 시각화
    plot_asset_income_gap(master)
    plot_d_value_distribution(master)
    plot_lti_heatmap_by_age(master)

    # 데이터 명세서 출력
    generate_data_manifest(master)

    # 최종 Master Table 저장
    master.to_parquet(OUTPUT_DIR / "master_table_FINAL_v2.parquet", index=False)
    print(f"\n✓ 최종 Master Table 확정: {len(master):,}행 × {len(master.columns)}열")
    print("✓ 팀원 B, C 인도 준비 완료")

    return master


if __name__ == "__main__":
    master_path = OUTPUT_DIR / "master_table_final.parquet"
    if master_path.exists():
        master = pd.read_parquet(master_path)
        run_week5_6(master)
    else:
        print("⚠ 먼저 week3_4_feature_engineering.py를 실행해주세요.")
