"""
══════════════════════════════════════════════════════════════════
팀원 A | Master Table 설계 및 구성 코드
2026 국가데이터 활용대회 — 고령층 경제 이질성 진단
══════════════════════════════════════════════════════════════════

[데이터 소스 및 항목명 출처]
① 인구통계등록부     : SGIS 소지역통계 (13개 항목)
② 가구통계등록부     : SGIS 소지역통계 (14개 항목)
③ 주택통계등록부     : SGIS 소지역통계 (10개 항목)
④ 가계금융복지조사   : 2023~2024 가구마스터 CSV (30개 항목)
⑤ 카드소비 데이터    : 내국인 국내카드 소비(서울) (29개 항목)
⑥ NICE 대출·연체    : 신용통계정보_대출 및 연체 (50개 항목)
⑦ NICE 소득통계     : 신용통계정보_소득 (20개 항목)

[JOIN 전략]
- 가구↔인구: HOHL_SPIDN (가구주통계목적고유번호)
- 가구↔주택: LVQT_SN (거처일련번호) + ADMDST_CLSF_CD
- 가계금융복지조사↔NICE: 시군구코드 + 연령구간 집계 단위
- 카드소비↔가구: 행정구역분류시군구코드 집계 단위
══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════
# LAYER 0. 항목명 검증 딕셔너리 (오타 방지용)
# XLS 파일 직접 확인한 실제 항목명만 수록
# ════════════════════════════════════════════════════════════════

# ─ 인구통계등록부 실제 항목명 (XLS 원본 확인) ──────────────────
POPULATION_COLS = {
    "CRTR_YR"       : "기준연도",
    "ADMDST_CLSF_CD": "행정구역분류코드",
    "HOHL_SPIDN"    : "가구주통계목적고유번호",
    "SPIDN"         : "통계목적고유번호",
    "LFNR_SE_CD"    : "내외국인구분코드",
    "HOHL_REL_CD"   : "가구주관계코드",
    "SD_CD"         : "성별코드",
    "AAGE"          : "만나이",
    "NTNLTY_CD"     : "국적코드",
    "FMNM_ACHM_CD"  : "성씨본관코드",
    "HS_SHAPE_CD"   : "가구형태코드",
    "BRTH_YR"       : "출생연도",
    "ENTCNY_YR"     : "입국연도",
}

# ─ 가구통계등록부 실제 항목명 (XLS 원본 확인) ──────────────────
HOUSEHOLD_COLS = {
    "CRTR_YR"       : "기준연도",
    "ADMDST_CLSF_CD": "행정구역분류코드",
    "HOHL_SPIDN"    : "가구주통계목적고유번호",
    "SD_CD"         : "성별코드",
    "AAGE"          : "만나이",
    "HS_SHAPE_CD"   : "가구형태코드",
    "HS_SE_CD"      : "가구구분코드",
    "MBHS_CNT"      : "가구원수",
    "LVQT_KIND_CD"  : "거처종류코드",
    "DTHS_TYPE_CD"  : "단독주택유형코드",
    "LVQT_SN"       : "거처일련번호",
    "HH_FMTN_CD"    : "세대구성코드",
    "HH_HS_TYPE_CD" : "세대가구유형코드",
    "MCLTR_HS_YN"   : "다문화가구여부",
}

# ─ 주택통계등록부 실제 항목명 (XLS 원본 확인) ──────────────────
HOUSING_COLS = {
    "CRTR_YR"        : "기준연도",
    "ADMDST_CLSF_CD" : "행정구역분류코드",
    "LVQT_KIND_CD"   : "거처종류코드",
    "DTHS_TYPE_CD"   : "단독주택유형코드",
    "RSDT_AREA"      : "주거용면적",
    "SIAR"           : "대지면적",
    "ARCH_APRV_YR"   : "건축승인연도",
    "BLDG_DLPD_PD_CD": "건물노후기간코드",
    "NHAB_HOUS_YN"   : "미거주주택여부",
    "LVQT_SN"        : "거처일련번호",
}

# ─ 가계금융복지조사 가구마스터 컬럼명 (R코드 colnames 기준) ─────
HFWS_COLS = [
    "조사연도", "MD제공용_가구고유번호", "가중값", "수도권여부", "가구원수",
    "노인가구여부", "가구주_만연령", "입주형태코드", "주택종류통합코드",
    "자산", "자산_금융자산", "자산_실물자산",
    "자산_실물자산_부동산_거주주택금액", "부채",
    "부채_금융부채_담보대출금액", "순자산",
    "경상소득(보완)", "경상소득_공적이전소득(보완)",
    "처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]",
    "지출_소비지출비", "지출_소비지출_식료품(외식비포함)",
    "지출_소비지출_의료비", "지출_비소비지출(보완)",
    "지출_비소비지출_세금(보완)",
    "지출_비소비지출_공적연금사회보험료(보완)",
    "지출_비소비지출_연간지급이자(조사2)",
    "가구주_은퇴여부", "가구주_미은퇴_최소생활비",
    "가구주_미은퇴_적정생활비", "가구주_은퇴_적정생활비충당여부",
]

# ─ NICE 대출·연체 항목명 (설명자료 기준) ──────────────────────
NICE_LOAN_KEY_COLS = [
    "기준년월", "구분명", "광역시도코드", "시군구코드(개정후)",
    "행정구역분류코드(개정후)", "광역시도명", "시군구명(개정후)",
    "행정구역분류명(개정후)", "성별", "연령구간대", "직업구분",
    "금융기관", "분위코드(총대출잔액기준)",
    "총대출_보유계좌수", "유효대출계좌_대상자수", "대출잔액_보유대상자수",
    "대출잔액_평균금액", "대출잔액_주택담보_평균금액",
    "대출잔액_신용_평균금액", "대출잔액_중위금액",
    "총대출잔액", "총대출잔액_주택담보", "총대출잔액_신용",
    "대출평균이자율",
    "월 연체보유자수 합계", "월평균연체건수", "평균연체일",
    "평균연체금액", "중위연체금액", "연체금액",
]
# 주의: "월 연체보유자수 합계" — 공백 포함된 항목명 그대로 사용

# ─ NICE 소득 항목명 (설명자료 기준) ───────────────────────────
NICE_INCOME_KEY_COLS = [
    "기준년월", "구분명", "광역시도코드", "시군구코드(개정후)",
    "행정구역분류코드(개정후)", "광역시도명", "시군구명(개정후)",
    "행정구역분류명(개정후)", "성별", "연령구간대",
    "직업 구분",  # 주의: 공백 포함 — 원본 항목명 그대로
    "분위코드(연소득기준)",
    "최저 연소득 금액", "최대 연소득 금액",
    "거주자수", "평균 연소득 금액", "중위 연소득 금액", "총연소득합계",
]
# 주의: "직업 구분", "평균 연소득 금액", "중위 연소득 금액" — 공백 포함

# ─ 카드소비 항목명 (설명자료 기준) ────────────────────────────
CARD_KEY_COLS = [
    "기준연월", "공휴일구분코드",
    "가맹점행정구역분류시도코드", "가맹점행정구역분류시군구코드",
    "가맹점행정구역분류코드", "가맹점법정동코드",
    "가맹점행정구역분류시도명", "가맹점행정구역분류시군구명",
    "가맹점행정구역분류명", "가맹점법정동명",
    "통합카드업종3레벨코드", "통합카드업종3레벨명",
    "조직구분코드", "통합카드성별코드", "통합카드5세단위연령코드",
    "통합카드직업코드",
    "고객행정구역분류시도코드", "고객행정구역분류시군구코드",
    "고객행정구역분류코드", "고객법정동코드",
    "고객행정구역분류시도명", "고객행정구역분류시군구명",
    "고객행정구역분류명", "고객법정동명",
    "통합카드혼인구분코드", "통합카드가구형태코드",
    "카드사용건수", "카드사용금액", "1인당평균사용금액",
]
# 주의: 카드사용건수/금액 순서 — 원본 27/28번 항목 순서 준수


# ════════════════════════════════════════════════════════════════
# LAYER 1. 원천 데이터 로드 함수
# ════════════════════════════════════════════════════════════════

def load_population_registry(crtr_yr: str) -> pd.DataFrame:
    """인구통계등록부 — 항목명: HOHL_SPIDN, SPIDN, AAGE, HOHL_REL_CD 등"""
    path = DATA_DIR / f"population_{crtr_yr}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={
        "CRTR_YR": str, "ADMDST_CLSF_CD": str,
        "HOHL_SPIDN": str, "SPIDN": str,
        "LFNR_SE_CD": str, "HOHL_REL_CD": str,
        "SD_CD": str, "NTNLTY_CD": str,
        "HS_SHAPE_CD": str, "BRTH_YR": str, "ENTCNY_YR": str,
    })
    df["AAGE"] = pd.to_numeric(df["AAGE"], errors="coerce").astype("Int64")
    return df


def load_household_registry(crtr_yr: str) -> pd.DataFrame:
    """가구통계등록부 — 항목명: MBHS_CNT, LVQT_SN, HS_SE_CD 등"""
    path = DATA_DIR / f"household_{crtr_yr}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={
        "CRTR_YR": str, "ADMDST_CLSF_CD": str, "HOHL_SPIDN": str,
        "SD_CD": str, "HS_SHAPE_CD": str, "HS_SE_CD": str,
        "LVQT_KIND_CD": str, "DTHS_TYPE_CD": str, "LVQT_SN": str,
        "HH_FMTN_CD": str, "HH_HS_TYPE_CD": str, "MCLTR_HS_YN": str,
    })
    df["AAGE"]     = pd.to_numeric(df["AAGE"], errors="coerce").astype("Int64")
    df["MBHS_CNT"] = pd.to_numeric(df["MBHS_CNT"], errors="coerce").astype("Int64")
    return df


def load_housing_registry(crtr_yr: str) -> pd.DataFrame:
    """주택통계등록부 — 항목명: ARCH_APRV_YR, BLDG_DLPD_PD_CD, RSDT_AREA 등"""
    path = DATA_DIR / f"housing_{crtr_yr}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={
        "CRTR_YR": str, "ADMDST_CLSF_CD": str,
        "LVQT_KIND_CD": str, "DTHS_TYPE_CD": str,
        "BLDG_DLPD_PD_CD": str, "NHAB_HOUS_YN": str, "LVQT_SN": str,
    })
    df["RSDT_AREA"]    = pd.to_numeric(df["RSDT_AREA"], errors="coerce")
    df["SIAR"]         = pd.to_numeric(df["SIAR"], errors="coerce")
    df["ARCH_APRV_YR"] = pd.to_numeric(df["ARCH_APRV_YR"], errors="coerce").astype("Int64")
    return df


def load_hfws(year: int) -> pd.DataFrame:
    """
    가계금융복지조사 가구마스터.
    
    컬럼명: R코드 colnames 기준 30개 — 가중값, 수도권여부,
            자산_실물자산_부동산_거주주택금액,
            처분가능소득(보완)[경상소득(보완)-비소비지출(보완)] 등
    """
    path = DATA_DIR / f"{year}_가구마스터.csv"
    df = pd.read_csv(
        path, encoding="cp949", header=0, sep=",",
        na_values=["*","**","***","****","*****","******","*******",
                   "********","*********","**********","."]
    )
    # 컬럼 수 검증 후 이름 부여
    if len(df.columns) == 30:
        df.columns = HFWS_COLS
    else:
        print(f"  ⚠ 컬럼 수 불일치: {len(df.columns)}열 (기대: 30열) — 원본 컬럼명 유지")
    return df


def load_nice_loan(year: int) -> pd.DataFrame:
    """
    NICE 대출·연체 — 연령구간대 65(65~70세미만), 70(70세이상) 필터.
    
    주의 항목명: "월 연체보유자수 합계" (공백 포함)
    """
    frames = []
    for m in range(1, 13):
        p = DATA_DIR / f"nice_loan_{year}{m:02d}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, encoding="utf-8-sig",
                dtype={c: str for c in ["기준년월","구분명","광역시도코드",
                    "시군구코드(개정후)","행정구역분류코드(개정후)",
                    "성별","연령구간대","직업구분","금융기관",
                    "분위코드(총대출잔액기준)"]}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["연령구간대"].isin(["65", "70"])]


def load_nice_income(year: int) -> pd.DataFrame:
    """
    NICE 소득 — 연령구간대 65, 70 필터.
    
    주의 항목명: "직업 구분", "평균 연소득 금액", "중위 연소득 금액" (공백 포함)
    """
    frames = []
    for m in range(1, 13):
        p = DATA_DIR / f"nice_income_{year}{m:02d}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, encoding="utf-8-sig",
                dtype={c: str for c in ["기준년월","구분명","광역시도코드",
                    "시군구코드(개정후)","행정구역분류코드(개정후)",
                    "성별","연령구간대","직업 구분","분위코드(연소득기준)"]}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["연령구간대"].isin(["65", "70"])]


def load_card(year: int) -> pd.DataFrame:
    """
    카드소비 — 통합카드5세단위연령코드 14(65~69), 15(70+) 필터.
    가구형태: 통합카드가구형태코드 5(노인가구).
    """
    frames = []
    for m in range(1, 13):
        p = DATA_DIR / f"card_{year}{m:02d}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, encoding="utf-8-sig",
                dtype={c: str for c in ["기준연월","공휴일구분코드",
                    "가맹점행정구역분류시도코드","가맹점행정구역분류시군구코드",
                    "통합카드업종3레벨코드","조직구분코드","통합카드성별코드",
                    "통합카드5세단위연령코드","통합카드직업코드",
                    "고객행정구역분류시군구코드","통합카드혼인구분코드",
                    "통합카드가구형태코드"]}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[
        df["통합카드5세단위연령코드"].isin(["14", "15"]) &
        (df["조직구분코드"] == "1") &
        (df["통합카드가구형태코드"] == "5")
    ]


# ════════════════════════════════════════════════════════════════
# LAYER 2. SGIS 3개 등록부 결합
# ════════════════════════════════════════════════════════════════

def build_sgis_base(crtr_yr: str) -> pd.DataFrame:
    """
    인구 + 가구 + 주택 등록부 결합.

    JOIN KEY:
      인구↔가구: HOHL_SPIDN (가구주통계목적고유번호)
      가구↔주택: LVQT_SN + ADMDST_CLSF_CD
    """
    pop = load_population_registry(crtr_yr)
    hhd = load_household_registry(crtr_yr)
    hsg = load_housing_registry(crtr_yr)

    # 가구주(HOHL_REL_CD='1')만 추출하여 JOIN
    pop_head = pop[pop["HOHL_REL_CD"] == "1"][[
        "HOHL_SPIDN", "SPIDN", "LFNR_SE_CD", "BRTH_YR", "NTNLTY_CD"
    ]]

    merged = hhd.merge(pop_head, on="HOHL_SPIDN", how="left")
    merged = merged.merge(
        hsg[["LVQT_SN", "ADMDST_CLSF_CD",
             "RSDT_AREA", "SIAR", "ARCH_APRV_YR",
             "BLDG_DLPD_PD_CD", "NHAB_HOUS_YN"]],
        on=["LVQT_SN", "ADMDST_CLSF_CD"], how="left"
    )

    # 건물 노후 파생 — ARCH_APRV_YR (건축승인연도) 사용
    merged["건물연령"] = 2024 - merged["ARCH_APRV_YR"].fillna(2024)
    merged["노후건물여부"] = merged["건물연령"] >= 30

    return merged


# ════════════════════════════════════════════════════════════════
# LAYER 3. 고령층 필터링 및 연령 분류
# ════════════════════════════════════════════════════════════════

def filter_elderly_sgis(df: pd.DataFrame) -> pd.DataFrame:
    """SGIS 등록부 — AAGE(만나이) 기준 65세 이상 필터"""
    df = df[df["AAGE"] >= 65].copy()
    df["연령그룹"] = np.where(df["AAGE"] < 75, "전기고령자(65-74)", "후기고령자(75+)")
    return df


def filter_elderly_hfws(df: pd.DataFrame) -> pd.DataFrame:
    """가계금융복지조사 — 가구주_만연령 기준 65세 이상 필터"""
    df = df[df["가구주_만연령"] >= 65].copy()
    df["연령그룹"] = np.where(df["가구주_만연령"] < 75, "전기고령자(65-74)", "후기고령자(75+)")
    return df


# ════════════════════════════════════════════════════════════════
# LAYER 4. 파생변수 생성
# ════════════════════════════════════════════════════════════════

# 연도별 정부 고시 기준 중위소득 100% (월, 원) — 보건복지부 고시
MEDIAN_INCOME_BY_YEAR = {
    2023: {1: 2_077_892, 2: 3_456_155, 3: 4_434_816,
           4: 5_400_964, 5: 6_330_688, 6: 7_227_981, 7: 8_107_515},
    2024: {1: 2_228_445, 2: 3_682_609, 3: 4_714_657,
           4: 5_729_913, 5: 6_695_735, 6: 7_618_369, 7: 8_514_994},
}
# 7인 초과 시 1인 추가당 가산액
MEDIAN_INCOME_ADD = {2023: 879_534, 2024: 896_625}


def get_dynamic_poverty_line(year: int, num_members: int) -> float:
    """연도·가구원수별 정부 고시 빈곤선(중위소득 50%, 연간 만원)."""
    year = int(year)
    num  = int(num_members) if not pd.isna(num_members) else 1
    if year not in MEDIAN_INCOME_BY_YEAR:
        year = max(MEDIAN_INCOME_BY_YEAR.keys())
    tbl = MEDIAN_INCOME_BY_YEAR[year]
    add = MEDIAN_INCOME_ADD[year]
    monthly = tbl[num] if num <= 7 else tbl[7] + (num - 7) * add
    return round((monthly * 0.5 * 12) / 10_000, 1)

INC_COL = "처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]"
RE_COL  = "자산_실물자산_부동산_거주주택금액"


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    파생변수 생성.
    
    [사용 항목명 — 가계금융복지조사 실제 컬럼명 준수]
      자산                                              ✓
      자산_금융자산                                     ✓
      자산_실물자산_부동산_거주주택금액                 ✓
      부채                                              ✓
      순자산                                            ✓
      처분가능소득(보완)[경상소득(보완)-비소비지출(보완)] ✓
      지출_소비지출_식료품(외식비포함)                  ✓
      지출_소비지출_의료비                              ✓
      가구원수                                          ✓
    """
    eps = 1e-6

    # 부동산 편중도
    df["부동산편중도"] = np.where(
        df["자산"] > 0, df[RE_COL] / df["자산"], np.nan
    )
    df["부동산편중_고위험"] = df["부동산편중도"] >= 0.80

    # D-Value (소비/소득 비율 — 카드 연계 전 소비지출 프록시)
    소비프록시 = (
        df["지출_소비지출_식료품(외식비포함)"].fillna(0) +
        df["지출_소비지출_의료비"].fillna(0)
    )
    df["D_Value_proxy"] = np.where(
        df[INC_COL] > 0, 소비프록시 / (df[INC_COL] + eps), np.nan
    )

    # LTI (유동성 함정 지수)
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + eps)

    c1 = df["부동산편중도"].fillna(0)
    c2 = _mm(1 - np.where(df["자산"] > 0, df["자산_금융자산"] / df["자산"], 0))
    c3 = _mm(np.where(df[INC_COL] > 0, df["부채"] / (df[INC_COL] + eps), 0)).clip(0, 1)
    생활비 = df["가구원수"] * 1_200 * 12
    c4 = _mm(1 - (df[INC_COL] / (생활비 + eps)).clip(0, 1))

    df["LTI"] = (0.35*c1 + 0.25*c2 + 0.25*c3 + 0.15*c4).clip(0, 1)
    df["LTI_등급"] = pd.cut(df["LTI"],
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["정상", "주의", "경고", "유동성함정"])

    # AAI (자산 노후도 지수 — 가계조사 단독 프록시)
    df["AAI_proxy"] = df["부동산편중도"].fillna(0) * df[RE_COL].fillna(0) / 1e4

    return df


def classify_persona(df: pd.DataFrame) -> pd.DataFrame:
    """
    3중 교차 검증 고령층 유형 분류.
    
    기준:
      절대빈곤: 2026 중위소득 50% (경상소득 기준 — 부동산 자산 미포함)
      상대소득: 고령층 내 소득분위 하위 40%(1~2분위)
      상대자산: 고령층 내 거주주택금액 상위 40%(4~5분위)
    
    주의: 정부 수급 선정 시 '소득인정액(재산 소득환산 포함)'과 다름
          보고서에 반드시 명시 필요
    """
    df["정부기준빈곤선"] = df.apply(
        lambda r: get_dynamic_poverty_line(
            r.get("조사연도", max(MEDIAN_INCOME_BY_YEAR.keys())),
            r["가구원수"]
        ),
        axis=1
    )
    df["자산분위_내부"] = pd.qcut(
        df[RE_COL].rank(method="first"), 5, labels=[1,2,3,4,5]
    )
    df["소득분위_내부"] = pd.qcut(
        df[INC_COL].rank(method="first"), 5, labels=[1,2,3,4,5]
    )

    def _cls(r):
        빈곤 = r[INC_COL] <= r["정부기준빈곤선"]
        저소득 = r["소득분위_내부"] <= 2
        고자산 = r["자산분위_내부"] >= 4
        if 빈곤 and 저소득 and 고자산:
            return "B유형_자산소득불일치"
        elif 빈곤 and r["자산분위_내부"] <= 2:
            return "A유형_구조적취약층"
        elif not 저소득 and 고자산:
            return "D유형_여유자산가층"
        return "C유형_기타일반가구"

    df["고령층유형"] = df.apply(_cls, axis=1)
    return df


# ════════════════════════════════════════════════════════════════
# LAYER 5. 민간 데이터 집계 (행정구역 단위 연계)
# ════════════════════════════════════════════════════════════════

def aggregate_nice_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    NICE 대출·연체 → 시군구 + 연령구간 집계.
    직업구분='0'(전체)만 사용 (중복 합산 방지).
    
    사용 항목:
      시군구코드(개정후), 연령구간대, 직업구분 ✓
      "월 연체보유자수 합계" (공백 포함 항목명 준수) ✓
    """
    if df.empty:
        return pd.DataFrame()
    df = df[df["직업구분"] == "0"]
    return df.groupby(["기준년월","시군구코드(개정후)","연령구간대"]).agg(
        대출잔액_평균금액            = ("대출잔액_평균금액", "mean"),
        대출잔액_주택담보_평균금액   = ("대출잔액_주택담보_평균금액", "mean"),
        총대출잔액                   = ("총대출잔액", "sum"),
        총대출잔액_주택담보          = ("총대출잔액_주택담보", "sum"),
        대출평균이자율               = ("대출평균이자율", "mean"),
        월_연체보유자수_합계         = ("월 연체보유자수 합계", "sum"),
        평균연체금액                 = ("평균연체금액", "mean"),
        연체금액                     = ("연체금액", "sum"),
    ).assign(
        연체비율=lambda x: np.where(x["총대출잔액"]>0,
                                    x["연체금액"]/x["총대출잔액"], 0)
    ).reset_index()


def aggregate_nice_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    NICE 소득 → 시군구 + 연령구간 집계.
    
    사용 항목:
      시군구코드(개정후), 연령구간대, "직업 구분"(공백 포함) ✓
      "평균 연소득 금액", "중위 연소득 금액" (공백 포함) ✓
    """
    if df.empty:
        return pd.DataFrame()
    df = df[df["직업 구분"] == "0"]
    return df.groupby(["기준년월","시군구코드(개정후)","연령구간대"]).agg(
        거주자수          = ("거주자수", "sum"),
        평균_연소득_금액  = ("평균 연소득 금액", "mean"),
        중위_연소득_금액  = ("중위 연소득 금액", "mean"),
        총연소득합계      = ("총연소득합계", "sum"),
    ).reset_index()


def aggregate_card(df: pd.DataFrame) -> pd.DataFrame:
    """
    카드소비 → 시군구 + 업종 단위 집계.
    
    의료업종 필터: 통합카드업종3레벨코드 L로 시작 (NICE 업종 대분류 L=의료/건강)
    
    사용 항목:
      가맹점행정구역분류시군구코드, 통합카드업종3레벨코드,
      통합카드5세단위연령코드(14=65~69, 15=70+) ✓
      카드사용건수, 카드사용금액, 1인당평균사용금액 ✓
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby([
        "기준연월", "가맹점행정구역분류시군구코드",
        "통합카드업종3레벨코드", "통합카드5세단위연령코드"
    ]).agg(
        카드사용건수      = ("카드사용건수", "sum"),
        카드사용금액      = ("카드사용금액", "sum"),
        평균1인당사용금액 = ("1인당평균사용금액", "mean"),
    ).reset_index()

    # 의료 업종 서브셋
    medical = agg[agg["통합카드업종3레벨코드"].str.startswith("L", na=False)]
    return agg, medical


# ════════════════════════════════════════════════════════════════
# LAYER 6. Master Table 스키마 정의
# ════════════════════════════════════════════════════════════════

MASTER_TABLE_SCHEMA = {
    # ── 식별자 ──────────────────────────────────────────────────
    "MD제공용_가구고유번호"  : ("str",   "가계금융복지조사 가구 고유 ID"),
    "HOHL_SPIDN"            : ("str",   "SGIS 가구주통계목적고유번호"),
    "조사연도"               : ("int",   "가계금융복지조사 기준연도"),
    "CRTR_YR"               : ("str",   "SGIS 기준연도"),

    # ── 공간 ─────────────────────────────────────────────────────
    "ADMDST_CLSF_CD"        : ("str",   "행정구역분류코드 (SGIS 공통 JOIN 키)"),
    "수도권여부"             : ("str",   "G1=수도권 / G2=비수도권"),

    # ── 가구주 인구 속성 ─────────────────────────────────────────
    "AAGE"                  : ("Int64", "만나이 (SGIS 인구/가구통계등록부)"),
    "가구주_만연령"          : ("float", "만연령 (가계금융복지조사)"),
    "연령그룹"               : ("str",   "전기고령자(65-74) / 후기고령자(75+)"),
    "SD_CD"                 : ("str",   "성별코드 (SGIS 가구통계등록부)"),
    "LFNR_SE_CD"            : ("str",   "내외국인구분코드 (SGIS 인구통계등록부)"),
    "BRTH_YR"               : ("str",   "출생연도 (SGIS 인구통계등록부)"),

    # ── 가구 구성 ────────────────────────────────────────────────
    "MBHS_CNT"              : ("Int64", "가구원수 (SGIS 가구통계등록부)"),
    "가구원수"               : ("float", "가구원수 (가계금융복지조사)"),
    "노인가구여부"           : ("str",   "G1=노인가구(전원65+) / G2=그외"),
    "HS_SE_CD"              : ("str",   "가구구분코드"),
    "HS_SHAPE_CD"           : ("str",   "가구형태코드"),
    "HH_FMTN_CD"            : ("str",   "세대구성코드"),
    "HH_HS_TYPE_CD"         : ("str",   "세대가구유형코드"),
    "MCLTR_HS_YN"           : ("str",   "다문화가구여부"),

    # ── 거주 유형 ────────────────────────────────────────────────
    "LVQT_KIND_CD"          : ("str",   "거처종류코드 (SGIS 가구/주택 공통)"),
    "DTHS_TYPE_CD"          : ("str",   "단독주택유형코드 (SGIS 가구/주택 공통)"),
    "LVQT_SN"               : ("str",   "거처일련번호 (가구↔주택 JOIN 키)"),
    "입주형태코드"           : ("str",   "1=자기집 2=전세 3=보증금월세 4=무보증월세 5=기타"),
    "주택종류통합코드"       : ("str",   "G1=단독 G2=아파트 G3=연립다세대 G4=기타"),

    # ── 주택 물리 속성 (주택통계등록부) ─────────────────────────
    "RSDT_AREA"             : ("float", "주거용면적"),
    "SIAR"                  : ("float", "대지면적"),
    "ARCH_APRV_YR"          : ("Int64", "건축승인연도"),
    "BLDG_DLPD_PD_CD"       : ("str",   "건물노후기간코드"),
    "NHAB_HOUS_YN"          : ("str",   "미거주주택여부"),
    "건물연령"               : ("float", "2024 - ARCH_APRV_YR (파생)"),
    "노후건물여부"           : ("bool",  "건물연령 ≥ 30년 (파생)"),

    # ── 자산 (가계금융복지조사) ─────────────────────────────────
    "자산"                   : ("float", "총자산 (만원)"),
    "자산_금융자산"          : ("float", "금융자산 (만원)"),
    "자산_실물자산"          : ("float", "실물자산 (만원)"),
    "자산_실물자산_부동산_거주주택금액": ("float", "거주주택금액 (만원)"),
    "부채"                   : ("float", "총부채 (만원)"),
    "부채_금융부채_담보대출금액": ("float", "담보대출 (만원)"),
    "순자산"                 : ("float", "순자산 (만원)"),

    # ── 소득 (가계금융복지조사) ─────────────────────────────────
    "경상소득(보완)"         : ("float", "경상소득 (만원)"),
    "경상소득_공적이전소득(보완)": ("float", "공적이전소득 (만원)"),
    "처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]":
                               ("float", "처분가능소득 (만원)"),

    # ── 지출 (가계금융복지조사) ─────────────────────────────────
    "지출_소비지출비"        : ("float", "소비지출 (만원)"),
    "지출_소비지출_식료품(외식비포함)": ("float", "식료품비 (만원)"),
    "지출_소비지출_의료비"   : ("float", "의료비 (만원)"),
    "지출_비소비지출(보완)"  : ("float", "비소비지출 (만원)"),
    "지출_비소비지출_세금(보완)": ("float", "세금 (만원)"),
    "지출_비소비지출_공적연금사회보험료(보완)": ("float", "공적연금보험료 (만원)"),
    "지출_비소비지출_연간지급이자(조사2)": ("float", "연간이자 (만원)"),
    "가중값"                 : ("float", "표본 가중치"),

    # ── 은퇴 관련 (가계금융복지조사) ────────────────────────────
    "가구주_은퇴여부"        : ("str",   "1=은퇴안함 / 2=은퇴함"),
    "가구주_미은퇴_최소생활비": ("float", "미은퇴 최소생활비 (만원)"),
    "가구주_미은퇴_적정생활비": ("float", "미은퇴 적정생활비 (만원)"),
    "가구주_은퇴_적정생활비충당여부": ("str",
        "1=충분히여유 2=여유 3=보통 4=부족 5=매우부족"),

    # ── 정책 기준 분류 (파생) ────────────────────────────────────
    "정부기준빈곤선"         : ("float",
        "2026 중위소득 50% 연간 만원 — 경상소득 기준 / 소득인정액 아님"),
    "소득분위_내부"          : ("int",   "고령층 내 처분가능소득 분위 1~5"),
    "자산분위_내부"          : ("int",   "고령층 내 거주주택금액 분위 1~5"),
    "고령층유형"             : ("str",
        "A유형_구조적취약층 / B유형_자산소득불일치(사각지대) / "
        "C유형_기타일반가구 / D유형_여유자산가층"),

    # ── 팀원 A 핵심 파생변수 3종 ────────────────────────────────
    "부동산편중도"           : ("float", "부동산자산/총자산 (0~1)"),
    "부동산편중_고위험"      : ("bool",  "부동산편중도 ≥ 0.80"),
    "D_Value_proxy"          : ("float", "(식료품비+의료비)/처분가능소득"),
    "LTI"                   : ("float", "유동성함정지수 (0~1)"),
    "LTI_등급"               : ("str",   "정상 / 주의 / 경고 / 유동성함정"),
    "AAI_proxy"              : ("float", "부동산편중도×거주주택금액/1만 (자산노후도 프록시)"),

    # ── NICE 대출·연체 연계 (시군구 집계, 파생) ─────────────────
    "nice_대출잔액_평균금액" : ("float", "NICE 대출잔액 평균 (천원)"),
    "nice_대출잔액_주택담보_평균": ("float", "NICE 주택담보대출 평균 (천원)"),
    "nice_총대출잔액_주택담보": ("float", "NICE 주택담보대출 합계 (천원)"),
    "nice_대출평균이자율"    : ("float", "NICE 대출 평균이자율 (%)"),
    "nice_연체비율"          : ("float", "연체금액/총대출잔액"),
    "nice_월_연체보유자수"   : ("float", "NICE 월 연체보유자수 합계 (명)"),
    "nice_평균연체금액"      : ("float", "NICE 평균연체금액 (천원)"),

    # ── NICE 소득 연계 (시군구 집계, 파생) ──────────────────────
    "nice_평균_연소득_금액"  : ("float", "NICE 추정 평균연소득 (천원)"),
    "nice_중위_연소득_금액"  : ("float", "NICE 추정 중위연소득 (천원)"),
    "nice_거주자수"          : ("float", "NICE 해당 구간 거주자수 (명)"),

    # ── 카드소비 연계 (시군구 집계, 파생) ───────────────────────
    "card_총사용금액_65이상" : ("float", "65세이상 카드 총사용금액 (원)"),
    "card_의료업종_사용금액" : ("float", "L코드(의료/건강) 사용금액 (원)"),
    "card_의료비_비중"       : ("float", "의료업종/전체사용금액 비중 (파생)"),
}


# ════════════════════════════════════════════════════════════════
# LAYER 7. Master Table 구성 및 출력
# ════════════════════════════════════════════════════════════════

def print_schema():
    """Master Table 스키마 출력."""
    print("\n" + "="*72)
    print("  Master Table 스키마 (총 항목 검증)")
    print("="*72)
    print(f"  {'컬럼명':<50} {'타입':<7} 설명")
    print("-"*72)

    sources = {
        "식별자": [k for k in MASTER_TABLE_SCHEMA if k in
            ["MD제공용_가구고유번호","HOHL_SPIDN","조사연도","CRTR_YR"]],
        "SGIS 인구통계등록부": list(POPULATION_COLS.keys()),
        "SGIS 가구통계등록부": list(HOUSEHOLD_COLS.keys()),
        "SGIS 주택통계등록부": list(HOUSING_COLS.keys()),
        "가계금융복지조사":  HFWS_COLS,
        "파생변수 (팀원A)": ["부동산편중도","부동산편중_고위험",
                              "D_Value_proxy","LTI","LTI_등급","AAI_proxy",
                              "정부기준빈곤선","소득분위_내부",
                              "자산분위_내부","고령층유형","연령그룹",
                              "건물연령","노후건물여부"],
        "NICE 대출·연체":   [k for k in MASTER_TABLE_SCHEMA if k.startswith("nice_대출") or k.startswith("nice_연체") or k.startswith("nice_월")],
        "NICE 소득":        [k for k in MASTER_TABLE_SCHEMA if k.startswith("nice_평균") or k.startswith("nice_중위") or k.startswith("nice_거주")],
        "카드소비":         [k for k in MASTER_TABLE_SCHEMA if k.startswith("card_")],
    }

    for src, cols in sources.items():
        print(f"\n  [{src}]")
        for c in cols:
            if c in MASTER_TABLE_SCHEMA:
                typ, desc = MASTER_TABLE_SCHEMA[c]
                print(f"    {c:<48} {typ:<7} {desc}")

    print(f"\n  ✓ 총 {len(MASTER_TABLE_SCHEMA)}개 컬럼")
    print("="*72)


def validate_columns(df: pd.DataFrame) -> None:
    """실제 DataFrame 컬럼과 스키마 대조 검증."""
    expected = set(MASTER_TABLE_SCHEMA.keys())
    actual   = set(df.columns)
    missing  = expected - actual
    extra    = actual - expected

    print("\n[컬럼명 검증]")
    if missing:
        print(f"  ⚠ 스키마 정의 but 미존재({len(missing)}): {sorted(missing)}")
    else:
        print("  ✓ 스키마 정의 컬럼 전부 존재")
    if extra:
        print(f"  ℹ 스키마 외 추가 컬럼({len(extra)}): {sorted(extra)[:5]}...")


def build_master_table(hfws: pd.DataFrame, sgis=None,
                       nice_loan_agg=None, nice_income_agg=None,
                       card_agg=None) -> pd.DataFrame:
    """Master Table 최종 구성."""
    master = filter_elderly_hfws(hfws)
    master = create_derived_variables(master)
    master = classify_persona(master)

    # 유형 분포 (가중치 적용 전국 추정)
    dist = master.groupby(["연령그룹","고령층유형"])["가중값"].sum().unstack().fillna(0).astype(int)
    print(f"\n{'='*60}")
    print("  고령층 유형 분포 (가중치 적용 전국 추정)")
    print(f"{'='*60}")
    print(dist)

    b_total = int(master[master["고령층유형"]=="B유형_자산소득불일치"]["가중값"].sum())
    print(f"\n📢 B유형(사각지대) 전국 추정: 약 {b_total:,}가구")
    print("   → 주택연금 최우선 정책 타겟")

    return master


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_schema()
    print("\n[실행 방법]")
    print("  hfws  = load_hfws(2024)")
    print("  nice_l = load_nice_loan(2024)")
    print("  nice_i = load_nice_income(2024)")
    print("  card   = load_card(2024)")
    print("  master = build_master_table(hfws)")
    print("  validate_columns(master)")
