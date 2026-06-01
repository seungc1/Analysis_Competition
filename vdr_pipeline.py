"""
══════════════════════════════════════════════════════════════════
VDR 반출 규정 준수 | Master Table 구성 파이프라인
2026 국가데이터 활용대회 — 고령층 경제 이질성 진단
══════════════════════════════════════════════════════════════════

[VDR 반출 규정 준수 체크리스트]
  ✓ 규정 1: 로우 데이터 반출 금지 → 모든 결과물 groupby 집계
  ✓ 규정 2: K-Anonymity → 건수 < 5 행 자동 삭제
  ✓ 규정 3: 파일 포맷 → .xlsx / .png 저장만 허용
  ✓ 규정 4: inf/NaN 방어 → 모든 연산 전 클렌징

[메모리 최적화 전략]
  - txt 데이터: 청크(chunk) 단위 처리 후 즉시 집계
  - dtype 사전 지정으로 메모리 최대 40% 절감
  - 가구마스터: 최적화 없이 전체 로드 (규정 준수)
  - 월별 12개 파일: 루프 처리 후 집계 결과만 메모리 보유
══════════════════════════════════════════════════════════════════
"""

import sys
# Windows 콘솔 한글 깨짐 방지: stdout/stderr를 UTF-8로 강제 설정
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
import logging
import warnings
import gc

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Jupyter 환경 감지: inline 백엔드가 이미 설정된 경우 Agg 강제 설정 생략
try:
    import IPython
    _in_jupyter = IPython.get_ipython() is not None
except ImportError:
    _in_jupyter = False

if not _in_jupyter:
    matplotlib.use('Agg')           # 화면 출력 없이 파일 저장

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path

# ── 한글 폰트 자동 탐지 (VDR/Jupyter/로컬 환경 공통) ─────────
_KOREAN_FONTS = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic',
                 'Apple SD Gothic Neo', 'AppleGothic', 'Gulim', 'Dotum']
_available = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((f for f in _KOREAN_FONTS if f in _available), None)

if _korean_font:
    plt.rcParams['font.family'] = _korean_font
else:
    # 폰트 없으면 한글 깨지지만 실행은 정상 진행
    pass
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 경로 설정  ★ 데이터 센터 사용 시 USE_VDR = True 로 변경
# ══════════════════════════════════════════════════════════════
USE_VDR = False   # False: 로컬 테스트 | True: VDR 데이터 센터

if USE_VDR:
    DRIVE = Path('/content/drive/MyDrive/Data_Analysis_Competition')
    RDATA = Path('/Rdata1/r1_user138/dataset')
else:
    DRIVE = Path('C:/Users/yooyj/OneDrive/문서/Code/Data/sample_data')
    RDATA = Path('C:/Users/yooyj/OneDrive/문서/Code/Data/sample_data')

# 가구마스터 (Google Drive)
HFWS_2024 = DRIVE / '가계금융복지조사/2024_가구마스터_20260512_38026.csv'
HFWS_2023 = DRIVE / '가계금융복지조사/2023_가구마스터_20260512_38026.csv'

# SGIS 등록부 (VDR)
POP_2023  = RDATA / 'A09_POPULATION/POPULATION_2023.txt'
HHD_2023  = RDATA / 'A10_HOUSEHOLD/HOUSEHOLD_2023.txt'
HSG_2023  = RDATA / 'A11_HOUSING/HOUSING_2023.txt'

# 민간 데이터 기본 경로 (월별 01~12)
CARD_SEOUL_DIR  = RDATA / 'P46_CARD_SALES/03.CARD_DOMESTIC_SEL'
CARD_APT_DIR    = RDATA / 'P46_CARD_SALES/07.CARD_APARTMENT'
NICE_LOAN_DIR   = RDATA / 'P47_NICE_CREDIT/01.LOAN_ADM'
NICE_INCOM_DIR  = RDATA / 'P47_NICE_CREDIT/02.INCOM_ADM'
NICE_POS_DIR    = RDATA / 'P47_NICE_CREDIT/07.POS_SGG'
NH_APT_DIR      = RDATA / 'P49_NHCARD/05.NH_APARTMENT'

# 출력 디렉터리
OUT_XLS = DRIVE / 'output/excel'
OUT_IMG = DRIVE / 'output/images'
OUT_XLS.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

# 구분자
SEP = '|'
# 청크 크기 (메모리 조절)
CHUNK = 200_000

# 2026 기준 중위소득 50% (월→연, 만원)
POVERTY_LINE_2026 = {
    1: round(2_564_238 * 0.5 * 12 / 10_000, 1),
    2: round(4_195_502 * 0.5 * 12 / 10_000, 1),
    3: round(5_357_117 * 0.5 * 12 / 10_000, 1),
    4: round(6_494_738 * 0.5 * 12 / 10_000, 1),
    5: round(7_576_722 * 0.5 * 12 / 10_000, 1),
    6: round(8_604_732 * 0.5 * 12 / 10_000, 1),
}

INC_COL = '처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]'
RE_COL  = '자산_실물자산_부동산_거주주택금액'
WT_COL  = '가중값'

HFWS_COLS = [
    '조사연도', 'MD제공용_가구고유번호', WT_COL, '수도권여부', '가구원수',
    '노인가구여부', '가구주_만연령', '입주형태코드', '주택종류통합코드',
    '자산', '자산_금융자산', '자산_실물자산',
    RE_COL, '부채', '부채_금융부채_담보대출금액', '순자산',
    '경상소득(보완)', '경상소득_공적이전소득(보완)',
    INC_COL,
    '지출_소비지출비', '지출_소비지출_식료품(외식비포함)',
    '지출_소비지출_의료비', '지출_비소비지출(보완)',
    '지출_비소비지출_세금(보완)',
    '지출_비소비지출_공적연금사회보험료(보완)',
    '지출_비소비지출_연간지급이자(조사2)',
    '가구주_은퇴여부', '가구주_미은퇴_최소생활비',
    '가구주_미은퇴_적정생활비', '가구주_은퇴_적정생활비충당여부',
]

NA_VALS = ['*','**','***','****','*****','******','*******',
           '********','*********','**********','.']


# ══════════════════════════════════════════════════════════════
# [공통] VDR 규정 4: inf/NaN 방어 클렌저
# ══════════════════════════════════════════════════════════════
def clean_inf_nan(df: pd.DataFrame, fill_val=0) -> pd.DataFrame:
    """inf/-inf → NaN → fill_val 순서로 클렌징."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = (df[num_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(fill_val))
    return df


# ══════════════════════════════════════════════════════════════
# [공통] VDR 규정 2: K-Anonymity 필터 (건수 < 5 행 제거)
# ══════════════════════════════════════════════════════════════
def apply_k_anonymity(df: pd.DataFrame, count_col: str, k: int = 5) -> pd.DataFrame:
    """집계 후 count_col < k 인 행 제거 → 소수 표본 재식별 차단."""
    before = len(df)
    df = df[df[count_col] >= k].copy()
    print(f"  [K-Anonymity] {before - len(df)}행 제거 → 잔여 {len(df)}행")
    return df


# ══════════════════════════════════════════════════════════════
# [공통] VDR 규정 3: 엑셀 저장
# ══════════════════════════════════════════════════════════════
def save_xlsx(df: pd.DataFrame, filename: str):
    """집계 결과를 .xlsx로 저장 (로우 데이터 저장 금지)."""
    path = OUT_XLS / filename
    df.to_excel(path, index=False, engine='openpyxl')
    print(f"  [저장] {path}")


# ══════════════════════════════════════════════════════════════
# STEP 1. 가계금융복지조사 로드 (최적화 없음 — 규정 준수)
# ══════════════════════════════════════════════════════════════
def load_hfws() -> pd.DataFrame:
    """
    2023·2024 가구마스터 병합.
    규정: 최적화 없이 전체 로드 (요청 사항).
    """
    print("\n[STEP 1] 가계금융복지조사 로드")

    dfs = []
    for path, yr in [(HFWS_2023, 2023), (HFWS_2024, 2024)]:
        df = pd.read_csv(
            path, header=None, sep=',',
            names=HFWS_COLS,
            na_values=NA_VALS,
            skiprows=1,
            encoding='cp949', 
            low_memory=False,
        )
        df['조사연도'] = yr
        dfs.append(df)
        print(f"  {yr}년 로드: {len(df):,}행")

    panel = pd.concat(dfs, ignore_index=True)

    # 규정 4: inf/NaN 방어
    panel = clean_inf_nan(panel, fill_val=0)

    # 고령층 필터 및 연령 그룹
    panel = panel[panel['가구주_만연령'] >= 65].copy()
    panel['연령그룹'] = np.where(panel['가구주_만연령'] < 75,
                                 '전기고령자(65-74)', '후기고령자(75+)')

    # 2026 정부기준빈곤선 (가구원수별 적용)
    panel['정부기준빈곤선'] = panel['가구원수'].apply(
        lambda n: POVERTY_LINE_2026.get(min(int(n) if pd.notna(n) else 1, 6),
                                        POVERTY_LINE_2026[6])
    )

    # 파생변수
    eps = 1e-6
    panel['부동산편중도'] = np.where(
        panel['자산'] > 0, panel[RE_COL] / panel['자산'], np.nan
    )
    panel['부동산편중도'] = panel['부동산편중도'].replace([np.inf, -np.inf], np.nan)

    # 내부 분위
    panel['자산분위_내부'] = pd.qcut(
        panel[RE_COL].rank(method='first'), 5, labels=[1,2,3,4,5]
    ).astype(int)
    panel['소득분위_내부'] = pd.qcut(
        panel[INC_COL].rank(method='first'), 5, labels=[1,2,3,4,5]
    ).astype(int)

    # 유형 분류
    def _classify(r):
        is_poor   = r[INC_COL] <= r['정부기준빈곤선']
        low_inc   = r['소득분위_내부'] <= 2
        high_ast  = r['자산분위_내부'] >= 4
        if is_poor and low_inc and high_ast:
            return 'B유형_자산소득불일치'
        elif is_poor and r['자산분위_내부'] <= 2:
            return 'A유형_구조적취약층'
        elif not low_inc and high_ast:
            return 'D유형_여유자산가층'
        return 'C유형_기타일반가구'

    panel['고령층유형'] = panel.apply(_classify, axis=1)
    print(f"  최종 고령 가구: {len(panel):,}행")
    return panel


# ══════════════════════════════════════════════════════════════
# STEP 1-A. 가계금융복지조사 집계 (규정 1·2·3 적용)
# ══════════════════════════════════════════════════════════════
def aggregate_hfws(panel: pd.DataFrame):
    """
    [규정 1] 로우 반출 금지 → groupby 집계만 저장
    [규정 2] K-Anonymity: 표본수 < 5 행 제거
    [규정 3] .xlsx 저장
    """
    print("\n[STEP 1-A] 가계금융복지조사 집계")

    # ── 집계 1: 유형 × 연령그룹 × 조사연도
    grp1 = panel.groupby(['조사연도','연령그룹','고령층유형']).agg(
        표본수            = (WT_COL,                         'count'),
        추정가구수_가중치 = (WT_COL,                         'sum'),
        평균_처분가능소득 = (INC_COL,                        'mean'),
        평균_거주주택금액 = (RE_COL,                         'mean'),
        평균_순자산       = ('순자산',                       'mean'),
        평균_부채         = ('부채',                         'mean'),
        평균_부동산편중도 = ('부동산편중도',                  'mean'),
        평균_의료비       = ('지출_소비지출_의료비',          'mean'),
        평균_식료품비     = ('지출_소비지출_식료품(외식비포함)', 'mean'),
    ).reset_index()
    grp1 = clean_inf_nan(grp1)
    grp1 = apply_k_anonymity(grp1, '표본수', k=5)
    save_xlsx(grp1, '01_유형별_집계.xlsx')

    # ── 집계 2: 수도권 × 연령그룹 × 유형
    grp2 = panel.groupby(['조사연도','수도권여부','연령그룹','고령층유형']).agg(
        표본수            = (WT_COL,    'count'),
        추정가구수        = (WT_COL,    'sum'),
        평균_처분가능소득 = (INC_COL,   'mean'),
        평균_거주주택금액 = (RE_COL,    'mean'),
        평균_부동산편중도 = ('부동산편중도', 'mean'),
    ).reset_index()
    grp2 = clean_inf_nan(grp2)
    grp2 = apply_k_anonymity(grp2, '표본수', k=5)
    save_xlsx(grp2, '02_수도권별_유형집계.xlsx')

    # ── 집계 3: 주택종류 × 연령그룹
    grp3 = panel.groupby(['조사연도','주택종류통합코드','연령그룹']).agg(
        표본수            = (WT_COL,    'count'),
        추정가구수        = (WT_COL,    'sum'),
        평균_순자산       = ('순자산',   'mean'),
        평균_거주주택금액 = (RE_COL,    'mean'),
        B유형_비율 = ('고령층유형',
                      lambda x: (x == 'B유형_자산소득불일치').mean()),
    ).reset_index()
    grp3 = clean_inf_nan(grp3)
    grp3 = apply_k_anonymity(grp3, '표본수', k=5)
    save_xlsx(grp3, '03_주택종류별_집계.xlsx')

    print("  [완료] 가계금융복지조사 집계 3종")
    return grp1


# ══════════════════════════════════════════════════════════════
# STEP 2. SGIS 등록부 — 청크 처리 (메모리 최적화)
# ══════════════════════════════════════════════════════════════

# dtype 사전 지정 — 메모리 절감 핵심
POP_DTYPE = {
    'CRTR_YR': 'category', 'ADMDST_CLSF_CD': 'category',
    'HOHL_SPIDN': str, 'SPIDN': str,
    'LFNR_SE_CD': 'category', 'HOHL_REL_CD': 'category',
    'SD_CD': 'category', 'NTNLTY_CD': 'category',
    'HS_SHAPE_CD': 'category', 'BRTH_YR': 'category', 'ENTCNY_YR': 'category',
}
HHD_DTYPE = {
    'CRTR_YR': 'category', 'ADMDST_CLSF_CD': 'category',
    'HOHL_SPIDN': str, 'SD_CD': 'category',
    'HS_SHAPE_CD': 'category', 'HS_SE_CD': 'category',
    'LVQT_KIND_CD': 'category', 'DTHS_TYPE_CD': 'category',
    'LVQT_SN': str, 'HH_FMTN_CD': 'category',
    'HH_HS_TYPE_CD': 'category', 'MCLTR_HS_YN': 'category',
}
HSG_DTYPE = {
    'CRTR_YR': 'category', 'ADMDST_CLSF_CD': 'category',
    'LVQT_KIND_CD': 'category', 'DTHS_TYPE_CD': 'category',
    'BLDG_DLPD_PD_CD': 'category', 'NHAB_HOUS_YN': 'category',
    'LVQT_SN': str,
}


def _open_txt(path: Path, sep: str, dtype: dict, chunk_size: int):
    """인코딩 자동 감지: UTF-8 실패 시 CP949로 재시도."""
    for enc in ('utf-8', 'cp949'):
        try:
            return pd.read_csv(
                path, sep=sep, dtype=dtype, chunksize=chunk_size,
                encoding=enc, on_bad_lines='skip', low_memory=False,
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 감지 실패: {path}")


def read_txt_chunked(path: Path, dtype: dict, sep: str = SEP,
                     chunk_size: int = CHUNK) -> pd.DataFrame:
    """
    대용량 txt 파일을 청크 단위로 읽어 반환.
    전체 행이 필요한 경우에만 사용 (SGIS HHD 등).
    청크를 분할 적재한 뒤 concat — 파일이 수천만 행이면
    aggregate_sgis 내부에서 직접 필터 후 집계 권장.
    """
    chunks = []
    reader = _open_txt(path, sep, dtype, chunk_size)
    for i, chunk in enumerate(reader):
        chunk = clean_inf_nan(chunk)
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"    청크 {i+1} 처리 중...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def aggregate_sgis():
    """
    SGIS 3개 등록부 처리.
    [규정 1] 가구 단위 JOIN 후 즉시 시군구 단위 집계
    [규정 2] K-Anonymity 적용
    [규정 3] .xlsx 저장
    """
    print("\n[STEP 2] SGIS 등록부 처리")

    # 가구통계등록부 로드
    print("  가구통계등록부 로드 중...")
    hhd = read_txt_chunked(HHD_2023, HHD_DTYPE)
    hhd['AAGE'] = pd.to_numeric(hhd.get('AAGE', pd.Series()), errors='coerce')
    hhd['MBHS_CNT'] = pd.to_numeric(hhd.get('MBHS_CNT', pd.Series()), errors='coerce')
    hhd['시군구코드'] = hhd['ADMDST_CLSF_CD'].astype(str).str[:5]

    # 65세 이상 필터 — 메모리 즉시 절감
    hhd_elderly = hhd[hhd['AAGE'] >= 65].copy()
    hhd_elderly['연령그룹'] = np.where(
        hhd_elderly['AAGE'] < 75, '전기고령자(65-74)', '후기고령자(75+)'
    )
    del hhd
    gc.collect()
    print(f"  고령 가구 필터: {len(hhd_elderly):,}행")

    # 주택통계등록부 청크 처리 후 즉시 집계
    print("  주택통계등록부 처리 중...")
    hsg_chunks = []
    reader = _open_txt(HSG_2023, SEP, HSG_DTYPE, CHUNK)
    for chunk in reader:
        chunk['ARCH_APRV_YR'] = pd.to_numeric(
            chunk.get('ARCH_APRV_YR', pd.Series()), errors='coerce'
        )
        chunk['RSDT_AREA'] = pd.to_numeric(
            chunk.get('RSDT_AREA', pd.Series()), errors='coerce'
        )
        chunk['건물연령'] = 2023 - chunk['ARCH_APRV_YR'].fillna(2023)
        chunk['노후건물'] = (chunk['건물연령'] >= 30).astype(int)
        # 청크 내 LVQT_SN 단위 집계 후 버퍼
        agg = chunk.groupby('LVQT_SN').agg(
            평균_주거면적 = ('RSDT_AREA', 'mean'),
            노후건물수    = ('노후건물',   'sum'),
            건물수        = ('노후건물',   'count'),
        ).reset_index()
        hsg_chunks.append(agg)
        del chunk
    hsg = pd.concat(hsg_chunks, ignore_index=True).groupby('LVQT_SN').sum().reset_index()
    hsg['노후건물비율'] = np.where(
        hsg['건물수'] > 0, hsg['노후건물수'] / hsg['건물수'], np.nan
    )
    del hsg_chunks
    gc.collect()

    # 가구 + 주택 JOIN
    merged = hhd_elderly.merge(
        hsg[['LVQT_SN', '평균_주거면적', '노후건물비율']],
        on='LVQT_SN', how='left'
    )
    merged = clean_inf_nan(merged)
    del hhd_elderly, hsg
    gc.collect()

    # [규정 1] 시군구 × 연령그룹 × 거처종류 집계
    grp = merged.groupby(['시군구코드','연령그룹','LVQT_KIND_CD']).agg(
        가구수          = ('HOHL_SPIDN', 'count'),
        평균_가구원수   = ('MBHS_CNT',   'mean'),
        평균_주거면적   = ('평균_주거면적', 'mean'),
        평균_노후건물비율 = ('노후건물비율', 'mean'),
        아파트비율      = ('HS_SHAPE_CD',
                           lambda x: (x == 'G2').mean()),
    ).reset_index()
    grp = clean_inf_nan(grp)
    # [규정 2] K-Anonymity
    grp = apply_k_anonymity(grp, '가구수', k=5)
    save_xlsx(grp, '04_SGIS_시군구별_주거현황.xlsx')

    del merged
    gc.collect()
    print("  [완료] SGIS 집계")


# ══════════════════════════════════════════════════════════════
# STEP 3. 월별 txt 파일 처리 유틸
# ══════════════════════════════════════════════════════════════

def iter_monthly_files(directory: Path, prefix: str,
                       suffix: str = '.txt', months=range(1, 13)):
    """
    01~12 월별 파일을 순서대로 yield.
    예: CARD_DOMESTIC_SEOUL_202401.txt ~ 202412.txt
    """
    for m in months:
        fname = f"{prefix}{m:02d}{suffix}"
        p = directory / fname
        if p.exists():
            yield m, p
        else:
            print(f"    [없음] {p.name}")


# ══════════════════════════════════════════════════════════════
# STEP 4. 카드소비(서울) — 월별 청크 처리
# ══════════════════════════════════════════════════════════════

CARD_SEOUL_DTYPE = {
    '기준연월': 'category', '공휴일구분코드': 'category',
    '가맹점행정구역분류시도코드': 'category',
    '가맹점행정구역분류시군구코드': 'category',
    '가맹점행정구역분류코드': str,
    '통합카드업종3레벨코드': 'category',
    '조직구분코드': 'category',
    '통합카드성별코드': 'category',
    '통합카드5세단위연령코드': 'category',
    '통합카드직업코드': 'category',
    '고객행정구역분류시군구코드': 'category',
    '통합카드혼인구분코드': 'category',
    '통합카드가구형태코드': 'category',
}


def aggregate_card_seoul():
    """
    내국인 국내카드소비(서울) — 65세 이상 × 시군구 × 업종 집계.
    [규정 1] 집계 결과만 반출
    [규정 2] 건수 < 5 제거
    [규정 3] .xlsx 저장
    """
    print("\n[STEP 4] 카드소비(서울) 월별 처리")

    monthly_agg = []
    prefix = 'CARD_DOMESTIC_SEOUL_2024'

    for month, fpath in iter_monthly_files(CARD_SEOUL_DIR, prefix, suffix='.txt'):
        print(f"  처리: {fpath.name}")
        m_chunks = []
        for chunk in _open_txt(fpath, SEP, CARD_SEOUL_DTYPE, CHUNK):
            # 65세 이상(연령코드 14·15) + 개인(1) + 노인가구(5) 필터
            mask = (
                chunk['통합카드5세단위연령코드'].isin(['14', '15']) &
                (chunk['조직구분코드'] == '1') &
                (chunk['통합카드가구형태코드'] == '5')
            )
            sub = chunk[mask].copy()
            if sub.empty:
                continue
            sub['카드사용금액'] = pd.to_numeric(sub.get('카드사용금액', 0), errors='coerce').fillna(0)
            sub['카드사용건수'] = pd.to_numeric(sub.get('카드사용건수', 0), errors='coerce').fillna(0)
            sub['1인당평균사용금액'] = pd.to_numeric(
                sub.get('1인당평균사용금액', 0), errors='coerce').fillna(0)

            # 청크 내 집계
            agg = sub.groupby([
                '가맹점행정구역분류시군구코드',
                '통합카드업종3레벨코드',
                '통합카드5세단위연령코드',
                '공휴일구분코드',
            ]).agg(
                카드사용건수합   = ('카드사용건수',       'sum'),
                카드사용금액합   = ('카드사용금액',       'sum'),
                평균1인당사용금액 = ('1인당평균사용금액', 'mean'),
                집계행수         = ('카드사용건수',       'count'),
            ).reset_index()
            agg['기준연월'] = f'2024{month:02d}'
            m_chunks.append(agg)
            del sub, chunk

        if m_chunks:
            monthly_agg.append(pd.concat(m_chunks, ignore_index=True))
        del m_chunks
        gc.collect()

    if not monthly_agg:
        print("  [경고] 카드소비 데이터 없음")
        return

    result = pd.concat(monthly_agg, ignore_index=True)
    result = clean_inf_nan(result)

    # 연간 집계 (시군구 × 업종)
    annual = result.groupby([
        '가맹점행정구역분류시군구코드',
        '통합카드업종3레벨코드',
        '통합카드5세단위연령코드',
    ]).agg(
        연간_카드사용건수 = ('카드사용건수합',    'sum'),
        연간_카드사용금액 = ('카드사용금액합',    'sum'),
        평균1인당사용금액 = ('평균1인당사용금액', 'mean'),
        집계행수          = ('집계행수',          'sum'),
    ).reset_index()
    annual = clean_inf_nan(annual)
    # [규정 2] 카드 건수 5 미만 제거
    annual = apply_k_anonymity(annual, '연간_카드사용건수', k=5)
    save_xlsx(annual, '05_카드소비서울_시군구업종별.xlsx')

    del monthly_agg, result, annual
    gc.collect()
    print("  [완료] 카드소비(서울) 집계")


# ══════════════════════════════════════════════════════════════
# STEP 5. NICE 대출·연체 — 월별 처리
# ══════════════════════════════════════════════════════════════

NICE_LOAN_DTYPE = {
    '기준년월': 'category', '구분명': 'category',
    '광역시도코드': 'category', '시군구코드(개정후)': 'category',
    '행정구역분류코드(개정후)': str,
    '성별': 'category', '연령구간대': 'category',
    '직업구분': 'category', '금융기관': 'category',
    '분위코드(총대출잔액기준)': 'category',
}
# 주의: '월 연체보유자수 합계' — 공백 포함 항목명


def aggregate_nice_loan():
    """
    NICE 대출·연체 월별 집계.
    연령구간대: 65(65~70세미만), 70(70세 이상) 필터
    직업구분: 0(전체) 만 사용
    """
    print("\n[STEP 5] NICE 대출·연체 월별 처리")

    monthly = []
    prefix = 'NICE_LOAN_AGE_2024'

    for month, fpath in iter_monthly_files(NICE_LOAN_DIR, prefix, suffix='.txt'):
        print(f"  처리: {fpath.name}")
        m_chunks = []
        for chunk in _open_txt(fpath, SEP, NICE_LOAN_DTYPE, CHUNK):
            mask = (
                chunk['연령구간대'].isin(['65', '70']) &
                (chunk['직업구분'] == '0') &
                (chunk['구분명'] == 'CNTY_GU')   # 시군구 단위만
            )
            sub = chunk[mask].copy()
            if sub.empty:
                continue

            # 수치 컬럼 변환
            num_cols = [
                '대출잔액_보유대상자수', '대출잔액_평균금액',
                '대출잔액_주택담보_평균금액', '대출잔액_신용_평균금액',
                '총대출잔액', '총대출잔액_주택담보', '총대출잔액_신용',
                '대출평균이자율',
                '월 연체보유자수 합계',  # 공백 포함 항목명 준수
                '평균연체금액', '연체금액',
            ]
            for c in num_cols:
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0)

            agg = sub.groupby([
                '시군구코드(개정후)', '연령구간대', '성별'
            ]).agg(
                대출보유자수            = ('대출잔액_보유대상자수',     'sum'),
                평균_대출잔액           = ('대출잔액_평균금액',         'mean'),
                평균_주담대_잔액        = ('대출잔액_주택담보_평균금액', 'mean'),
                총대출잔액              = ('총대출잔액',                'sum'),
                총주담대_잔액           = ('총대출잔액_주택담보',        'sum'),
                평균이자율              = ('대출평균이자율',             'mean'),
                월_연체보유자수         = ('월 연체보유자수 합계',       'sum'),
                평균_연체금액           = ('평균연체금액',              'mean'),
                연체금액합              = ('연체금액',                  'sum'),
            ).reset_index()
            agg['기준년월'] = f'2024{month:02d}'
            m_chunks.append(agg)
            del sub, chunk

        if m_chunks:
            monthly.append(pd.concat(m_chunks, ignore_index=True))
        del m_chunks
        gc.collect()

    if not monthly:
        print("  [경고] NICE 대출·연체 데이터 없음")
        return

    result = pd.concat(monthly, ignore_index=True)
    result = clean_inf_nan(result)

    # 연간 집계
    annual = result.groupby(['시군구코드(개정후)', '연령구간대', '성별']).agg(
        대출보유자수     = ('대출보유자수',     'sum'),
        평균_대출잔액    = ('평균_대출잔액',    'mean'),
        평균_주담대_잔액 = ('평균_주담대_잔액', 'mean'),
        총대출잔액       = ('총대출잔액',       'sum'),
        총주담대_잔액    = ('총주담대_잔액',    'sum'),
        평균이자율       = ('평균이자율',       'mean'),
        월_연체보유자수  = ('월_연체보유자수',  'sum'),
        평균_연체금액    = ('평균_연체금액',    'mean'),
        연체금액합       = ('연체금액합',       'sum'),
    ).reset_index()

    annual['연체비율'] = np.where(
        annual['총대출잔액'] > 0,
        annual['연체금액합'] / annual['총대출잔액'], 0
    )
    annual = clean_inf_nan(annual)
    # [규정 2] 대출보유자수 < 5 제거
    annual = apply_k_anonymity(annual, '대출보유자수', k=5)
    save_xlsx(annual, '06_NICE_대출연체_시군구별.xlsx')

    del monthly, result, annual
    gc.collect()
    print("  [완료] NICE 대출·연체 집계")


# ══════════════════════════════════════════════════════════════
# STEP 6. NICE 소득 — 월별 처리
# ══════════════════════════════════════════════════════════════

NICE_INCOM_DTYPE = {
    '기준년월': 'category', '구분명': 'category',
    '광역시도코드': 'category', '시군구코드(개정후)': 'category',
    '행정구역분류코드(개정후)': str,
    '성별': 'category', '연령구간대': 'category',
    '직업 구분': 'category',          # 공백 포함 항목명 준수
    '분위코드(연소득기준)': 'category',
}
# 주의: '직업 구분', '평균 연소득 금액', '중위 연소득 금액' — 공백 포함


def aggregate_nice_income():
    """
    NICE 소득 월별 집계.
    '직업 구분'(공백 포함) '평균 연소득 금액'(공백 포함) 항목명 준수.
    """
    print("\n[STEP 6] NICE 소득 월별 처리")

    monthly = []
    prefix = 'NICE_INCOM_AGE_2024'

    for month, fpath in iter_monthly_files(NICE_INCOM_DIR, prefix, suffix='.txt'):
        print(f"  처리: {fpath.name}")
        m_chunks = []
        for chunk in _open_txt(fpath, SEP, NICE_INCOM_DTYPE, CHUNK):
            mask = (
                chunk['연령구간대'].isin(['65', '70']) &
                (chunk['직업 구분'] == '0') &       # 공백 포함 항목명
                (chunk['구분명'] == 'CNTY_GU')
            )
            sub = chunk[mask].copy()
            if sub.empty:
                continue

            # 공백 포함 항목명 그대로 접근
            for c in ['최저 연소득 금액', '최대 연소득 금액',
                      '평균 연소득 금액', '중위 연소득 금액', '총연소득합계', '거주자수']:
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0)

            agg = sub.groupby(['시군구코드(개정후)', '연령구간대', '성별']).agg(
                거주자수합            = ('거주자수',        'sum'),
                평균_연소득_금액      = ('평균 연소득 금액', 'mean'),
                중위_연소득_금액      = ('중위 연소득 금액', 'mean'),
                총연소득합계          = ('총연소득합계',     'sum'),
            ).reset_index()
            agg['기준년월'] = f'2024{month:02d}'
            m_chunks.append(agg)
            del sub, chunk

        if m_chunks:
            monthly.append(pd.concat(m_chunks, ignore_index=True))
        del m_chunks
        gc.collect()

    if not monthly:
        print("  [경고] NICE 소득 데이터 없음")
        return

    result = pd.concat(monthly, ignore_index=True)
    result = clean_inf_nan(result)

    annual = result.groupby(['시군구코드(개정후)', '연령구간대', '성별']).agg(
        거주자수합       = ('거주자수합',       'sum'),
        평균_연소득_금액 = ('평균_연소득_금액', 'mean'),
        중위_연소득_금액 = ('중위_연소득_금액', 'mean'),
        총연소득합계     = ('총연소득합계',     'sum'),
    ).reset_index()
    annual = clean_inf_nan(annual)
    # [규정 2] 거주자수 < 5 제거
    annual = apply_k_anonymity(annual, '거주자수합', k=5)
    save_xlsx(annual, '07_NICE_소득_시군구별.xlsx')

    del monthly, result, annual
    gc.collect()
    print("  [완료] NICE 소득 집계")


# ══════════════════════════════════════════════════════════════
# STEP 7. 아파트단지별소비(통합카드) — 월별 처리
# ══════════════════════════════════════════════════════════════

CARD_APT_DTYPE = {
    '기준연월': 'category',
    '고객행정구역분류시도코드': 'category',
    '고객행정구역분류시군구코드': 'category',
    '통합카드아파트단지코드': str,
    '통합카드성별코드': 'category',
    '통합카드10세단위연령코드': 'category',
}


def aggregate_card_apt():
    """
    아파트단지별소비(통합카드) — 60대 이상 + 시군구 집계.
    행정동·단지 단위 수치 반출 금지 → 시군구 집계 후 저장.
    """
    print("\n[STEP 7] 아파트단지별소비(통합카드) 처리")

    monthly = []
    prefix = 'CARD_APARTMENT_2024'

    for month, fpath in iter_monthly_files(CARD_APT_DIR, prefix, suffix='.txt'):
        print(f"  처리: {fpath.name}")
        m_chunks = []
        for chunk in _open_txt(fpath, SEP, CARD_APT_DTYPE, CHUNK):
            # 60대 이상(6) 필터
            sub = chunk[chunk['통합카드10세단위연령코드'] == '6'].copy()
            if sub.empty:
                continue

            관심사_cols = [
                '건강관심비율', '골프관심비율', '독서관심비율',
                '여행관심비율', '식도락관심비율', '뷰티관심비율',
                '영화관심비율', '온라인쇼핑관심비율', '쇼핑관심비율',
            ]
            for c in ['연평균소득금액'] + 관심사_cols:
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0)

            agg = sub.groupby([
                '고객행정구역분류시군구코드', '통합카드성별코드'
            ]).agg(
                단지수           = ('통합카드아파트단지코드', 'nunique'),
                평균_연평균소득  = ('연평균소득금액',         'mean'),
                **{f'평균_{c}': (c, 'mean') for c in 관심사_cols 
                   if c in sub.columns},
            ).reset_index()
            agg['기준연월'] = f'2024{month:02d}'
            m_chunks.append(agg)
            del sub, chunk

        if m_chunks:
            monthly.append(pd.concat(m_chunks, ignore_index=True))
        del m_chunks
        gc.collect()

    if not monthly:
        print("  [경고] 아파트단지별소비 데이터 없음")
        return

    result = pd.concat(monthly, ignore_index=True)
    result = clean_inf_nan(result)

    annual = result.groupby(['고객행정구역분류시군구코드', '통합카드성별코드']).agg(
        평균_단지수      = ('단지수',          'mean'),
        평균_연평균소득  = ('평균_연평균소득',  'mean'),
        **{c: (c, 'mean') for c in result.columns 
           if c.startswith('평균_') and '관심' in c},
    ).reset_index()
    annual['집계단지수'] = annual['평균_단지수'].round(0).astype(int)
    annual = clean_inf_nan(annual)
    # [규정 2] 단지수 < 5 제거
    annual = apply_k_anonymity(annual, '집계단지수', k=5)
    save_xlsx(annual, '08_아파트단지소비_시군구별.xlsx')

    del monthly, result, annual
    gc.collect()
    print("  [완료] 아파트단지별소비 집계")


# ══════════════════════════════════════════════════════════════
# STEP 8. 농협카드 아파트 단지별소비 — 동일 패턴 처리
# ══════════════════════════════════════════════════════════════

def aggregate_nh_apt():
    """농협카드 아파트단지별소비 — CARD_APT와 동일 로직."""
    print("\n[STEP 8] 농협카드 아파트단지별소비 처리")

    monthly = []
    prefix = 'NHCRD_APARTMENT_2024'

    for month, fpath in iter_monthly_files(NH_APT_DIR, prefix, suffix='.txt'):
        print(f"  처리: {fpath.name}")
        m_chunks = []
        for chunk in _open_txt(fpath, SEP, CARD_APT_DTYPE, CHUNK):
            sub = chunk[chunk['통합카드10세단위연령코드'] == '6'].copy()
            if sub.empty:
                continue
            if '연평균소득금액' in sub.columns:
                sub['연평균소득금액'] = pd.to_numeric(
                    sub['연평균소득금액'], errors='coerce').fillna(0)
            agg = sub.groupby(
                ['고객행정구역분류시군구코드', '통합카드성별코드']
            ).agg(
                단지수          = ('통합카드아파트단지코드', 'nunique'),
                평균_연평균소득 = ('연평균소득금액',         'mean'),
            ).reset_index()
            agg['기준연월'] = f'2024{month:02d}'
            m_chunks.append(agg)
            del sub, chunk

        if m_chunks:
            monthly.append(pd.concat(m_chunks, ignore_index=True))
        del m_chunks
        gc.collect()

    if not monthly:
        print("  [경고] 농협카드 데이터 없음")
        return

    result = pd.concat(monthly, ignore_index=True)
    result = clean_inf_nan(result)
    annual = result.groupby(
        ['고객행정구역분류시군구코드', '통합카드성별코드']
    ).agg(
        집계단지수      = ('단지수',          'mean'),
        평균_연평균소득 = ('평균_연평균소득',  'mean'),
    ).reset_index()
    annual['집계단지수'] = annual['집계단지수'].round(0).astype(int)
    annual = clean_inf_nan(annual)
    annual = apply_k_anonymity(annual, '집계단지수', k=5)
    save_xlsx(annual, '09_농협카드_아파트단지소비_시군구별.xlsx')

    del monthly, result, annual
    gc.collect()
    print("  [완료] 농협카드 아파트단지별소비 집계")


# ══════════════════════════════════════════════════════════════
# STEP 9. 외식업물가(NICE) — 단일 파일
# ══════════════════════════════════════════════════════════════

def aggregate_pos_sgg():
    """외식업물가 시군구 — 단일 파일, 청크 처리."""
    print("\n[STEP 9] 외식업물가 처리")

    fpath = NICE_POS_DIR / 'TN_PD_NICE_POG_SGG_202501.txt'
    if not fpath.exists():
        print("  [경고] 파일 없음")
        return

    chunks = []
    for chunk in _open_txt(fpath, SEP, {}, CHUNK):
        chunk = clean_inf_nan(chunk)
        chunks.append(chunk)

    if not chunks:
        return

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # 시군구 단위 집계 (컬럼명은 실제 파일 확인 후 조정)
    group_cols = [c for c in df.columns 
                  if '시군구' in c or '코드' in c][:3]
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()

    if group_cols and num_cols:
        grp = df.groupby(group_cols)[num_cols].agg(['mean', 'count']).reset_index()
        grp.columns = ['_'.join(c).strip('_') for c in grp.columns]
        # count 컬럼 찾아서 K-Anonymity 적용
        cnt_col = [c for c in grp.columns if 'count' in c]
        if cnt_col:
            grp = grp[grp[cnt_col[0]] >= 5]
        save_xlsx(grp, '10_외식업물가_시군구별.xlsx')

    del df
    gc.collect()
    print("  [완료] 외식업물가 집계")


# ══════════════════════════════════════════════════════════════
# STEP 10. 시각화 — VDR 규정 3 준수
# ══════════════════════════════════════════════════════════════

def create_visualizations(grp1: pd.DataFrame):
    """
    [규정 3] 행정동 이하 단위 수치 반출 금지
             → 시각화 이미지(.png)로만 저장
             → 라벨링(수치 표기) 없는 상대적 분포도
    """
    print("\n[STEP 10] 시각화 생성")

    # ── 그림 1: B유형 비중 막대 (수치 라벨 없음)
    b_data = grp1[grp1['조사연도'] == 2024].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = b_data.pivot_table(
        index='연령그룹', columns='고령층유형',
        values='추정가구수_가중치', aggfunc='sum', fill_value=0
    )
    pivot.plot(kind='bar', ax=ax, width=0.7)
    ax.set_title('고령층 유형별 전국 추정 가구 분포 (2024)', fontsize=13)
    ax.set_xlabel('연령 그룹')
    ax.set_ylabel('추정 가구 수')
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(axis='x', rotation=0)
    # 수치 라벨 미표시 (규정 3)
    fig.tight_layout()
    fig.savefig(OUT_IMG / '01_유형별_분포.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [저장] 01_유형별_분포.png")

    # ── 그림 2: 자산·소득 산점도 (라벨 없음 — 상대 분포만)
    if '평균_거주주택금액' in b_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for utype, color in zip(
            ['A유형_구조적취약층', 'B유형_자산소득불일치',
             'C유형_기타일반가구', 'D유형_여유자산가층'],
            ['#3B8BD4', '#E8593C', '#B4B2A9', '#1D9E75']
        ):
            sub = b_data[b_data['고령층유형'] == utype]
            if sub.empty:
                continue
            ax.scatter(sub['평균_처분가능소득'], sub['평균_거주주택금액'],
                       label=utype, color=color, alpha=0.7, s=80)
        ax.set_title('유형별 평균 소득-자산 분포 (2024, 수치 미표기)', fontsize=12)
        ax.set_xlabel('평균 처분가능소득 상대값')
        ax.set_ylabel('평균 거주주택금액 상대값')
        # 축 눈금 수치 제거 (규정 3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_IMG / '02_자산소득_산점도.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [저장] 02_자산소득_산점도.png")

    # ── 그림 3: 연도별 B유형 추정 가구 추이 (수치 라벨 없음)
    trend = grp1[grp1['고령층유형'] == 'B유형_자산소득불일치'].groupby(
        '조사연도')['추정가구수_가중치'].sum().reset_index()
    if len(trend) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(trend['조사연도'], trend['추정가구수_가중치'],
                'o-', color='#E8593C', lw=2.5, ms=8)
        ax.fill_between(trend['조사연도'], trend['추정가구수_가중치'],
                        alpha=0.15, color='#E8593C')
        ax.set_title('B유형(자산-소득 불일치) 추정 가구 추이', fontsize=12)
        ax.set_xlabel('조사연도')
        ax.set_ylabel('추정 가구 수')
        ax.set_yticklabels([])  # 수치 미표기 (규정 3)
        fig.tight_layout()
        fig.savefig(OUT_IMG / '03_B유형_추이.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [저장] 03_B유형_추이.png")

    print("  [완료] 시각화 저장")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  VDR 반출 규정 준수 파이프라인 시작")
    print("=" * 62)

    # STEP 1: 가계금융복지조사 (최적화 없음)
    panel = load_hfws()
    grp1  = aggregate_hfws(panel)
    del panel
    gc.collect()

    # STEP 2: SGIS 등록부 (청크)
    aggregate_sgis()

    # STEP 4~9: 민간 데이터 (월별 청크)
    aggregate_card_seoul()
    aggregate_nice_loan()
    aggregate_nice_income()
    aggregate_card_apt()
    aggregate_nh_apt()
    aggregate_pos_sgg()

    # STEP 10: 시각화 (.png 저장)
    create_visualizations(grp1)

    print("\n" + "=" * 62)
    print("  파이프라인 완료")
    print(f"  엑셀 출력: {OUT_XLS}")
    print(f"  이미지 출력: {OUT_IMG}")
    print("=" * 62)


if __name__ == '__main__':
    main()
