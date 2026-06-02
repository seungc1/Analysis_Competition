# INSTRUCTIONS2.md
# vdr_pipeline.py 개선 작업 — 3가지 변경사항 적용

## 작업 개요
외부 팀원 코드(유승찬_실행소스코드.py) 분석 결과,
우리 vdr_pipeline.py에 적용할 개선사항 3가지를 반영한다.

## 수정 대상 파일
- vdr_pipeline.py (주 수정 파일)
- week3_4_feature_engineering.py (연관규칙 함수 교체)

---

## 변경 1: 빈곤선 — 연도별 동적 적용으로 교체

### 배경
현재 코드는 2026년 기준 단일 상수(POVERTY_LINE_2026)를 사용한다.
2023·2024 데이터를 합쳐 분석할 때 각 연도의 실제 정부 고시값을 적용해야 정확하다.

### [1-1] vdr_pipeline.py — 상수 교체

현재 코드 (74~82번째 줄):
```python
# 2026 기준 중위소득 50% (월→연, 만원)
POVERTY_LINE_2026 = {
    1: round(2_564_238 * 0.5 * 12 / 10_000, 1),
    2: round(4_195_502 * 0.5 * 12 / 10_000, 1),
    3: round(5_357_117 * 0.5 * 12 / 10_000, 1),
    4: round(6_494_738 * 0.5 * 12 / 10_000, 1),
    5: round(7_576_722 * 0.5 * 12 / 10_000, 1),
    6: round(8_604_732 * 0.5 * 12 / 10_000, 1),
}
```

교체할 코드:
```python
# 연도별 정부 고시 기준 중위소득 100% (월, 원)
# 출처: 보건복지부 고시 — 각 연도 실제 적용값
MEDIAN_INCOME_BY_YEAR = {
    2023: {1: 2_077_892, 2: 3_456_155, 3: 4_434_816,
           4: 5_400_964, 5: 6_330_688, 6: 7_227_981, 7: 8_107_515},
    2024: {1: 2_228_445, 2: 3_682_609, 3: 4_714_657,
           4: 5_729_913, 5: 6_695_735, 6: 7_618_369, 7: 8_514_994},
}
# 7인 초과 시 1인 추가당 가산액
MEDIAN_INCOME_ADD = {2023: 879_534, 2024: 896_625}
```

### [1-2] vdr_pipeline.py — 함수 추가

`MEDIAN_INCOME_BY_YEAR` 상수 선언 바로 아래에 다음 함수를 추가한다:
```python
def get_dynamic_poverty_line(year: int, num_members: int) -> float:
    """
    연도·가구원수별 정부 고시 빈곤선(중위소득 50%) 산출.
    - 데이터가 없는 연도는 가장 최신 연도 값으로 대체
    - 반환값: 연간 만원 단위
    """
    year = int(year)
    num = int(num_members) if not pd.isna(num_members) else 1

    # 연도 데이터 없으면 최신 연도 사용
    if year not in MEDIAN_INCOME_BY_YEAR:
        year = max(MEDIAN_INCOME_BY_YEAR.keys())

    tbl = MEDIAN_INCOME_BY_YEAR[year]
    add = MEDIAN_INCOME_ADD[year]

    if num <= 7:
        monthly_100 = tbl[num]
    else:
        monthly_100 = tbl[7] + (num - 7) * add

    return round((monthly_100 * 0.5 * 12) / 10_000, 1)
```

### [1-3] vdr_pipeline.py — load_hfws() 내부 수정

`load_hfws()` 함수 내에서 현재:
```python
    # 2026 정부기준빈곤선 (가구원수별 적용)
    panel['정부기준빈곤선'] = panel['가구원수'].apply(
        lambda n: POVERTY_LINE_2026.get(min(int(n) if pd.notna(n) else 1, 6),
                                        POVERTY_LINE_2026[6])
    )
```
를 다음으로 교체한다:
```python
    # 연도별 동적 빈곤선 (2023·2024 각각 다른 정부 고시값 적용)
    panel['정부기준빈곤선'] = panel.apply(
        lambda r: get_dynamic_poverty_line(r['조사연도'], r['가구원수']),
        axis=1
    )
```

### [1-4] vdr_pipeline.py — 파생변수 추가

`load_hfws()` 함수 내 `부동산편중도` 계산 코드 바로 아래에 다음을 추가한다:
```python
    # 경직적_비용비중: 세금+공적보험료 / 경상소득 (유승찬 코드 Block 1 채택)
    # 의미: 소득 중 줄일 수 없는 고정 지출 비중 → 높을수록 가처분소득 압박
    panel['경직적_비용비중'] = (
        panel['지출_비소비지출_세금(보완)'].fillna(0) +
        panel['지출_비소비지출_공적연금사회보험료(보완)'].fillna(0)
    ) / (panel['경상소득(보완)'] + 1e-6)
    panel['경직적_비용비중'] = panel['경직적_비용비중'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
```

---

## 변경 2: 분산(var) 집계 추가 — 다각도 그루핑 매트릭스

### 배경
평균(mean)만으로는 "같은 수도권 아파트라도 강남·노원의 격차가 극심하다"는
양극화 논리를 증명할 수 없다. 분산(var)을 함께 산출해야 한다.

### [2-1] vdr_pipeline.py — aggregate_hfws() 함수 내부에 집계 4 추가

`aggregate_hfws()` 함수 내에서 `save_xlsx(grp3, '03_주택종류별_집계.xlsx')` 라인
바로 아래, `print("  [완료] 가계금융복지조사 집계 3종")` 라인 바로 위에
다음 집계 블록 전체를 추가한다:

```python
    # ── 집계 4: 다각도 교차 그루핑 매트릭스 (mean + var 동시 산출)
    # 수도권 × 주택종류 × 고령층유형 × 부채유무 교차 분석
    # 분산(var): "같은 수도권 아파트라도 자산 격차 극심" 양극화 입증용
    df_multi = panel.copy()
    df_multi['부채유무'] = np.where(df_multi['부채'] > 0, '부채있음', '부채없음')
    df_multi['수도권여부'] = df_multi['수도권여부'].fillna('미분류')
    df_multi['주택종류통합코드'] = df_multi['주택종류통합코드'].fillna('미분류')
    df_multi['부채유무'] = df_multi['부채유무'].fillna('미분류')

    group_keys_multi = ['조사연도', '연령그룹', '수도권여부', '주택종류통합코드',
                        '고령층유형', '부채유무']

    grp4 = df_multi.groupby(group_keys_multi).agg(
        표본수                  = (WT_COL,       'count'),
        추정가구수_가중치합     = (WT_COL,       'sum'),
        처분가능소득_평균       = (INC_COL,      'mean'),
        처분가능소득_분산       = (INC_COL,      'var'),
        거주주택금액_평균       = (RE_COL,       'mean'),
        거주주택금액_분산       = (RE_COL,       'var'),
        유동성_정체지수_평균    = ('유동성_정체지수_raw', 'mean'),
        유동성_정체지수_분산    = ('유동성_정체지수_raw', 'var'),
        경직적_비용비중_평균    = ('경직적_비용비중', 'mean'),
    ).reset_index()
    grp4 = clean_inf_nan(grp4)
    # 분산 컬럼 결측(표본 1건일 때 발생) → 0으로 처리
    var_cols = [c for c in grp4.columns if '분산' in c]
    grp4[var_cols] = grp4[var_cols].fillna(0).round(2)
    mean_cols = [c for c in grp4.columns if '평균' in c]
    grp4[mean_cols] = grp4[mean_cols].round(2)
    grp4 = apply_k_anonymity(grp4, '표본수', k=5)
    save_xlsx(grp4, '04_다각도_그루핑_매트릭스.xlsx')
    del df_multi
```

### [2-2] vdr_pipeline.py — load_hfws()에 유동성_정체지수_raw 추가

`load_hfws()` 함수 내에서 `경직적_비용비중` 계산 코드 바로 아래에 추가:
```python
    # 유동성_정체지수_raw: 거주주택금액 / (처분가능소득 + 1)
    # 집계 4의 분산 계산용 원시 지수 (LTI와 별도 보완 지표)
    panel['유동성_정체지수_raw'] = panel[RE_COL] / (panel[INC_COL] + 1)
    panel['유동성_정체지수_raw'] = panel['유동성_정체지수_raw'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
```

---

## 변경 3: 연관규칙 함수 교체 — VDR 전용 순수 Pandas 버전

### 배경
week3_4_feature_engineering.py의 run_association_rules()는
mlxtend(fpgrowth) 라이브러리에 의존한다.
VDR 환경에서는 외부 패키지 설치가 불가할 수 있으므로,
순수 Pandas만으로 동작하는 함수로 교체한다.

### [3-1] week3_4_feature_engineering.py — 함수 교체

파일에서 `def run_association_rules(` 함수 전체를 찾아
다음 코드로 완전히 교체한다 (함수 시작부터 return 포함 끝까지):

```python
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

    print(f"\n📌 연관 규칙 분석 결과 ({len(result_filtered)}개 규칙 통과):")
    print(result_filtered.to_string(index=False))

    return result_filtered
```

### [3-2] week3_4_feature_engineering.py — discretize_for_apriori 보완

`discretize_for_apriori()` 함수 내 `items` DataFrame 생성 코드에서
현재 사용하는 컬럼명이 `week3_4`의 독자적 컬럼명("부동산자산", "총소득" 등)으로
되어 있으므로, vdr_pipeline.py의 실제 컬럼명과 맞게 보완한다.

`discretize_for_apriori()` 함수 상단 주석 바로 다음 줄에 아래 매핑 로직을 추가한다:
```python
    # vdr_pipeline.py 컬럼명 호환: 두 네이밍 모두 지원
    re_col  = RE_COL  if RE_COL  in df.columns else '부동산자산'
    inc_col = INC_COL if INC_COL in df.columns else '총소득'
    med_col = '지출_소비지출_의료비' if '지출_소비지출_의료비' in df.columns else '의료비지출'
    age_col = '가구주_만연령' if '가구주_만연령' in df.columns else '가구주_연령'
```

그리고 같은 함수 내 `q60_re`, `q30_in`, `q25_med` 계산을 다음으로 교체:
```python
    q60_re   = df[re_col].quantile(0.60)
    q30_in   = df[inc_col].quantile(0.30)
    q25_med  = df[med_col].quantile(0.25) if med_col in df.columns else 0
```

그리고 `items` DataFrame 내 컬럼 참조를 교체:
```python
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
```

---

## 수정 금지 항목

다음은 절대 건드리지 않는다:
- VDR 반출 규정 함수 3종 (clean_inf_nan, apply_k_anonymity, save_xlsx)
- 메모리 최적화 로직 (chunksize, gc.collect, dtype 사전 지정)
- METRO_SIDO 수도권 필터 (이전 작업에서 적용된 경우 유지)
- K-Anonymity 임계값 k=5
- 파일 경로 변수 전체
- '직업 구분', '월 연체보유자수 합계' 공백 포함 항목명

---

## 수정 완료 후 검증

수정이 끝나면 다음 코드를 실행하고 결과를 출력한다:

```python
import ast

results = {}

# vdr_pipeline.py 검증
with open('vdr_pipeline.py', 'r', encoding='utf-8') as f:
    src_vdr = f.read()

results['변경1_동적빈곤선_상수']      = 'MEDIAN_INCOME_BY_YEAR' in src_vdr
results['변경1_get_dynamic함수']      = 'def get_dynamic_poverty_line' in src_vdr
results['변경1_load_hfws_적용']       = 'get_dynamic_poverty_line' in src_vdr and \
                                         'POVERTY_LINE_2026' not in src_vdr
results['변경1_경직적비용비중']       = '경직적_비용비중' in src_vdr
results['변경2_유동성정체지수_raw']   = '유동성_정체지수_raw' in src_vdr
results['변경2_분산집계_var']         = "'var'" in src_vdr
results['변경2_04파일저장']           = '04_다각도_그루핑_매트릭스.xlsx' in src_vdr
results['수정금지_clean_inf_nan']     = 'def clean_inf_nan' in src_vdr
results['수정금지_k5유지']            = 'k=5' in src_vdr
results['문법오류없음_vdr']           = True

try:
    ast.parse(src_vdr)
except SyntaxError as e:
    results['문법오류없음_vdr'] = False
    print(f"vdr_pipeline.py 문법 오류: {e}")

# week3_4 검증
with open('week3_4_feature_engineering.py', 'r', encoding='utf-8') as f:
    src_w34 = f.read()

results['변경3_mlxtend제거']          = 'from mlxtend' not in src_w34
results['변경3_VDR연관규칙함수']      = 'K-Anonymity' in src_w34 and \
                                         'def run_association_rules' in src_w34
results['변경3_컬럼명_호환']          = 're_col' in src_w34 and 'inc_col' in src_w34
results['문법오류없음_w34']           = True

try:
    ast.parse(src_w34)
except SyntaxError as e:
    results['문법오류없음_w34'] = False
    print(f"week3_4_feature_engineering.py 문법 오류: {e}")

print("\n=== 3가지 개선사항 적용 검증 결과 ===")
all_pass = True
for k, v in results.items():
    icon = '✓' if v else '✗'
    if not v:
        all_pass = False
    print(f"  {icon} {k}")

print(f"\n{'전체 통과 ✓' if all_pass else '실패 항목 수정 필요 ✗'}")
```
