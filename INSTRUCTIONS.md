# Claude Code 프롬프트 — vdr_pipeline.py 수도권 전환

## 작업 대상 파일
`vdr_pipeline.py`

---

## 배경 (읽고 이해한 뒤 수정할 것)

이 파일은 고령층 경제 이질성 분석을 위한 VDR 데이터 파이프라인이다.
기존에는 지역 필터가 없거나 서울(시도코드 '11')만 대상으로 설계되었으나,
분석 범위를 **수도권(서울+인천+경기)**으로 통일한다.

수도권 시도코드는 다음 세 개다:
- 서울 : '11'
- 인천 : '28'
- 경기 : '41'

이 세 값을 묶어 **파일 상단에 상수로 선언**하고,
코드 전체에서 이 상수를 참조하도록 한다.

---

## 수정 규칙 (반드시 준수)

### 규칙 1 — 상수 선언 (파일 최상단 SEP = '|' 바로 아래에 추가)
```python
# 분석 범위: 수도권 (서울 11 + 인천 28 + 경기 41)
METRO_SIDO = ['11', '28', '41']
```
이후 모든 필터에서 직접 리터럴을 쓰지 말고 `METRO_SIDO` 상수를 참조한다.

---

### 규칙 2 — 수정하지 않는 것 (절대 건드리지 말 것)

다음 항목은 수도권 전환과 무관하므로 **변경 금지**:
- VDR 반출 규정 4종 (`clean_inf_nan`, `apply_k_anonymity`, `save_xlsx`, inf/NaN 처리)
- 메모리 최적화 로직 (`chunksize`, `gc.collect()`, `dtype` 사전 지정)
- 고령층 연령 필터 (`가구주_만연령 >= 65`, `AAGE >= 65`)
- 파생변수 계산 로직 (`부동산편중도`, `LTI`, 유형 분류 `_classify`)
- K-Anonymity 임계값 (`k=5`)
- 파일 경로 변수 (`DRIVE`, `RDATA`, 각 데이터 디렉터리)
- 함수 시그니처, 반환값, 파일 저장 형식

---

### 규칙 3 — 함수별 수정 명세

#### [1] `load_hfws()` — 수정 없음
`수도권여부 == 'G1'` 컬럼이 이미 수도권(서울+인천+경기)을 의미하므로
**코드를 건드리지 않는다.**
단, 함수 docstring에 아래 한 줄을 추가한다:
```
# 수도권여부 'G1' = 서울+인천+경기 → METRO_SIDO와 정확히 일치하므로 별도 필터 불필요
```

#### [2] `aggregate_hfws()` — 수정 없음
`load_hfws()`의 panel이 이미 수도권 필터링된 상태이므로 변경 없음.
단, STEP 1-A 제목 주석을 다음으로 교체:
```python
print("\n[STEP 1-A] 가계금융복지조사 집계 (수도권)")
```

#### [3] `aggregate_sgis()` — 수정 위치 1곳
`read_txt_chunked`로 가구통계등록부를 전체 로드한 직후,
`hhd['시군구코드'] = ...` 라인 **바로 다음 줄**에 수도권 필터를 추가한다.

추가할 코드:
```python
# 수도권 필터 (서울 11 + 인천 28 + 경기 41)
hhd = hhd[hhd['ADMDST_CLSF_CD'].astype(str).str[:2].isin(METRO_SIDO)].copy()
```

주택통계등록부(`HSG_2023`) 청크 처리 루프 내부,
`chunk['건물연령'] = ...` 라인 **바로 앞**에 청크 단위 수도권 필터를 추가한다:
```python
# 청크 내 수도권 필터
chunk = chunk[chunk['ADMDST_CLSF_CD'].astype(str).str[:2].isin(METRO_SIDO)].copy()
if chunk.empty:
    continue
```

STEP 제목 주석 교체:
```python
print("\n[STEP 2] SGIS 등록부 처리 (수도권)")
```

#### [4] `aggregate_card_seoul()` — 수정 위치 1곳
chunk 내부 `mask` 정의 블록에 **고객 거주지 수도권 조건을 추가**한다.

기존:
```python
mask = (
    chunk['통합카드5세단위연령코드'].isin(['14', '15']) &
    (chunk['조직구분코드'] == '1') &
    (chunk['통합카드가구형태코드'] == '5')
)
```

변경 후:
```python
mask = (
    chunk['통합카드5세단위연령코드'].isin(['14', '15']) &
    (chunk['조직구분코드'] == '1') &
    (chunk['통합카드가구형태코드'] == '5') &
    chunk['고객행정구역분류시도코드'].isin(METRO_SIDO)  # 수도권 거주자
)
```

주의: 가맹점은 서울 고정(파일 구조상 변경 불가)이므로 가맹점 시도코드 필터는 추가하지 않는다.
함수 docstring 첫 줄을 다음으로 교체:
```
내국인 국내카드소비(서울 가맹점) — 수도권 거주 65세 이상 × 시군구 × 업종 집계.
```

#### [5] `aggregate_nice_loan()` — 수정 위치 1곳
chunk 내부 `mask` 정의 블록에 광역시도코드 수도권 조건을 추가한다.

기존:
```python
mask = (
    chunk['연령구간대'].isin(['65', '70']) &
    (chunk['직업구분'] == '0') &
    (chunk['구분명'] == 'CNTY_GU')
)
```

변경 후:
```python
mask = (
    chunk['연령구간대'].isin(['65', '70']) &
    (chunk['직업구분'] == '0') &
    (chunk['구분명'] == 'CNTY_GU') &
    chunk['광역시도코드'].isin(METRO_SIDO)  # 수도권
)
```

STEP 제목 주석 교체:
```python
print("\n[STEP 5] NICE 대출·연체 월별 처리 (수도권)")
```

#### [6] `aggregate_nice_income()` — 수정 위치 1곳
chunk 내부 `mask` 정의 블록에 광역시도코드 수도권 조건을 추가한다.

기존:
```python
mask = (
    chunk['연령구간대'].isin(['65', '70']) &
    (chunk['직업 구분'] == '0') &
    (chunk['구분명'] == 'CNTY_GU')
)
```

변경 후:
```python
mask = (
    chunk['연령구간대'].isin(['65', '70']) &
    (chunk['직업 구분'] == '0') &       # 공백 포함 항목명 유지
    (chunk['구분명'] == 'CNTY_GU') &
    chunk['광역시도코드'].isin(METRO_SIDO)  # 수도권
)
```

주의: `'직업 구분'`의 공백을 절대 제거하지 않는다.

STEP 제목 주석 교체:
```python
print("\n[STEP 6] NICE 소득 월별 처리 (수도권)")
```

#### [7] `aggregate_card_apt()` — 수정 위치 1곳
chunk 내부, `sub = chunk[chunk['통합카드10세단위연령코드'] == '6'].copy()` 라인을
다음으로 교체한다:
```python
sub = chunk[
    (chunk['통합카드10세단위연령코드'] == '6') &
    chunk['고객행정구역분류시도코드'].isin(METRO_SIDO)  # 수도권 거주자
].copy()
```

STEP 제목 주석 교체:
```python
print("\n[STEP 7] 아파트단지별소비(통합카드) 처리 (수도권)")
```

#### [8] `aggregate_nh_apt()` — 수정 위치 1곳
chunk 내부, `sub = chunk[chunk['통합카드10세단위연령코드'] == '6'].copy()` 라인을
다음으로 교체한다:
```python
sub = chunk[
    (chunk['통합카드10세단위연령코드'] == '6') &
    chunk['고객행정구역분류시도코드'].isin(METRO_SIDO)  # 수도권 거주자
].copy()
```

STEP 제목 주석 교체:
```python
print("\n[STEP 8] 농협카드 아파트단지별소비 처리 (수도권)")
```

#### [9] `aggregate_pos_sgg()` — 수정 위치 1곳
청크를 `chunks` 리스트에 append하기 전,
`chunk = clean_inf_nan(chunk)` 라인 **바로 다음**에 수도권 필터를 추가한다.

추가할 코드:
```python
# 수도권 필터 (시군구코드 앞 2자리)
sido_col = next((c for c in chunk.columns if '시군구코드' in c or '시도코드' in c), None)
if sido_col:
    chunk = chunk[chunk[sido_col].astype(str).str[:2].isin(METRO_SIDO)].copy()
```

STEP 제목 주석 교체:
```python
print("\n[STEP 9] 외식업물가 처리 (수도권)")
```

#### [10] `create_visualizations()` — 제목 문자열 2곳 수정
다음 두 문자열을 찾아 교체한다:

- `'고령층 유형별 전국 추정 가구 분포 (2024)'`
  → `'고령층 유형별 수도권 추정 가구 분포 (2024)'`

- `'B유형(자산-소득 불일치) 추정 가구 추이'`
  → `'B유형(자산-소득 불일치) 수도권 추이'`

---

### 규칙 4 — 파일 상단 모듈 docstring 수정
파일 맨 위 삼중 따옴표 docstring 안에서
다음 줄을 찾아:
```
  - 가구마스터: 최적화 없이 전체 로드 (규정 준수)
```
그 바로 아래에 한 줄 추가:
```
  - 분석 범위: 수도권 (서울 11 + 인천 28 + 경기 41)
```

---

## 수정 완료 후 검증 체크리스트

수정이 끝나면 다음 항목을 직접 확인하고 결과를 출력할 것:

```python
import ast, re

with open('vdr_pipeline.py', 'r', encoding='utf-8') as f:
    src = f.read()

checks = {
    '상수_METRO_SIDO_선언':             "METRO_SIDO = ['11', '28', '41']" in src,
    'SGIS_수도권_필터':                 "ADMDST_CLSF_CD" in src and "METRO_SIDO" in src,
    '카드서울_고객시도코드_필터':        "고객행정구역분류시도코드" in src and "METRO_SIDO" in src,
    'NICE대출_광역시도코드_필터':        "광역시도코드" in src and "METRO_SIDO" in src,
    'NICE소득_광역시도코드_필터':        src.count("광역시도코드") >= 2,
    '아파트통합_고객시도코드_필터':      src.count("고객행정구역분류시도코드") >= 2,
    '농협_고객시도코드_필터':            src.count("고객행정구역분류시도코드") >= 2,
    '가계금융_미수정_확인':              "수도권여부" in src and "METRO_SIDO" not in src.split("수도권여부")[0][-50:],
    '직업구분_공백_보존':                "'직업 구분'" in src,
    '시각화_제목_수도권':                '수도권' in src and '전국' not in src.split('수도권')[1][:200],
    'K익명성_임계값_유지':               'k=5' in src,
    '문법_오류_없음':                    True,  # ast.parse로 별도 확인
}

try:
    ast.parse(src)
except SyntaxError as e:
    checks['문법_오류_없음'] = False
    print(f"문법 오류 발생: {e}")

print("\n=== 수도권 전환 검증 결과 ===")
all_pass = True
for k, v in checks.items():
    icon = '✓' if v else '✗'
    print(f"  {icon} {k}")
    if not v:
        all_pass = False

print(f"\n{'전체 통과 ✓' if all_pass else '실패 항목 확인 필요 ✗'}")
```

---

## 최종 주의사항

1. `METRO_SIDO` 상수를 선언하지 않고 `['11','28','41']`을 코드 여러 곳에 직접 쓰지 말 것
2. `'직업 구분'` (공백 포함) 항목명을 `'직업구분'`으로 바꾸지 말 것 — KeyError 발생
3. `'월 연체보유자수 합계'` (공백 포함) 항목명도 건드리지 말 것
4. `load_hfws()` 함수 본문의 `수도권여부` 관련 코드는 이미 수도권 필터이므로 수정 금지
5. 청크 필터 추가 후 반드시 `if chunk.empty: continue` 패턴을 유지하거나 추가할 것
6. 수정 범위는 `vdr_pipeline.py` 단일 파일에 한정하며 다른 파일은 건드리지 말 것
