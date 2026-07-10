# TAG_LENS

AI 기반 스마트 갤러리. 사진을 업로드하면 자동으로 장르, 동물, 주야간, 실내/실외 등을 분석하고 태그를 붙여줍니다.

## 주요 기능

- 🏷️ **자동 태그 생성** — CLIP 제로샷 AI로 사진 자동 분류
  - 장르 (자연/도시/음식/사물/인물)
  - 주야간 (EXIF 촬영시각 우선, 카메라 시계 오류 자동 보정)
  - 실내/실외 (도시/인물 장르)
  - 동물 (강아지/고양이/말/동물)
  - 특징 (밝기, 색온도, 채도, 흑백)

- 📊 **필터링 & 검색**
  - 태그 필터 (AND 모드: 모든 태그 포함 / OR 모드: 하나 이상)
  - 평점 필터 (별⭐ UI, 토글 방식)
  - 날짜 범위 필터
  - **모든 필터 즉시 적용** (새로고침 없음)

- ⭐ **별점 관리**
  - 토글 방식 (라이트룸 스타일)
  - 상세 페이지에서 별 클릭해서 평가
  - 같은 별 다시 클릭하면 초기화

- 🔍 **유사 사진 검색** — CNN 특징 벡터 기반 코사인 유사도

- 📝 **수동 편집** — 태그 추가/삭제, 사진 삭제

---

## 설치 & 실행

### 1. 환경 설정
```bash
# conda 환경 활성화 (Python 3.12)
conda activate firstvenv

# tag_lens 디렉토리로 이동
cd tag_lens
```

### 2. Flask 서버 시작
```bash
python app.py
```

### 3. 브라우저 접속
```
http://localhost:5000
```

---

## 파이프라인 (Processing Pipeline)

사진을 업로드하면 다음 순서로 처리됩니다:

```
사진 업로드 (app.py → upload())
    ↓
[1] 이미지 로드 & 전처리
    └─ opencv/preprocess.py → load_image() (OpenCV)
    ↓
[2] 얼굴 감지
    └─ ai/face_detect.py → count_faces()
       ├─ mediapipe BlazeFace (사전학습 CNN, 검출)
       └─ CLIP 크롭 검증 (사전학습 비전-랭귀지 모델, 오탐 방지)
    ↓
[3] 장르 분류
    ├─ 얼굴 감지됨 → "인물"
    └─ 얼굴 없음 → ai/genre_predict.py (CLIP 제로샷)
       ├─ 5-class: 자연/도시/음식/사물/인물
       └─ 확신도 60% 미만 → "기타"
    ↓
[4] 동물 판별
    └─ ai/animal_predict.py (CLIP 제로샷)
       ├─ 강아지/고양이/말/동물/없음
       ├─ 종(species) gap 체크 (로짓값 기반, 오탐 방지)
       └─ mediapipe Object Detector (사전학습 CNN, 크기 체크)
    ↓
[5] 장르 보정
    └─ ai/genre_predict.py → resolve_genre_with_animal() (로직 기반)
       └─ 동물 감지되었는데 얼굴 없음 → 장르를 "기타"로 보정
    ↓
[6] 주야간 판별
    └─ ai/daynight_predict.py
       ├─ [우선순위 1] EXIF 촬영시각 분석 (카메라 시계 오류 자동 보정)
       ├─ [우선순위 2] CLIP 제로샷 폴백 (EXIF 없을 때)
       └─ 흑백 사진 → 판별 안 함
    ↓
[7] 실내/실외 판별
    └─ ai/indoor_predict.py (CLIP 제로샷, 도시/인물 장르만)
    ↓
[8] 특징 분석
    └─ opencv/feature_analyzer.py (OpenCV 기반)
       ├─ Brightness (그레이스케일 중앙값)
       ├─ Color Tone (BGR 히스토그램 가중평균)
       ├─ Saturation (HSV 채도 75 백분위수)
       └─ is_bw (픽셀별 표준편차 분석)
    ↓
[9] 이미지 해싱
    └─ hash/average_hash.py (Average Hash)
       └─ 중복 사진 감지
    ↓
[10] 특징 벡터 추출
    └─ ai/embedding.py → extract_features() (MobileNetV2 사전학습 CNN)
       └─ 1280차원 벡터 (유사도 검색용)
    ↓
[11] 태그 생성
    └─ ai/tagging.py → generate_tags() (로직 기반)
       └─ 위 모든 결과 조합해서 태그 목록 생성:
          #자연/#도시/#음식/#사물/#인물
          #주간/#야간/#야경
          #흑백
          #따뜻한/#차가운
          #실내
          #강아지/#고양이/#말/#동물
          #기타
    ↓
[12] DB 저장
    └─ database/database.py → insert_photo() (SQLite)
       └─ photo 테이블에 모든 정보 저장
    ↓
[13] 결과 표시
    └─ templates/result.html (HTML/CSS/JavaScript)
```

---

## 주요 파일 설명

### 🎯 AI 모델들 (`ai/` 디렉토리)

| 파일 | 역할 | 모델 | 입력 | 출력 |
|------|------|------|------|------|
| `genre_predict.py` | 장르 분류 + 동물 기반 보정 | CLIP | 이미지 + 동물 판별 결과 | 장르 + 확률 |
| `animal_predict.py` | 동물 판별 | CLIP | 이미지 | 동물 종류 + 로짓값 |
| `daynight_predict.py` | 주야간 판별 | CLIP + EXIF | 이미지 + 시간 정보 | 주간/야간 |
| `indoor_predict.py` | 실내/실외 판별 | CLIP | 이미지 | 실내/실외 |
| `face_detect.py` | 얼굴 감지 + 검증 | mediapipe + CLIP | 이미지 | 얼굴 개수 |
| `clip_utils.py` | CLIP 통합 관리 | CLIP | — | 모델 인스턴스 (공유) |
| `embedding.py` | 유사도 검색 | MobileNetV2 | 이미지 | 1280차원 벡터 |
| `tagging.py` | 최종 태그 생성 | — | 위 모든 결과 | 태그 목록 |

### 📷 이미지 처리 (`opencv/` 디렉토리)

| 파일 | 역할 |
|------|------|
| `preprocess.py` | 이미지 로드, 회전 보정, 크기 조정 |
| `feature_analyzer.py` | 밝기, 색온도, 채도, 흑백 분석 |

### 🔐 기타

| 파일 | 역할 |
|------|------|
| `hash/average_hash.py` | 중복 사진 감지 (Average Hash) |
| `database/database.py` | SQLite 쿼리 (photo/tag/photo_tag 테이블) |
| `app.py` | Flask 서버 + 모든 라우트 + 업로드 파이프라인 |
| `templates/` | 웹 UI (HTML) |
| `static/css/` | 스타일 (CSS) |

---

## 데이터베이스 구조

```sql
photo 테이블:
├─ id (PK)
├─ filename (파일명)
├─ upload_date (촬영 날짜, EXIF 기반)
├─ genre (자연/도시/음식/사물/인물/기타)
├─ brightness (밝기, 0~255)
├─ color_tone (색온도: 따뜻한/차가운)
├─ saturation (채도, 0~255, 75 백분위수)
├─ is_bw (흑백 여부, 0/1)
├─ rating (별점, 0~5)
├─ hash (중복 감지용)
├─ face_count (감지된 얼굴 개수)
├─ embedding (유사도 검색용 1280차원 벡터, JSON)
├─ genre_probs (CLIP 확률, JSON)
└─ daynight_probs (CLIP 확률, JSON)

tag 테이블:
├─ id (PK)
└─ name (#야경, #인물, #흑백 등)

photo_tag 테이블 (Many-to-Many):
├─ photo_id (FK)
└─ tag_id (FK)
```

---

## CLIP 제로샷 분류 원리

모든 분류는 **OpenAI CLIP** 기반입니다. CLIP은 텍스트와 이미지를 같은 임베딩 공간에 매핑합니다.

**예시: 장르 분류**
```python
# 프롬프트 (ai/genre_predict.py)
prompts = [
    "a photo of nature landscape",
    "a photo of urban city",
    "a photo of food",
    "a photo of object product",
    "a photo of person people"
]

# 동작
# 1. 이미지를 CLIP 인코더에 통과
# 2. 각 프롬프트를 CLIP 텍스트 인코더에 통과
# 3. 이미지 벡터와 각 프롬프트 벡터의 유사도 계산
# 4. 가장 높은 프롬프트 = 장르
```

**장점**: 재학습 없이 즉시 새로운 클래스 추가 가능  
**단점**: 프롬프트 엔지니어링 필요, 특정 사진에서는 CNN보다 약할 수 있음

---

## 웹 인터페이스

### 갤러리 페이지 (`/gallery`)
```
┌─ 왼쪽 ─────────────────┬─ 오른쪽 ──────────┐
│ Tags                   │ 평점 이상 (★★★★★) │
│ ○ 모두 포함 (AND)      │ 시작일 [____]     │
│ ○ 하나 이상 (OR)       │ 종료일 [____]     │
│                        │                  │
│ □ #야경 (5)           │ [초기화]          │
│ □ #인물 (12)          │                  │
│ □ #흑백 (3)           │                  │
│                        │                  │
│ [태그 선택 해제]       │                  │
└────────────────────────┴──────────────────┘

사진 그리드 (갤러리)
```

### 상세 페이지 (`/photo/<id>`)
- 사진 정보
  - Genre, Date, Brightness, Color Tone, Saturation
  - Rating (★ 토글식, 같은 별 재클릭 = 초기화)
  - Tags (수동 추가/삭제 가능)
  
- CLIP 라이트박스
  - 사진 클릭 시 확대
  - 배경 95% 어둡게 (집중도 향상)

- 유사 사진 검색
  - 코사인 유사도 TOP 10

---

## 개발 환경

- **Python**: 3.12
- **Framework**: Flask 2.x
- **AI 모델**: 
  - **CLIP** (openai/clip-vit-base-patch32) — 분류 (장르/동물/주야간/실내외)
  - **사전학습 CNN/신경망**:
    - BlazeFace (Google mediapipe) — 얼굴 감지
    - EfficientDet-Lite0 (mediapipe Object Detector) — 동물 크기 측정
    - MobileNetV2 (ImageNet) — 유사도 검색 임베딩
  - ~~직접 학습 CNN~~ (2026-07-09 이후 폐기)
- **이미지 처리**: OpenCV, PIL/Pillow
- **DB**: SQLite3
- **임베딩**: 1280차원

---

## 주의사항

⚠️ **Runtime 환경**
- 실제 런타임: conda 환경 `firstvenv` (Python 3.12)
- 절대 시스템 기본 `python` 사용 금지
- 항상 `conda activate firstvenv` 후 실행

⚠️ **CLIP 모델**
- 첫 실행 시 ~350MB 자동 다운로드 (인터넷 필요)
- `~/.cache/huggingface/` 에 저장됨

⚠️ **메모리 요구사항**
- CLIP + mediapipe + 각 모델 로드로 ~3GB 메모리 필요
- GPU 가속 미지원 (CPU 기반)

---

## 알려진 한계

1. **daynight의 특정 사진** — CLIP이 `04541da4`(밤바다)에서 CNN보다 약함
2. **장르의 추상 사진** — 장노출/흔들림 사진은 CLIP도 판정 어려움
3. **실내/실외의 잔여 케이스** — HAKATA 간판 같은 특수한 클로즈업은 오탐
4. **얼굴 감지의 작은 얼굴** — 배경 속 행인 얼굴은 감지되지만 필터링됨

---

## 모델 진화 과정

### Phase 1: CNN 기반 (초기)

**접근법**: 각 작업별로 직접 학습 CNN 모델 구축
- **장르**: ResNet50 Fine-tuning (5-class)
- **주야간**: 간단한 CNN (2-class)
- **실내/실외**: CNN (2-class)
- **동물**: ImageNet MobileNetV2 (상위 1 로짓값만 사용)

**한계**:
- 장르 CNN: 클로즈업/텍스처 이미지에서 도메인 외 실패 (예: 아스팔트 타이어자국을 자연으로 분류)
- 주야간 CNN: 자연광 도메인 학습 데이터 부족 (숲 캐노피, 특이 색보정)
- 실내/실외 CNN: "하늘 안 보이는 실외"를 실내로 오탐 (웹캠/건물 파사드)
- 동물: ImageNet 편향 (개 118개 품종 vs 고양이 5개 품종)

### Phase 2: CLIP 제로샷 전환 (2026-07-09)

**이유**: 재학습 없이 즉시 개선 가능 + 도메인 일반화 우수

**전환 과정**:
1. **장르** → CLIP 5-class 프롬프트
   - 결과: 클로즈업 도메인 개선 (아스팔트 타이어자국 정확화)
   
2. **주야간** → CLIP 제로샷 + EXIF 카메라 시계 오류 보정
   - 결과: 자연광 도메인 오류 대폭 감소
   
3. **실내/실외** → CLIP 제로샷 + 프롬프트 개정 (카메라 위치 명시)
   - 결과: 야간 매장 전면 오탐 해결
   
4. **동물** → CLIP 제로샷 + 종(species) gap 체크 + 크기 게이트
   - 결과: ImageNet 편향 제거, 오탐 방어층 강화

**임베딩은 사전학습 CNN 유지**: 
- MobileNetV2 (ImageNet 사전학습, 재학습 X)
- 용도: 유사 사진 검색 (1280차원 특징 벡터)
- 분류 작업은 전부 CLIP으로 전환

**실측 결과** (82장 사용자 실사진 테스트):
- 오탐 감소: 14/25 (56%) → 1/25 (4%)
- 성능 향상: 거의 모든 약점 해결 (몇 가지 CLIP 자체 오답 제외)

---

## 다음 개선 방향

- [ ] CNN+CLIP 앙상블 (특정 약점 보완, 예: daynight `04541da4`)
- [ ] 사용자 피드백 기반 부분 재학습
- [ ] 다중 언어 CLIP 프롬프트
- [ ] GPU 가속 (CUDA/ROCm)
- [ ] 배치 재분류 최적화

---

---

## CNN 파이프라인 (레거시, 현재 미사용)

> 2026-07-09 이전에 사용했던 CNN 기반 접근법. 현재는 CLIP으로 전환되었으나, 코드와 스크립트는 참고용으로 남겨둠.

### [1단계] 데이터셋 준비 (`scripts/prepare/`)

**목표**: Kaggle/HuggingFace에서 원본 다운로드 → train/val 폴더로 구성

| 작업 | 스크립트 | 데이터 규모 |
|------|---------|-----------|
| 장르 분류 데이터 | `prepare_genre_dataset.py` | train 8,000장 / val 2,000장 |
| 주야간 분류 데이터 | `prepare_daynight_dataset.py` | train 320장 / val 80장 (도로 CCTV) |
| 실내/실외 분류 데이터 | `prepare_indoor_dataset.py` | train 16,000장 / val 4,000장 |

**왜 이렇게?** Kaggle/HuggingFace에서 다운로드한 원본은 이미지만 있고, train/val 폴더 구조가 없거나 클래스별 폴더가 없어서 직접 정리 필요.

### [2단계] 데이터 보강 (`scripts/expand/`)

**목표**: 학습 데이터의 다양성을 높임 (도메인 편향 줄이기)

| 작업 | 추가 데이터 | 이유 |
|------|-----------|------|
| 사물 데이터 확대 | Caltech-101 (18개 카테고리) | 원본 음식/사물이 1000장뿐이라 부족 |
| 주야간 다양성 | DNIM (17개 웹캠, 실제 시각 라벨) | 도로 CCTV 400장만으로는 다양한 실외 환경 부족 |
| 실내/실외 다양성 | Places365 (365개 장면) | "하늘 안 보이는 실외"와 "배경 없는 실내" 패턴 학습 필요 |

**결과**: train 규모 3배 확대 (dataset_indoor의 경우 train 5000 → 19,996장)

**주의**: 초기에 5-class 이미지에 밝기 기반 pseudo-label을 붙여 주야간 데이터를 늘리려 했으나, 그림자 짙은 대낮 사진을 야간으로 잘못 라벨링하는 문제로 폐기 → **신뢰할 수 있는 실제 라벨(촬영 시각)이 있는 데이터만 사용**하는 원칙 수립.

### [3단계] 모델 학습 (`scripts/train/`)

**기본 구조**: MobileNetV2 사전학습 + 2단계 fine-tuning

**단계별 학습 전략**:

| 단계 | 방법 | 에폭 | 왜? |
|------|------|------|-----|
| **1단계 (Base 동결)** | MobileNetV2 base 레이어 고정, 상위 128차원 레이어만 학습 | ~15 | 초기엔 급격한 변화 방지, 기존 ImageNet 지식 보존 |
| **2단계 (Fine-tuning)** | 상위 30레이어 해동, 매우 낮은 learning_rate로 미세조정 | ~14 (EarlyStopping patience=7) | base 레이어도 조금씩 조정해서 도메인 적응, 과적합 방지 |

**검증 정확도** (2단계 최고):
- 장르: 99.72%
- 주야간: 98.75%
- 실내/실외: 100% (train 18,996, val 4,996)

**한계**: 
- 검증셋 정확도 97%+ 달성해도 **실사진 정확도는 40~50%** (실내/실외는 13.3%)
- "맥락 정보 부족한 클로즈업" 이미지(간판, 아스팔트, 하늘 안 보이는 실외)에서 확신도 높게 틀림
- 이 한계 → **CLIP 제로샷으로 전환** (2026-07-09)

**GPU 학습**: Colab GPU(T4)를 사용할 수 있도록 `colab_train.ipynb` 제공 (로컬 CPU는 에폭당 100~250초 소요)

### [4단계] 결과 적용 (`scripts/maintenance/`)

**작업**: 새 모델로 기존 DB 사진들 재분류

```bash
python scripts/maintenance/reclassify_photos.py
```

- 기존 DB의 모든 사진을 새 모델로 다시 분류
- 변경된 장르, 태그를 DB에 반영
- 모델 갱신 후마다 실행

---

**최종 업데이트**: git commit `1185a10` (2026-07-09)  
**개발 환경**: Python 3.12, conda `firstvenv`  
**상세 진행 과정**: [DEVLOG.md](DEVLOG.md) 참고
