# TAG_LENS

AI 기반 사진 장르 분석 및 태그 기반 스마트 갤러리 웹 서비스.

## 주요 기능

> **2026-07-09부터 장르/주야간/실내외/동물 분류가 전부 자체 학습 CNN → CLIP 제로샷으로 교체됨.**
> 이유와 실측 수치는 [DEVLOG.md](DEVLOG.md) 참고, 마이그레이션 배경은 아래 [모델](#모델) 섹션 참고.

- 사진 업로드 (다중 업로드 지원)
- 하이브리드 장르 분류
  1. mediapipe 얼굴 검출 — 얼굴이 감지되면 곧바로 `인물`
  2. 얼굴이 없으면 CLIP 제로샷 5-class (`자연 / 도시 / 음식 / 사물 / 인물`, 확신도(60%) 미만이면 `기타`)
  3. 동물 태그가 붙었는데 얼굴이 감지 안 됐다면, CNN 5-class 중 뭘 골랐든 `기타`로 보정 (`resolve_genre_with_animal` — 동물 클로즈업을 장르가 사람/음식 등으로 잘못 보는 문제 대응)
- 주간/야간 판별 (EXIF 촬영시각 우선, 없으면 CLIP 제로샷으로 폴백 — 흑백 사진은 판별 제외). EXIF의 `OffsetTimeOriginal`이 시간대 미설정 기본값(`+00:00`)인데 `OffsetTime`은 실제 시간대를 가리키면 카메라 시계 오류로 보고 보정
- 실내/실외 CLIP 제로샷 이진 분류 (도시/인물 장르에만 적용, `#실내`만 태그, 확신도(60%) 낮으면 태그 없음. 자연은 2026-07-09부터 제외 — 실내 오탐이 훨씬 흔해서 신뢰도가 낮았음)
- OpenCV 특징 분석 (밝기 / 색감 / 채도 / 흑백 여부)
- 동물 세부 태그 (`#강아지`/`#고양이`/`#말`/`#동물`) — CLIP 제로샷으로 판별. 종(species) 프롬프트 1등-2등 점수차가 좁으면(동물이 없는데 색감/구도가 막연히 "동물틱"해서 걸리는 경우) 태그 안 붙임. mediapipe Object Detector(EfficientDet-Lite0)로 실제 바운딩박스 크기까지 확인해서, 화면 대비 너무 작은(주제가 아닌 곁다리) 동물도 태그 안 붙임
- 자동 다중 태그 생성 (`#장르`, `#야간`/`#야경`, `#흑백`, `#따뜻한`/`#차가운`, `#실내`, 동물 태그, 확신도 낮으면 `#기타` 등)
- 태그 AND 필터 + 별점 필터 + 날짜 필터
- CNN 임베딩 기반 유사 사진 검색 (임베딩 없으면 Average Hash로 폴백)
- Average Hash 기반 업로드 중복 감지
- 사진 별점(0~5) 및 삭제
- 사진 상세보기에서 태그 수동 추가

## 프로젝트 구조

```text
tag_lens/
├── app.py                      Flask 앱 — 라우팅 + 업로드 파이프라인
├── requirements.txt
├── tag_lens.db                  SQLite DB (photo / tag / photo_tag)
│
├── ai/                          판단/추론 로직 (CLIP 제로샷 위주 + 일부 레거시 CNN/사전학습 모델)
│   ├── genre_predict.py         5-class 장르 분류 — CLIP 제로샷(2026-07-09, 이전엔 직접 학습 CNN). resolve_genre_with_animal()도 여기 위치
│   ├── daynight_predict.py      주간/야간 판별 — EXIF 촬영시각 우선(카메라 시계 오류 보정 포함) + CLIP 제로샷 폴백(2026-07-09, 이전엔 직접 학습 CNN)
│   ├── indoor_predict.py        실내/실외 이진 분류 — CLIP 제로샷(2026-07-09, 이전엔 직접 학습 CNN). clip_utils.py를 안 쓰고 자체 CLIP 로더를 따로 둠(의도적)
│   ├── animal_predict.py        동물 세부 태그(강아지/고양이/말/동물) — CLIP 제로샷(2026-07-09, 이전엔 ImageNet MobileNetV2). 종 프롬프트 1등-2등 gap 임계값 + mediapipe Object Detector 크기 게이트로 오탐/곁다리 동물 방어
│   ├── clip_utils.py            CLIP 모델(openai/clip-vit-base-patch32) 공유 로더 — genre/daynight/animal predict가 사용 (indoor는 제외, 위 참고)
│   ├── face_detect.py           얼굴 검출 — mediapipe(Google) 사전학습 BlazeFace + CLIP 크롭 2차 검증(원형 물체 오탐 방어, 2026-07-09)
│   ├── embedding.py             유사 사진 검색용 특징 추출 — MobileNetV2/ImageNet, 순수 사전학습(직접 학습 안 함)
│   └── tagging.py               generate_tags() — app.py와 scripts/가 공유하는 태그 생성 로직
│
├── opencv/                      순수 이미지 처리 (OpenCV만 사용, 판단 로직 없음)
│   ├── preprocess.py            이미지 로드 / CNN 입력 전처리
│   └── feature_analyzer.py      밝기/색감/채도/흑백 분석
│
├── hash/
│   └── average_hash.py          Average Hash — 중복 감지 + 임베딩 없을 때 유사도 폴백
│
├── database/
│   └── database.py              스키마 생성 + CRUD / 필터 / 유사도 쿼리
│
├── templates/, static/          Flask 뷰 템플릿 + CSS
│
├── models/
│   ├── genre_model.h5                  장르 5-class CNN (레거시, 2026-07-09부터 추론에서 미사용 — CLIP으로 대체됨)
│   ├── daynight_model.h5               주야간 이진 CNN (레거시, 위와 동일)
│   ├── indoor_model.h5                 실내/실외 이진 CNN (레거시, 위와 동일)
│   ├── blaze_face_full_range.tflite    mediapipe 얼굴 검출 사전학습 모델 (현재 사용)
│   └── efficientdet_lite0.tflite       mediapipe 객체 검출 사전학습 모델, COCO 90-class — 동물 태그 크기 게이트용(현재 사용, 2026-07-09 추가)
│
├── uploads/                                        업로드된 사진 (gitignore)
├── dataset/, dataset_daynight/, dataset_indoor/     학습용 데이터셋 (gitignore, scripts/prepare/로 생성)
│
└── scripts/                     일회성 학습·유지보수 스크립트 (앱 런타임과 무관, 수동 실행) — 역할별 하위 폴더
    ├── prepare/                     데이터셋을 처음부터 구성 (원본 다운로드 → train/val 폴더)
    │   ├── prepare_genre_dataset.py     장르 5-class 데이터셋 구성 (Kaggle 다운로드)
    │   ├── prepare_daynight_dataset.py  주야간 데이터셋 구성 (Kaggle 다운로드)
    │   └── prepare_indoor_dataset.py    실내/실외 데이터셋 구성 (HuggingFace 다운로드)
    │
    ├── expand/                      이미 있는 데이터셋에 데이터를 추가로 보강
    │   ├── expand_object_dataset.py         사물 데이터셋에 Caltech-101 카테고리 추가
    │   ├── expand_daynight_with_dnim.py     주야간 데이터셋에 DNIM(다지점 웹캠, 실제 촬영시각 라벨) 다양성 추가
    │   ├── expand_indoor_with_places365.py  실내/실외 데이터셋에 Places365 다양성 추가
    │   └── IO_places365.txt                 Places365 indoor/outdoor 공식 매핑 (위 스크립트가 참조)
    │
    ├── train/                       준비된 데이터셋으로 CNN 학습 → models/*.h5
    │   ├── train_genre_model.py         장르 CNN 학습 1단계(base 동결) + 2단계(fine-tuning) 통합
    │   ├── train_genre_model_phase2.py  2단계(fine-tuning)만 독립 재실행 — 세션을 나눠 이어갈 때 사용
    │   ├── train_daynight_model.py      주야간 CNN 학습
    │   ├── train_indoor_model.py        실내/실외 CNN 학습
    │   └── colab_train.ipynb            장르+주야간 CNN을 Colab GPU에서 학습(로컬 CPU가 느릴 때)
    │
    └── maintenance/                 이미 DB에 있는 사진들에 결과 반영
        ├── reclassify_photos.py         기존 DB 사진 재분류(장르/태그 갱신, 모델 갱신 후 실행)
        ├── migrate_embeddings.py        기존 사진에 CNN 임베딩 백필
        └── _test_kaggle.py              kagglehub 연결 테스트
```

## 이미지 분석 파이프라인 (`app.py` 업로드 처리 순서)

1. mediapipe 얼굴 검출 → 얼굴 있으면 `인물`로 확정
2. 얼굴이 없으면 CLIP 제로샷 5-class 장르 분류 (`자연/도시/음식/사물/인물`, 확신도(60%) 낮으면 `기타`)
3. CLIP 제로샷으로 동물 세부 판별 (`강아지`/`고양이`/`말`/`동물`, 종 프롬프트 1등-2등 점수차가 좁거나 확신도 낮으면 태그 없음)
4. 동물 태그가 붙었는데 얼굴 미감지 상태면, 장르를 `기타`로 보정 (`resolve_genre_with_animal`)
5. 주간/야간 판별 — EXIF 촬영시각이 있으면 확정 판정(카메라 시계 오류 보정 포함), 없으면 CLIP 제로샷으로 폴백 (흑백 사진은 판별 제외)
6. 장르가 `도시`/`인물`이면 실내/실외 CLIP 제로샷 이진 분류 (`사물`/`음식`은 배경 맥락이 없어 제외, `자연`은 실내 오탐이 흔해 2026-07-09부터 제외)
7. OpenCV 특징 분석 (밝기/색감/채도/흑백)
8. Average Hash로 업로드 중복 검사
9. CNN 임베딩 추출 → 유사 사진 검색에 사용
10. 태그 생성 (`#장르`, `#주간`/`#야간`, 조건부 `#야경`, `#흑백`, 흑백이 아니면 `#따뜻한`/`#차가운`, 실내면 `#실내`, 동물이면 `#강아지`/`#고양이`/`#말`/`#동물`)

## 실행 방법

```bash
cd tag_lens
pip install -r requirements.txt   # 또는 conda 환경 firstvenv 활성화
python app.py
```

- http://127.0.0.1:5000 접속
- 실행 시 `uploads/` 폴더와 SQLite 테이블(`photo`/`tag`/`photo_tag`)이 자동 생성됨

## 모델

### 현재 (CLIP 제로샷, 2026-07-09~)

- 장르/주야간(폴백)/실내외/동물 분류 전부 `openai/clip-vit-base-patch32`(HuggingFace `transformers`) 제로샷으로 판별 — 별도 학습 없이 텍스트 프롬프트와 이미지의 유사도만 비교
- 계기: 자체 학습 CNN들이 "맥락 정보 부족한 클로즈업/질감" 사진(간판, 아스팔트, 하늘 안 보이는 실외 등)에서 확신도 높게 틀리는 패턴이 반복 확인됨. 실내/실외는 데이터 보강 재학습까지 시도했으나 검증셋 97.49%에도 실사진 정확도가 13.3%로 급락하는 과적합이 발생해 롤백하고 CLIP으로 전환
- 각 도메인마다 확신도 임계값(genre/indoor 60%, animal은 절대 로짓 23.0 + 종 프롬프트 1등-2등 gap 2.5) 을 별도로 튜닝해 "애매하면 태그 안 붙임/기타 처리"하는 안전장치를 둠 — CLIP도 프롬프트 중 반드시 하나를 고르는 구조라 확신도 자체 필터링이 필수
- 자세한 실측 수치, 실패 사례, 임계값 산출 근거는 [DEVLOG.md](DEVLOG.md) 참고

### 레거시 (자체 학습 CNN, `models/*.h5`)

- `models/genre_model.h5`, `models/daynight_model.h5`, `models/indoor_model.h5` — 더 이상 추론에서 로드되지 않음(2026-07-09부로 CLIP 전면 교체). `predict_*` 함수들이 `model_path` 인자를 여전히 받지만 호출부 호환용으로만 남겨뒀을 뿐 내부에서 쓰지 않음
- 모두 MobileNetV2 전이학습 기반, 2단계 학습(1단계 base 동결 → 2단계 상위 30레이어 fine-tuning)으로 만들어졌던 것들 — `scripts/train/`, `scripts/prepare/`, `scripts/expand/`와 아래 데이터셋 출처 표는 이 CNN들을 학습시킬 때 썼던 기록으로 남겨둠
- CNN을 다시 쓰게 될 경우를 대비해 코드/스크립트는 삭제하지 않음. 모델 파일이 없거나 로드 실패 시 각 `predict_*`는 밝기 기반 휴리스틱으로 폴백하는 구조도 유지됨

### 공통

- 학습/추론에 쓰는 실제 패키지(tensorflow/opencv/kagglehub/mediapipe/torch/transformers)는 conda 환경 **`firstvenv`**(Python 3.12)에 설치돼 있음. 프로젝트 루트의 `.venv`는 비어있는 미사용 환경이므로 혼동 주의
- 로컬에 GPU가 없어 CNN 학습은 CPU로 진행되며, MobileNetV2 전이학습도 데이터 규모에 따라 에폭당 100~250초 소요. 데이터셋이 크면 `scripts/train/colab_train.ipynb`로 Google Colab(GPU T4)에서 학습 가능 — `dataset/`, `dataset_daynight/`를 zip으로 압축해 Drive에 올린 뒤 노트북 안내대로 진행

## 데이터셋 출처

> 아래는 레거시 CNN(`models/*.h5`, 현재 미사용) 학습에 썼던 출처 기록. 현재 쓰이는 CLIP/mediapipe/ImageNet 항목은 학습이 아니라 사전학습 가중치를 그대로 쓰는 것이라 "데이터셋"이 따로 없음.

| 용도 | 출처 |
|---|---|
| 장르 — 자연/도시 | Kaggle [`puneet6060/intel-image-classification`](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (sea/glacier/mountain/forest → 자연, buildings/street → 도시) |
| 장르 — 음식 | Kaggle [`kmader/food41`](https://www.kaggle.com/datasets/kmader/food41) (Food-101, 12개 카테고리 선택) |
| 장르 — 사물 | Kaggle [`imbikramsaha/caltech-101`](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) (Caltech-101, 34개 카테고리 선택) |
| 장르 — 인물 | Kaggle `imbikramsaha/caltech-101`(Faces/Faces_easy) + Kaggle [`ashwingupta3012/human-faces`](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) |
| 주간/야간 (기본) | Kaggle [`ibrahimalobaid/day-and-night-image`](https://www.kaggle.com/datasets/ibrahimalobaid/day-and-night-image) (400장, 도로 CCTV 1곳) |
| 주간/야간 (다양성 확장) | Kaggle [`stevemark/daynight-dataset`](https://www.kaggle.com/datasets/stevemark/daynight-dataset) (DNIM, Archive of Many Outdoor Scenes 기반 — 17개의 서로 다른 웹캠 위치, 파일명의 실제 촬영 시각으로 라벨링) |
| 실내/실외 (기본) | HuggingFace [`prithivMLmods/IndoorOutdoorNet-20K`](https://huggingface.co/datasets/prithivMLmods/IndoorOutdoorNet-20K) (Apache-2.0) |
| 실내/실외 (다양성 확장) | Kaggle `puneet6060/intel-image-classification`(도시 재활용, 전부 실외) + HuggingFace [`ljnlonoljpiljm/places365-256px`](https://huggingface.co/datasets/ljnlonoljpiljm/places365-256px) (Places365, MIT — 365개 장면 카테고리를 [공식 indoor/outdoor 매핑](https://github.com/CSAILVision/places365)으로 라벨링) |
| 얼굴 검출 모델 (현재 사용) | mediapipe BlazeFace full-range 사전학습 모델(Google MediaPipe Model Zoo) — `models/blaze_face_full_range.tflite` |
| 동물 태그 크기 게이트 (현재 사용) | mediapipe Object Detector, EfficientDet-Lite0 사전학습 모델(COCO 90-class, Google MediaPipe Model Zoo) — `models/efficientdet_lite0.tflite`. 종 분류는 CLIP이 이미 하므로 여기선 바운딩박스 크기만 참고 |
| 유사도 검색 임베딩 (현재 사용) | MobileNetV2 ImageNet 사전학습 가중치(Keras Applications, 장르로 fine-tuning 안 된 원본) |
| 장르/주야간(폴백)/실내외/동물 (현재 사용) | `openai/clip-vit-base-patch32` 사전학습 가중치(HuggingFace `transformers`), 제로샷 그대로 사용 — 자세한 프롬프트/임계값은 `ai/genre_predict.py`, `ai/daynight_predict.py`, `ai/indoor_predict.py`, `ai/animal_predict.py` 참고 |
| 동물 세부 태그 (레거시) | MobileNetV2 ImageNet 1000-class 사전학습 가중치(Keras Applications, include_top=True 그대로 사용, 학습 안 함) — CLIP으로 교체되기 전에 쓰던 방식, 견종 위주 편향 문제로 폐기 |

> **참고**: 주간/야간을 늘리려고 5-class 이미지에 밝기 기반 자동 라벨(pseudo-label)을 붙이는 방식을 한 번 시도했으나, 그림자 짙은 대낮 사진을 야간으로 잘못 라벨링하는 문제가 발견돼 폐기했다(해당 스크립트는 삭제됨). 대신 위 표의 DNIM처럼 **신뢰할 수 있는 실제 라벨(촬영 시각 등)이 있는 데이터**로만 확장하는 방향으로 정리했다. 자세한 경위는 [DEVLOG.md](DEVLOG.md) 참고.

## 개발 현황

작업 로그·알려진 이슈·다음 할 일은 [DEVLOG.md](DEVLOG.md) 참고.
