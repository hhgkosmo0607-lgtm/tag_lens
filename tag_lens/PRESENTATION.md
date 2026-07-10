# TAG_LENS 발표 대본

시간 제한 없이, 파이프라인을 따라가면서 각 단계의 실제 코드를 바로 보여주는 구조.
프로젝트 개요/기술 문서는 [README.md](README.md), 개발 히스토리는 [DEVLOG.md](DEVLOG.md) 참고.

**진행 방식**: 화면을 발표 슬라이드 ↔ 에디터(코드) 사이를 오가며 진행. 각 섹션에 `(코드: 파일명)` 표시가 있는 곳에서 실제 파일을 열어 보여줄 것.

---

## 0. 오프닝

> "안녕하세요, 오늘 소개할 프로젝트는 **TAG_LENS**, 사진을 업로드하면 AI가 자동으로 장르·주야간·실내외·동물 같은 태그를 붙여주는 스마트 갤러리입니다. 이 프로젝트에서 제가 가장 신경 쓴 부분은 '모델 하나 돌려서 끝'이 아니라, **여러 AI 모델의 판단을 서로 검증하고 보정하는 파이프라인**을 만든 것입니다. 오늘은 다이어그램을 따라가면서, 중요한 지점마다 실제 코드를 같이 보여드리겠습니다."

**(화면: 프로젝트 홈 화면 또는 실행 데모 대기)**

---

## 1. 기술 스택 개요

| 영역 | 기술 | 역할 |
|---|---|---|
| 분류 | OpenAI CLIP (`clip-vit-base-patch32`) | 장르/동물/주야간/실내외 — 전부 제로샷 |
| 검출 | mediapipe (BlazeFace, EfficientDet-Lite0) | 얼굴 검출, 동물 크기 측정 |
| 임베딩 | MobileNetV2 (ImageNet 사전학습) | 유사 사진 검색용 1280차원 벡터 |

> "중요한 포인트 하나 — 이 프로젝트에 **직접 재학습시킨 모델이 하나도 없습니다.** 전부 사전학습된 모델을 그대로 쓰거나, CLIP처럼 프롬프트 문장으로 제로샷 판단을 시킵니다."

---

## 2. 파이프라인 전체 흐름 — 다이어그램 + 코드 워크스루

**(화면: 다이어그램 전체 표시)**

```
                              사진 업로드
                                   │
                                   ▼
                        이미지 로드 & 전처리
                       (OpenCV, EXIF 회전 보정)
                                   │
                                   ▼
                    얼굴 감지 (mediapipe BlazeFace
                       + CLIP 크롭 2차 검증)
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                  ▼
              얼굴 있음                           얼굴 없음
                  │                                  │
             장르 = 인물                    CLIP 제로샷 장르 분류
                  │                     (자연/도시/음식/사물/인물)
                  │                                  │
                  │                          확신도 60% 미만?
                  │                          ┌────┴────┐
                  │                         Yes        No
                  │                          │          │
                  │                       장르=기타   장르 확정
                  │                          └────┬─────┘
                  └───────────────┬───────────────┘
                                  ▼
                      동물 판별 (CLIP 제로샷)
                                  │
                       (4단계 게이트 — 뒤에서 전체 코드로 자세히)
                                  ▼
                장르 보정 (동물 있고 얼굴 없으면 → "기타")
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          주야간 판별                      실내/실외 판별
    (EXIF 촬영시각 우선                       (CLIP 제로샷)
     + CLIP 제로샷 폴백)
                  └───────────────┬───────────────┘
                                  ▼
                       OpenCV 특징 분석
                (밝기 / 색온도 / 채도 / 흑백 여부)
                                  │
                                  ▼
                   이미지 해싱 (Average Hash)
                      → 중복 업로드 감지
                                  │
                                  ▼
                     이미지 임베딩 추출
              (MobileNetV2, 1280차원 벡터)
                                  │
                                  ▼
                          태그 생성
                                  │
                                  ▼
                        SQLite DB 저장
                                  │
                                  ▼
                         결과 화면 출력
```

### ① 이미지 로드 & 전처리

> "사진 파일을 열어서 OpenCV로 컴퓨터가 계산할 수 있는 숫자 배열(픽셀 격자)로 바꿉니다. 스마트폰 세로 사진은 파일 안에서 회전 정보만 따로 저장되고 실제 픽셀은 가로로 누워있는 경우가 있어서, 이때 회전을 바로잡아 줍니다. AI가 아니라 순수 이미지 처리 단계입니다."

### ② 얼굴 감지 — mediapipe + CLIP 2차 검증

**(코드: `ai/face_detect.py`)**

> "**mediapipe**는 구글의 사전학습 얼굴 검출 라이브러리입니다. 카메라 앱에서 얼굴에 네모 박스가 자동으로 뜨는 것과 같은 원리입니다."

```python
_options = mp_vision.FaceDetectorOptions(base_options=_base_options, min_detection_confidence=0.65)

_FACE_CROP_PROMPTS = [
    "a close-up photo of a human face",
    "a photo of a round object, sign, button, or device that is not a face",
]
_FACE_CROP_VERIFY_THRESHOLD = 0.15

def _verify_face_crop(image, box):
    x, y, bw, bh = box
    mx, my = int(bw * 0.6), int(bh * 0.6)     # 60% 여백 — 너무 딱 맞게 자르면
    crop = image[y0:y1, x0:x1]                 # CLIP도 맥락(머리카락/목)을 못 봄
    scores = model(...).logits_per_image.softmax(dim=1)[0]
    return float(scores[0]) >= _FACE_CROP_VERIFY_THRESHOLD
```

> "mediapipe 단독으로는 벽에 붙은 화재경보기 벨(원형·대칭 패턴)을 신뢰도 0.696으로 얼굴이라 오탐했는데, 이 값이 실제 얼굴 일부(0.656~0.691)보다 오히려 높아서 임계값만 올리는 걸로는 못 갈랐습니다. 그래서 검출 박스를 잘라 **CLIP**에게 '이게 진짜 얼굴 클로즈업이냐'고 다시 물어보는 2단계 검증을 추가했습니다. 진짜 얼굴 10건은 전부 0.270 이상, 오탐은 0.049로 깨끗하게 갈렸습니다."

> "여기서 **CLIP** 개념을 짚고 가겠습니다 — 사진과 문장을 같이 넣으면 어떤 문장과 가장 잘 어울리는지 점수로 알려주는 AI입니다. 인터넷의 수억 장 '사진+설명' 쌍으로 미리 학습되어 있어서, 문장만 바꾸면 재학습 없이 새로운 기준으로 즉시 판단할 수 있습니다. 이걸 **제로샷(zero-shot)**이라 부르고, 이 프로젝트의 모든 분류 작업이 이 방식으로 동작합니다."

### ③ 장르 분류 — CLIP 5-class 제로샷

**(코드: `ai/genre_predict.py`)**

```python
_GENRE_PROMPTS = [
    "a photo of nature, such as a forest, mountain, or beach landscape",
    "a photo of a city or urban street scene",
    "a photo of food",
    "a photo of an object or product",
    "a photo of a person",
    "a photo of a crowd of people at a market or event",
]
LOW_CONFIDENCE_THRESHOLD = 0.6
...
if probs[genre] < LOW_CONFIDENCE_THRESHOLD:
    genre = "기타"
```

> "얼굴이 있으면 바로 '인물' 확정. 없으면 CLIP에게 다섯 개 문장을 주고 가장 잘 맞는 걸 고르게 합니다. 확신도가 60% 미만이면 억지로 고르지 않고 '기타'로 보냅니다 — 애매한 사진을 억지로 정답 취급하는 것보다, 모르겠으면 모른다고 하는 게 더 정직한 결과라고 판단했습니다."

> "프롬프트가 6개인 이유도 재미있는데요, '인물' 문장 하나만으로는 역광 군중/노점 사진을 못 잡아서(각 부분만 잘라 테스트하면 '인물'이 1등인데 전체 구도로는 '음식' 82.1%로 오분류) '군중' 문장을 추가해서 점수를 합산했습니다."

### ④ 동물 판별 — 미리보기

> "다음은 동물 판별인데, 이 프로젝트에서 방어 로직이 가장 두꺼운 부분이라 뒤에서 전체 코드를 펼쳐서 자세히 보여드리겠습니다. 지금은 '강아지/고양이/말/기타동물 중 하나거나 동물이 아니다'를 CLIP으로 판단한다는 것만 알아두시면 됩니다."

### ⑤ 장르 보정 — 동물 감지 결과로 장르를 되돌리기

**(코드: `ai/genre_predict.py` — `resolve_genre_with_animal`)**

```python
def resolve_genre_with_animal(genre: str, animal: str | None, face_count: int) -> str:
    if genre != "기타" and face_count == 0 and animal is not None:
        return "기타"
    return genre
```

> "동물이 감지됐는데 얼굴이 없으면 장르를 다시 '기타'로 되돌립니다. 장르 분류 AI가 원래 '사람 얼굴 클로즈업' 학습 감각을 갖고 있어서, 고양이 얼굴 클로즈업을 사람으로 착각하는 경우(실측: 96.6% 확신도로 '인물' 오판)가 있었기 때문입니다. `face_count == 0` 조건이 핵심인데, 진짜 사람 얼굴이 검출됐다면(반려동물과 함께 찍은 사진 등) 그 '인물' 판정은 근거가 있으니 보정하지 않습니다."

### ⑥ 주야간 판별 — EXIF 우선

**(코드: `ai/daynight_predict.py`)**

> "**EXIF**는 사진 파일에 같이 저장되는 보이지 않는 메타데이터입니다 — 촬영 시각, 카메라 기종 같은 정보가 들어있습니다."

```python
def daynight_from_hour(hour: int) -> tuple[str, dict[str, float]]:
    label = "주간" if DAY_START_HOUR <= hour < DAY_END_HOUR else "야간"
    return label, {...}
```

> "픽셀이 밝은지 AI로 추측하는 것보다, 사진에 이미 기록된 실제 촬영 시각을 읽는 게 훨씬 정확합니다. EXIF가 있으면 그 시각이 06~18시 사이인지만 보고 바로 확정하고, EXIF가 없을 때만 CLIP 제로샷으로 대체 판단합니다. EXIF를 신뢰할 수 없는 경계 케이스는 뒤에서 전체 코드로 다시 다루겠습니다."

### ⑦ 실내/실외 판별 — 프롬프트의 관점이 중요했던 사례

**(코드: `ai/indoor_predict.py`)**

```python
# 실패했던 버전: "장면에 실내스러운 디테일이 보이는가"에 반응
#   → 밤에 조명 켜진 매장 앞 사진이 89.2%로 "실내" 오판
# 성공한 버전: "카메라/촬영자의 위치"를 명시
_INDOOR_PROMPTS = [
    "a photo taken from inside a room or building, surrounded by walls and a ceiling",
    "the camera is indoors, inside an enclosed space",
]
_OUTDOOR_PROMPTS = [
    "a photo taken while standing outdoors on a street, sidewalk, or open area",
    "the camera is outdoors in the open air, not enclosed by walls",
]
```

> "이것도 CLIP 제로샷인데, 프롬프트 문장 하나 바꾼 게 결과를 크게 바꾼 사례입니다. '무엇이 보이는가'가 아니라 '카메라가 어디 있는가'로 질문을 바꾸자, 오탐 사진의 '실내' 확신도가 89.2% → 58.2%(임계값 미만, 태그 제거)로 떨어졌습니다."

### ⑧ OpenCV 특징 분석 / ⑨ 이미지 해싱 / ⑩ 임베딩

> "여기서부터는 AI가 아니라 전통적인 이미지 처리입니다. OpenCV로 밝기 중앙값, 색상 히스토그램(따뜻한/차가운 톤), 흑백 여부를 계산합니다. **Average Hash**는 사진을 8x8 픽셀로 축소해 밝기 패턴 지문을 만들어 중복 업로드를 잡는 알고리즘입니다. 마지막으로 **MobileNetV2**로 사진을 1280차원 벡터로 바꿔서, 비슷한 사진일수록 벡터가 가깝게 나오는 성질을 이용해 유사 사진 검색에 씁니다."

### ⑪⑫ 태그 생성 & 저장

**(코드: `ai/tagging.py` — 순수 로직, AI 아님)**

```python
def generate_tags(genre, daynight, is_bw=False, indoor=None, color_tone=None, animal=None):
    tags: set[str] = set()
    tags.update(genre_map.get(genre, []))

    if daynight is not None:
        tags.add(f"#{daynight}")
        if genre in ("도시", "자연") and daynight == "야간":
            tags.add("#야경")

    if is_bw:
        tags.add("#흑백")
    elif color_tone is not None:
        tags.add(f"#{color_tone}")

    if indoor == "실내":
        tags.add("#실내")

    if animal is not None:
        tags.add(f"#{animal}")

    return sorted(tags)
```

> "지금까지 나온 모든 AI 판단 결과를 이 함수 하나로 모아서 태그를 만듭니다. AI 모델이 바뀌어도 태그 생성 규칙은 안 바뀌고, 업로드 경로와 재분류 스크립트가 이 함수 하나를 공유합니다. SQLite에 저장한 뒤 결과 화면을 보여주면 파이프라인이 끝납니다."

---

## 3. 전체 코드 딥다이브 — 동물 판별 4중 게이트

**(화면: `ai/animal_predict.py` 전체)**

> "아까 미뤄뒀던 동물 판별을 자세히 보겠습니다. CLIP 제로샷 하나만 쓰면 오탐이 많아서, 4단계 게이트를 순서대로 추가했습니다. 각 단계는 '이전 단계를 통과한 새로운 오탐'을 막기 위해 추가된 겁니다."

```python
_ANIMAL_LOGIT_THRESHOLD = 23.0        # ① 절대 임계값
_ANIMAL_SPECIES_GAP_THRESHOLD = 2.5   # ② 종 간 로짓 차이
_MIN_ANIMAL_AREA_RATIO = 0.01         # ③ 화면 대비 크기(mediapipe)

best_idx = int(species_logits.argmax())
best_logit = float(species_logits[best_idx])
animal_presence = float(species_scores.sum())
nonanimal_presence = float(nonanimal_scores.sum())
second_logit = float(sorted(species_logits.tolist())[-2])
species_gap = best_logit - second_logit

if (
    best_logit < _ANIMAL_LOGIT_THRESHOLD
    or animal_presence <= nonanimal_presence
    or species_gap < _ANIMAL_SPECIES_GAP_THRESHOLD
):
    return None
...
if not _passes_size_gate(image):
    return None
```

```python
def _passes_size_gate(image) -> bool:
    """동물로 보이는 COCO 카테고리 박스 중 최대 면적 비율이 임계값 이상인지 확인."""
    result = _get_object_detector().detect(mp_image)
    best_ratio = 0.0
    for detection in result.detections:
        if detection.categories[0].category_name in _ANIMAL_COCO_CATEGORIES:
            box = detection.bounding_box
            best_ratio = max(best_ratio, (box.width * box.height) / (w * h))
    return best_ratio >= _MIN_ANIMAL_AREA_RATIO
```

> "순서대로 설명드리면:
> 1. **처음**: 단순히 가장 높은 확률의 클래스를 고르는 방식이었는데, 장미 사진이 '마카오앵무'로 41% 확률로 오탐됐습니다.
> 2. **동물 vs 비동물 상대 비교**를 추가했는데, 이번엔 커피잔 사진이 '새'로 96% 오탐. 이 사진은 어떤 프롬프트와도 강하게 안 맞는데(원본 로짓 17~21점대), 그나마 제일 덜 안 맞는 게 상대 비교에서 부풀려진 것이었습니다. **절대 로짓 임계값 23.0**을 추가 — 정탐 사진들의 최고 로짓은 24~28점대, 오탐은 21점을 못 넘는다는 실측 근거입니다.
> 3. 그래도 깜깜한 야경 반사 사진이 로짓 24.99로 임계값을 통과했습니다. 정탐은 1등 종의 로짓이 2등보다 확 튀는데(gap 3.68~5.68), 오탐은 개/고양이/말/새가 고만고만하게 붙어있었습니다(gap 0.80~1.56). **종 간 로짓 차이 gap 2.5**를 세 번째 게이트로 추가.
> 4. 마지막으로, 노을 사진 속 점 하나만 한 새까지 태그가 붙는 게 과했습니다. CLIP은 전역 유사도만 재서 크기 정보가 없기 때문입니다. mediapipe Object Detector로 실제 바운딩박스 크기를 재서, 화면의 1% 미만이면 태그를 안 붙입니다. 점만 한 새는 최선의 감지로도 0.047%인데 확실한 동물 사진은 4.4% 이상 — 165배 격차라 1%로 잡았습니다."

> "**한 번에 완벽한 임계값을 찾은 게 아니라, 실패 사례를 계속 모아서 새로운 신호를 발굴해나간 반복 과정**이었다는 게 이 코드가 보여주는 것입니다."

---

## 4. 전체 코드 딥다이브 — EXIF 파싱과 카메라 시계 오류 보정

**(화면: `ai/daynight_predict.py` 전체)**

```python
# ① DateTimeOriginal(36867)은 IFD0가 아니라 Exif SubIFD(태그 34665)에 있다.
exif_ifd = exif_data.get_ifd(IFD.Exif)
if 36867 in exif_ifd:
    return exif_ifd[36867], offset_original, offset_time
if 306 in exif_data:   # 최후 폴백 — Lightroom이 덮어쓴 값일 수 있음
    return exif_data[306], offset_original, offset_time
```

> "Pillow의 기본 `getexif()`는 최상위 메타데이터만 반환하는데, 실제 촬영 시각은 중첩된 구조(Exif SubIFD)에 따로 들어있습니다. 이걸 놓치면 편집 프로그램의 '내보내기 시각'을 촬영 시각으로 착각하게 됩니다 — 실제 촬영 11시 26분인데 편집 내보내기 시각 19시 29분을 쓰는, 7시간 넘게 차이 나는 사례를 발견했습니다."

```python
# ② 카메라 시계 자체가 잘못 설정된 경우 자동 보정
if offset_original == "+00:00" and offset_time and offset_time != "+00:00":
    correction = _parse_offset_hours(offset_time)
    hour = int(round(hour + correction)) % 24
```

> "카메라가 시간대를 모를 때 흔히 남기는 기본값(+00:00)과 실제 시간대를 가리키는 다른 값이 같이 있으면, 그 차이만큼 시각을 자동 보정합니다."

```python
# ③ 경계(06시/18시) ±1시간 이내면 CLIP으로 교차검증
DAYNIGHT_BOUNDARY_MARGIN = 1
near_boundary = _near_boundary(exif_hour) or _near_boundary(raw_exif_hour)
if near_boundary:
    cnn_label, cnn_probs = predict_daynight(image, model_path)
    if cnn_probs is not None and cnn_label != label and cnn_probs[cnn_label] > 0.7:
        return cnn_label, cnn_probs
```

> "가장 흥미로운 설계 결정은, 처음에 EXIF + CLIP + 밝기 휴리스틱 **3중 다수결**을 시도했다가 되돌린 겁니다. 그림자 짙은 대낮 숲 사진 250장 중 47%에서 픽셀 밝기 기반 신호(모델 예측과 밝기 휴리스틱)가 **동시에** 야간으로 오판했습니다. 두 신호가 둘 다 '어두우면 야간'이라는 같은 가정에 기대고 있어서 독립적이지 않았던 겁니다. 그래서 다수결이 오히려 정확한 EXIF를 2:1로 뒤집어버렸습니다. **교차검증 신호는 서로 독립적이어야 의미가 있다**는 걸 이때 배웠고, 지금은 EXIF가 경계 시간대에 있을 때만 좁게 CLIP과 교차검증하는 방식으로 재설계했습니다."

---

## 5. UI/UX & 검증 방식

**(화면: 갤러리/상세 페이지 데모, 가능하면 라이브 데모)**

> "결과물은 갤러리 필터(태그 AND/OR, 별점, 날짜, 선택 즉시 반영), 라이트룸 스타일 별점 토글, MobileNetV2 임베딩 기반 유사 사진 검색으로 구성됩니다."

> "그런데 사실 이 프로젝트에서 가장 중요했던 건 UI가 아니라 **검증 방식**이었습니다. 자동화된 스크립트 검증도 했지만, 실제로 대부분의 버그는 **Flask 서버를 띄우고 82장의 실사진을 직접 업로드**해보며 발견했습니다. 검증 데이터셋만으로는 못 잡는 도메인 갭 문제들이었습니다."

---

## 6. 마무리

> "정리하면 세 가지를 말씀드리고 싶습니다.
> 첫째, **단일 모델의 판단을 그대로 믿지 않는다** — CLIP도 뭔가는 항상 1등으로 뽑는 구조라, 로짓 절대값·gap·크기 같은 서로 다른 성질의 신호를 겹쳐서 검증했습니다.
> 둘째, **교차검증 신호는 독립적이어야 한다** — 같은 종류의 신호끼리 다수결을 하면 오히려 정답을 뒤집을 수 있다는 걸 실측으로 확인했습니다.
> 셋째, **버그는 실사용에서 나온다** — 오늘 보여드린 모든 방어 로직이 검증 스크립트가 아니라 실제 사진을 업로드해보며 발견한 것들입니다.
>
> 감사합니다. 질문 받겠습니다."

---

## Q&A 대비

**Q1. 동물 판별 4중 게이트를 왜 그렇게 겹쳤나요? 하나로는 안 되나요?**
> 각 게이트는 서로 다른 실패 유형을 막습니다. 절대 임계값(23.0)만 있으면 여러 종에 고르게 걸치는 오탐을 못 잡고, gap 체크만 있으면 화면 구석의 곁다리 동물을 못 잡습니다. 순서대로 하나씩 추가한 이유가 "이전 방어를 통과한 새로운 오탐이 계속 나왔기 때문"입니다.

**Q2. EXIF 파싱에서 SubIFD 문제는 정확히 뭔가요?**
> Pillow의 `image.getexif()`는 IFD0(최상위 메타데이터)만 반환합니다. 실제 촬영 시각(`DateTimeOriginal`, 태그 36867)은 중첩된 `Exif SubIFD`(태그 34665가 가리키는 구조)에 들어있어서 `get_ifd(IFD.Exif)`로 따로 열어야 합니다. 놓치면 편집 프로그램이 덮어쓴 IFD0의 `DateTime`(306)으로 폴백해 실제 촬영 시각과 몇 시간씩 차이가 날 수 있습니다.

**Q3. 3중 다수결(EXIF+CLIP+밝기 휴리스틱)을 왜 포기했나요?**
> 픽셀 밝기 기반 신호 두 개가 서로 독립적이지 않다는 게 실측으로 드러났기 때문입니다. 그림자 짙은 대낮 숲 사진 250장 중 47%에서 둘이 동시에 야간으로 오판했고, 다수결이 2:1로 정확한 EXIF를 뒤집어버렸습니다.

**Q4. CLIP 프롬프트 하나 바꾼 게 왜 그렇게 효과가 컸나요? (실내/실외 사례)**
> "실내 장면처럼 보이는가"에서 "카메라/촬영자가 실내에 있는가"로 질문의 관점을 바꿨습니다. 앞의 표현은 조명 같은 실내스러운 디테일이 화면에 보이기만 해도 반응했는데, 뒤의 표현은 카메라의 물리적 위치라는 더 명확한 기준으로 판단하게 만들었습니다.

**Q5. 재학습을 전혀 안 하는데 정확도는 어떻게 확보하나요?**
> 82장의 실사용 사진으로 전수 테스트하면서 오분류 사례를 하나씩 찾아, 프롬프트 문장 개선이나 로짓 임계값/gap 같은 후처리 로직으로 보완하는 방식입니다.

**Q6. 배포/실서비스로 확장 가능한가요?**
> 현재는 로컬 Flask 개발 서버 기준입니다. CLIP+mediapipe+MobileNetV2를 합쳐 메모리 약 3GB가 필요하고 GPU 가속은 미지원(CPU 기반)이라, 실서비스 확장 시 모델 서빙 최적화(양자화, GPU, 배치 처리)가 필요합니다.

**Q7. 가장 어려웠던 버그는?**
> 동물 판별에서, 완전히 깜깜한 야경 반사 사진처럼 진짜 동물이 없는데도 CLIP이 "그나마 동물틱한" 프롬프트를 1등으로 뽑는 문제였습니다. 로짓 절대값 임계값만으로는 정탐과 안 갈렸고, "종별 1등-2등 로짓 차이(gap)"라는 새 신호를 찾아내서 해결했습니다.

---

## 오탐 방어 계층 요약표 (질문 대응용 참고자료)

| 방어 기법 | 위치 | 막는 문제 |
|---|---|---|
| 상대 비교(동물 vs 비동물) | `animal_predict.py` | 명백히 동물이 아닌데 1등으로 뽑히는 경우 |
| 절대 로짓 임계값(23.0) | `animal_predict.py` | 상대 비교에서도 약한 신호가 부풀려지는 경우 |
| 종 간 로짓 gap(2.5) | `animal_predict.py` | 여러 종에 고만고만하게 걸치는 오탐 |
| 크기 게이트(1%, mediapipe) | `animal_predict.py` | 화면 구석의 곁다리 동물 |
| CLIP 크롭 2차 검증 | `face_detect.py` | 원형·대칭 물체를 얼굴로 오탐 |
| 카메라 시계 오류 보정 | `daynight_predict.py` | 시간대 미설정 카메라의 EXIF 오기록 |
| 경계 근처 교차검증 | `daynight_predict.py` | EXIF 자체가 틀린 극소수 케이스 |
| 저확신 스킵(0.6) | `genre_predict.py`, `indoor_predict.py` | 애매한 사진을 억지로 하나로 우기는 것 |
| 프롬프트 거부권 | `genre_predict.py` | 보강 프롬프트의 부작용(반복 인쇄물 오탐) |
| 촬영자 위치 명시 프롬프트 | `indoor_predict.py` | "장면에 보이는 디테일"과 "카메라 위치" 혼동 |
