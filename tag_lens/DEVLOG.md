# TAG_LENS 개발일지

세션별 작업 로그. 프로젝트 개요/구조는 [README.md](README.md) 참고.

## 2026-07-08

### 완료

- **장르 CNN 5-class 확장**: 사물(1066장)·인물(2000장) 데이터를 새로 확보해 자연 2000 / 도시 938 / 음식 2000 / 사물 1066 / 인물 2000, train:val = 8:2로 데이터셋 구성
- **얼굴 검출기 mediapipe로 교체**: `opencv/face_detect.py`가 Haar Cascade → mediapipe(BlazeFace) 기반으로 교체됨
- **주간/야간 이진 분류기 추가**: `models/daynight_model.h5`, 검증 정확도 98.75%로 학습 완료
- **코드 공용화**: `ai/tagging.py`로 태그 생성 로직을 `app.py`/`scripts/reclassify_photos.py`가 공유하도록 정리, `database/database.py`에 `replace_tags()`/`update_classification()` 추가
- **파일 구조 정리**: 루트에 흩어져 있던 학습/유지보수 스크립트 6개(`prepare_genre_dataset.py`, `prepare_daynight_dataset.py`, `train_daynight_model.py`, `reclassify_photos.py`, `migrate_embeddings.py`, `_test_kaggle.py`)를 `scripts/`로 이동하고 경로 참조 수정
- **`.gitignore` 보강**: 실제 git 루트(`Python_project/`)의 `.gitignore`에 `__pycache__/`, `*.pyc`, `*.log`를 추가하고, 이미 커밋돼 있던 캐시 파일 11개를 추적 해제
- **`genre_model.h5` 2단계 fine-tuning 완료**
  - 1단계(base 동결, 15 epoch): val_accuracy 99.72%, val_loss 0.0125
  - 2단계(상위 30레이어 해동, `EarlyStopping patience=7`): epoch 14에서 조기 종료, 최고 성능이던 **epoch 7 가중치로 복원**(val_accuracy 99.72%, val_loss 0.0126) — fine-tuning으로 큰 개선은 없었지만 성능 저하 없이 안전하게 마무리됨
  - `train_genre_model.py`/`train_genre_model_phase2.py`도 학습 종료 후 `scripts/`로 이동 완료
- **얼굴 검출 파이프라인 복구**: 두 가지 문제가 겹쳐서 얼굴 검출이 완전히 죽어있던 걸 확인 후 수정
  - **mediapipe Solutions API 소실**: 설치된 mediapipe(0.10.35, Python 3.12 Windows 빌드)에 구버전 `mp.solutions.face_detection`이 아예 빠져있어서 (`tasks/`만 존재) `opencv/face_detect.py`가 임포트 시점에 죽고 있었음 → 신버전 **Tasks API**(`mediapipe.tasks.python.vision.FaceDetector`)로 재작성하고, `models/blaze_face_full_range.tflite` 모델 파일을 받아서 사용하도록 변경
  - **opencv-python 설치 자체가 손상됨**: `site-packages/cv2/`에 실제 모듈 파일(`__init__.py`, 컴파일된 `.pyd`)이 없이 DLL 하나만 남아있어 `cv2.imread` 등 모든 cv2 함수가 죽어있었음(얼굴 검출뿐 아니라 업로드 파이프라인 전체가 영향받는 문제였음) → `pip install --force-reinstall opencv-python==4.10.0.84`로 복구, 이 과정에서 딸려온 `numpy 2.5.1`(요구사항 `numpy<2.0`, tensorflow `numpy<2.1` 위반)도 `numpy==1.26.4`로 다시 고정
  - 얼굴 있는 사진/없는 사진 양쪽으로 검출 결과 확인 완료
- **CNN 분류 확률을 DB에 저장**: 기존엔 업로드 직후 `session`에만 임시로 담아뒀다가 result 페이지 한 번 렌더링하고 날아갔음 → `photo` 테이블에 `genre_probs`/`daynight_probs` 컬럼 추가, 업로드·재분류 시 DB에 저장, `/result/<id>`가 언제 봐도(갤러리에서 나중에 상세보기 들어가도) 동일한 확률을 보여주도록 변경
- **유사 사진 검색 임베딩 교체**: 기존엔 5-class 장르 분류기(`genre_model.h5`)의 분류 직전 레이어(128차원)를 유사도 임베딩으로 재사용했는데, 이 레이어는 fine-tuning 과정에서 "5개 장르 구분"에만 최적화되면서 같은 장르 안의 시각적 차이를 뭉개버려 무관한 사진끼리도 코사인 유사도 0.9+가 나오는 문제가 있었음(실측 확인됨) → 장르로 fine-tuning되지 않은 순수 ImageNet 사전학습 MobileNetV2(1280차원, `ai/genre_predict.py`의 `_get_feature_extractor()`)로 교체, 기존 19장 임베딩도 재계산 완료
- **흑백 사진 주야간 판별 스킵**: 흑백은 색온도/밝기 신호가 왜곡되므로 `opencv/feature_analyzer.py`의 `is_bw_image()`로 판별해 `predict_daynight()`가 `(None, None)`을 반환하도록 변경 — `ai/tagging.py`가 None이면 주야간 태그를 안 붙임
- **얼굴 검출 오탐 수정**: mediapipe full-range 모델이 노을 진 구름 텍스처를 얼굴로 착각(신뢰도 0.53~0.54)해서 사람 없는 풍경 사진이 `인물`로 잘못 분류되는 사례 발견 → `min_detection_confidence`를 0.5→0.6으로 상향(실제 얼굴은 0.64~0.91로 훨씬 높아 영향 없음 확인)
- **CNN 저확신 예측 → `기타` 분류**: 애매한 사진(예: 인물 53.6% vs 음식 40.8%)이 5개 장르 중 하나로 억지로 배정되던 문제 → `ai/genre_predict.py`에 `LOW_CONFIDENCE_THRESHOLD=0.6` 추가, 최고 확률이 이 미만이면 `기타`로 분류하고 `#기타` 태그 부여. 확신도 높은 예측(98~100%)은 영향 없음 확인
- **`풍경` → `자연` 태그 명칭 변경**: 코드 전반(`CLASS_NAMES`, 태그 매핑, `dataset/{train,val}/풍경` 폴더 등) 일괄 변경
- **`#흑백` 태그 추가**: `is_bw`는 DB 컬럼으로만 있고 태그화가 안 돼있었음 → `ai/tagging.py`에 반영
- **실내/실외 CNN 추가 착수**: HuggingFace `prithivMLmods/IndoorOutdoorNet-20K`(2만 장, Apache-2.0, 도메인 다양)를 클래스당 2500장으로 축소해 로컬 학습 진행 중. `자연`/`도시`/`인물` 장르에만 적용(`사물`/`음식`은 배경 맥락 없어 제외), `#실내`만 태그(실외는 자연/도시 태그가 이미 커버해서 생략)
- **`#따뜻한`/`#차가운` 색온도 태그 추가**: `color_tone`도 `#흑백`처럼 DB에만 있고 태그화가 안 돼있었음 → 흑백 사진은 색 정보가 왜곡되므로 색온도 태그를 붙이지 않도록 예외 처리
- **README를 README.md(개요)와 DEVLOG.md(작업 로그)로 분리**
- **장르 5-class + 주야간 CNN 재학습 (Colab GPU) 완료**: `사물` 카테고리 18개 확장(Caltech-101 캐시 재활용, 1066→2136장), 주야간 데이터셋에 5-class 이미지 밝기 기반 pseudo-label 추가(400→약 2160장)로 도메인 다양성 확보 후 Colab에서 재학습(`scripts/colab_train.ipynb`). 완료된 `genre_model.h5`/`daynight_model.h5`를 로컬 `models/`에 반영하고 검증(5-class/2-class 출력 shape 확인, 실제 사진으로 예측 확인)
  - `ImageDataGenerator`가 CPU 싱글스레드로 데이터를 로딩해 GPU 대비 병목이 있다는 걸 확인했음(에폭당 로컬과 큰 차이 없는 속도) — 다음에 다시 쓸 일이 있으면 `tf.data` 파이프라인이나 `workers` 병렬화 고려
- **실내/실외 CNN 로컬 학습 완료**: `scripts/train_indoor_model.py`, 클래스당 2500장, val_accuracy 100%(loss 0.0041). 실사용 사진으로 검증한 결과 대체로 잘 맞으나, **하늘이 안 보이는 실외 사진(건물 파사드 클로즈업, 저녁 시간대 따뜻한 실내조명이 창문으로 비치는 구도)을 실내로 오분류하는 경향 확인**(99.9% 확신도로 틀림 — 확신도 임계값으로는 못 거름). 실사용 가능한 수준이나 이 약점은 남아있음
- **`python scripts/reclassify_photos.py`로 기존 35장 재분류**: 새 genre/daynight/indoor 모델 + `자연` 태그명 + `#흑백`/`#따뜻한`/`#차가운`/`#실내` 태그 전부 반영, 5장(14.3%) 장르 변경 확인
- **텀블러/음료 사진이 `음식`으로 분류되는 사례 확인**: 재학습 후에도 동일 — 근본 원인은 데이터 부족이 아니라 **장르 정의 자체의 애매함**(내용물 든 음료잔 클로즈업이 사물 학습 데이터의 "빈 컵 스튜디오샷"보다 음식 학습 데이터의 "카페 음료 클로즈업"과 시각적으로 더 유사함) — 보류, 아래 "다음 할 일" 참고
- **업로드 응답 속도 개선**: `predict_genre()`/`predict_daynight()`/`predict_indoor()`가 매 업로드마다 `.h5` 모델을 디스크에서 새로 로드하고 있었음(요청당 1~3초씩 낭비) → `extract_features()`에 이미 적용했던 것과 같은 전역 캐시 패턴(`_MODEL_CACHE`)을 세 함수에 모두 적용. 실측: 최초 1회(모델 로드 포함) 8.58초 → 캐시 재사용 시 0.22초(약 39배). 서버 켜고 첫 사진만 느리고 이후로는 거의 즉시 처리됨
- **주간/야간 pseudo-label 오염 발견 및 수정**: 아침에 도메인 다양성 확보용으로 5-class 이미지에 밝기 기반 pseudo-label을 붙여 `dataset_daynight`를 4배 이상 늘렸는데(400→2160장), **"화면 전체 평균 밝기가 낮으면 야간"이라는 기준이 틀렸음이 확인됨** — 그림자 짙은 대낮 숲, 어두운 옷의 인물사진처럼 "낮이지만 어두운 톤"인 사진이 대량으로 `야간`으로 잘못 라벨링돼 모델이 오염됨(pseudo-label이 실라벨보다 4배 많아서 지배적이었음). 상위 퍼센타일 밝기로 바꿔봐도 고대비 장면(일부만 밝음)엔 여전히 안 먹힘 → **pseudo-label 전량 제거, 원 데이터(400장, CCTV 도로 도메인)만으로 재학습**(val_accuracy 97.53%, 이전의 의심스러운 100%보다 현실적)
- **주간/야간 도메인 다양성 재확보 (DNIM)**: 위 재학습만으로는 도메인이 다시 좁아져 자연/사물/음식 사진에 약함 → Kaggle `stevemark/daynight-dataset`(DNIM, Archive of Many Outdoor Scenes 기반, **17개의 서로 다른 웹캠 위치**)를 받아, **밝기 추정이 아니라 파일명에 박힌 실제 촬영 시각**(`20151101_072507.jpg` 등)으로 확실한 낮(08~16시)/밤(21~04시) 시간대만 골라 추가(주간 682/169, 야간 626/154). 재학습 진행 중(백그라운드)
- **실내/실외 오분류 패턴 발견**: 반사 유리 건물, 화보 촬영처럼 **반사면+블러 배경**이 있는 사진을 실내로 오판(확신도 90%대 이상)하는 경향 확인. 원인은 라벨 오염이 아니라 학습 데이터(HuggingFace IndoorOutdoorNet-20K)가 일반 캐주얼 사진 위주라 전문/예술 사진 스타일이 부족했던 것 → 기존 로컬 `도시`(Intel buildings/street, 751장, 전부 실외) 재활용 + **Places365**(MIT, 365개 장면 카테고리, 공식 indoor/outdoor 매핑 제공)에서 스트리밍으로 1000장씩(실내/실외 각각) 다양한 스타일 추가(train 2800/3576, val 700/862). 재학습 진행 중(백그라운드)
- **주간/야간 판별에 EXIF 촬영시각 우선 사용**: CNN으로 픽셀에서 낮/밤을 추정하는 것보다, 사진에 이미 있는 EXIF `DateTimeOriginal`이 훨씬 신뢰도 높은 실측 신호라는 걸 뒤늦게 깨달음 → `app.py`/`scripts/reclassify_photos.py`에 `extract_exif_hour()` 추가, EXIF 시각이 있으면 `daynight_from_hour()`(06~17시=주간, 그 외=야간)로 확정 판별하고 **EXIF가 없을 때만 CNN(`predict_daynight`)으로 폴백**. 실측: 업로드 테스트 사진 14장 전부 EXIF 있었고, CNN이 계속 틀렸던 대낮 숲 사진도 EXIF로는 정확히 "주간" 판정됨. 흑백 사진에도 적용 가능(색 정보 불필요)
- **파일 구조 재정리**:
  - `_raw_sorted/`(2.6GB, `prepare_genre_dataset.py`의 임시 중간 산출물, 스크립트 자체가 "삭제해도 됨"이라 명시) 삭제
  - 루트/`scripts/`의 `__pycache__` 정리
  - 완료된 학습 로그(`phase2_train.log`, `indoor_train.log`, `daynight_retrain.log`, `daynight_retrain2.log`, `places365_expand.log`)를 `logs/`로 이동 — 진행 중인 로그(`flask_run.log`, `indoor_retrain.log`)는 프로세스가 물고 있어 그대로 둠
  - 폐기된 `scripts/expand_daynight_dataset.py`(pseudo-label 방식) 삭제
  - **`dataset_daynight/`에 pseudo-label 잔재 16장 추가 발견 및 제거**: 이전 정리 때 `pseudo_*.jpg`만 지웠는데, Caltech-101 소스 이미지 중 `.png`/`.jpeg` 확장자인 pseudo 파일 16개가 안 지워지고 남아있었음 → 두 번째 daynight 재학습(DNIM 버전)에도 이 16장이 섞여 들어갔었음(전체의 약 1.2%, 영향은 작지만) → 완전 제거 후 최종 daynight 데이터셋은 100% 실라벨(주간 673/168, 야간 620/154)
  - 장르 스크립트 이름이 daynight/indoor랑 안 맞았음(`prepare_dataset.py`/`train_model.py`/`train_model_phase2.py`만 "genre"가 이름에 안 담김) → `prepare_genre_dataset.py`/`train_genre_model.py`/`train_genre_model_phase2.py`로 이름 통일, import·문서·`colab_train.ipynb` 참조 전부 수정
  - **`opencv/face_detect.py` → `ai/face_detect.py` 이동**: 이 파일은 이름/폴더 위치와 달리 실제로는 OpenCV가 아니라 mediapipe(구글 사전학습 모델)를 씀 — `opencv/`에는 진짜 OpenCV만 쓰는 코드(전처리, 특징 분석)만 남기고, "판단 로직"은 전부 `ai/`로 모음. 단 mediapipe는 우리가 학습시킨 게 아니라 외부 사전학습 모델이라, 같은 `ai/`여도 genre/daynight/indoor_predict.py(직접 학습)와는 다르다는 걸 모듈 docstring에 명시
  - **`ai/genre_predict.py`에서 유사도 검색용 특징 추출 로직을 `ai/embedding.py`로 분리**: 한 파일 안에 "직접 학습한 장르 CNN"(`predict_genre`)과 "순수 사전학습 MobileNetV2 그대로 씀"(`extract_features`)이 섞여 있었음 — 위 face_detect.py 구분과 같은 원칙으로, 직접 학습 여부에 따라 파일 자체를 분리. `app.py`/`scripts/migrate_embeddings.py`의 import 갱신
    - 이 과정에서 Flask 자동 재시작이 파일 수정 중간 타이밍에 걸려(구 `app.py` import + 신 `genre_predict.py`) `ImportError`로 서버가 죽은 적 있음 → import 수정 완료 후 재시작해서 복구
- **`#야경` 태그 조건에 `자연` 장르 추가**: 기존엔 `도시`+`야간`일 때만 붙었는데, `자연`(별이 보이는 산/바다 야경 등)도 야경으로 쳐야 한다는 피드백 반영 — `genre in ("도시", "자연") and daynight == "야간"`으로 변경
- **흑백 사진 주야간 스킵 규칙이 EXIF 경로에서 빠져있던 것 수정**: 원래 "흑백이면 주야간 판별 안 함" 규칙은 `predict_daynight()`(CNN) 안에만 있었는데, 오늘 추가한 EXIF 우선 판별(`daynight_from_hour()`)은 이 체크를 안 거치고 바로 태그를 붙이고 있었음 → `app.py`/`scripts/reclassify_photos.py`에서 EXIF 확인보다 먼저 `is_bw_image()`(또는 `features["is_bw"]`)를 체크해서, 흑백이면 EXIF 유무와 무관하게 무조건 `(None, None)`으로 스킵하도록 수정
- **EXIF `DateTimeOriginal`이 최상위 IFD가 아니라 Exif SubIFD에 있는 걸 놓친 버그 수정**: `image.getexif()`는 IFD0만 반환하는데, 실제 촬영시각(`DateTimeOriginal`, 36867)은 중첩된 Exif SubIFD(태그 34665가 가리킴)에 들어있음 → 우리 코드가 이걸 못 찾고 IFD0의 `DateTime`(306)으로 폴백했는데, **이 306번 태그는 Lightroom/포토샵 같은 편집 프로그램이 "내보내기 시각"으로 덮어써버리는 경우가 많아** 실제 촬영 시각과 전혀 다른 값이 됨(확인된 사례: 실제 촬영 11:26, Lightroom 내보내기 19:29 — 7시간 넘게 차이). `_get_exif_datetime_str()` 헬퍼로 통합해서 SubIFD의 `DateTimeOriginal`/`DateTimeDigitized`를 먼저 보고, 없을 때만 IFD0의 `DateTime`을 최후 폴백으로 사용하도록 수정. `extract_photo_date()`(업로드 날짜 표시)도 같은 버그였어서 같이 고쳐짐 — 편집된 사진들의 업로드 날짜도 부정확했을 것
- **`ai/genre_predict.py`가 흑백 사진에서도 정상 작동함을 검증**: 컬러→흑백 변환 테스트에서 5개 장르 전부 100% 정확 — 특정 사진(신발 매장 진열대, 91.4% 확신도로 `음식` 오분류)의 원인은 흑백 처리 버그가 아니라, 매장 진열대처럼 5개 장르 어디에도 안 맞는 구도라 CNN이 억지로 하나 고른 것(텀블러 사례와 동일 패턴) — 코드 버그 아님을 확인, 별도 조치 없이 알려진 한계로 기록
- **`scripts/` 하위 폴더 재정리**: 14개 스크립트가 한 폴더에 평평하게 쌓여있어 헷갈린다는 피드백 → `prepare/`(데이터셋 원본 구성) / `expand/`(기존 데이터셋 보강) / `train/`(CNN 학습) / `maintenance/`(DB에 이미 있는 사진에 결과 반영) 4개 하위 폴더로 분리. 한 단계 깊어진 만큼 모든 스크립트의 `BASE_DIR = Path(__file__).resolve().parent.parent` → `.parent.parent.parent`로 수정, `IO_places365.txt`도 이를 참조하는 스크립트와 같은 폴더(`expand/`)로 이동, 전부 import/경로 재검증 완료
- **실내/실외 "하늘 안 보이는 실외 사진" 오분류 — Places365/도시 보강 후에도 미해결, 여기서 중단**: 재학습 후 재검증한 결과 확신도는 낮아짐(TSINGTAO 99.9% → PIMFY 매장 외벽 74.0%)이나 여전히 실내로 오분류. Places365와 우리 도시 데이터 모두 "공간 전체가 보이는" 사진 위주라 "벽/간판에 딱 붙은 클로즈업" 구도 자체가 부족했던 것으로 보임 — 추가 데이터 보강 없이 알려진 한계로 남기고 오늘은 여기서 종료
- **동물 세부 태그(`#강아지`/`#고양이`/`#말`/`#동물`) 추가**: 새로 학습하지 않고 이미 쓰던 ImageNet 사전학습 MobileNetV2를 `include_top=True`(1000-class 그대로)로 한 번 더 로드해서 판별. `imagenet_class_index.json`을 직접 훑어서 인덱스 0~397번이 ImageNet-1k의 동물 클래스 전체(398번 abacus부터 사물/장면)임을 확인 후, 그 안에서 강아지(151~268)·고양이(281~285)만 넓은 범위라 세분화하고, 말은 별도 "horse" 클래스가 없어 밤색 말을 뜻하는 "sorrel"(339번)을 사실상 말 클래스로 사용(얼룩말 340 zebra는 별개 동물 취급). 나머지 동물 범위는 전부 `#동물`로 통합. `ai/animal_predict.py` 신규 작성(embedding.py와 같은 "순수 사전학습, 직접 학습 안 함" 패턴), `ai/tagging.py`/`app.py`/`scripts/maintenance/reclassify_photos.py`에 연결. top-1 확신도 0.3 미만이면 태그 없음. 인덱스 매핑 로직은 모델 출력을 mock해서 강아지/고양이/말/동물/비동물/저확신 6개 케이스 전부 검증 완료, 실제 파이프라인은 건물 사진(비동물)으로 정상적으로 `None` 반환하는 것까지 확인(동물 사진은 현재 dataset에 없어 직접 검증은 못함)

### 알려진 이슈

- **`tag_lens.db`가 git에 추적 중**: 실제 사진이 쌓이면 DB가 계속 바뀔 텐데, 계속 커밋할지 `.gitignore`에 추가할지 아직 결정 안 됨
- **실내/실외 CNN이 하늘 안 보이는 실외 사진(건물 벽/간판 클로즈업)을 실내로 오분류**: Places365 + 도시 데이터로 한 차례 보강했으나(위 참고) 여전히 남아있음. 확신도는 낮아졌지만(99.9%→74%) 단순 임계값으로 거르기엔 애매함 — 재시도한다면 "벽/간판에 딱 붙은 클로즈업" 구도의 실외 사진을 콕 집어 보강해야 함
- **텀블러/음료 등 "내용물 든 용기" 사진이 사물 대신 음식으로 분류됨**: 위 완료 항목 참고 — 장르 정의 자체가 애매한 경계 케이스, 데이터 추가 시 반대급부(진짜 음식 사진이 사물로 잘못 갈 위험) 있음
- **CNN 입력 정규화가 MobileNetV2 사전학습 방식과 불일치**: `train_genre_model.py`/`train_daynight_model.py`/`train_indoor_model.py`와 `opencv/preprocess.py`의 `prepare_for_cnn()`이 `rescale=1./255`(0~1 범위)를 쓰는데, `weights="imagenet"` MobileNetV2는 원래 `mobilenet_v2.preprocess_input`(-1~1 범위)로 학습됨. 오늘 새로 만든 `extract_features()`(유사도 검색용)는 이걸 맞게 썼는데 정작 분류기 학습/추론 쪽은 안 맞음. 학습·추론이 서로 일관돼 있어서 당장 오작동하는 건 아니고(치명적이지 않음), 전이학습 효과를 100% 못 뽑아내고 있을 뿐 — 재학습 없이는 못 고치는 문제라 **다음에 어차피 재학습할 일이 생기면 그때 같이 반영**하기로 보류

- **`#흑백` 판별 로직을 "전체 평균" 대신 "색 픽셀 비율"로 교체**: 기존 `is_bw_image()`는 이미지 전체의 채널 표준편차를 평균내서 8.0과 비교했는데, 이러면 국소적으로 진짜 색이 있어도(예: 금속에 반사된 색이 전체 픽셀의 4.7%) 나머지 무채색 영역에 희석돼 흑백으로 오판하는 사례 발견(기계 다이얼 매크로 사진, 반사광 색상 BGR=[2,83,134] 등 뚜렷한 색 확인됨). 반대로 "색 픽셀이 조금이라도 있으면 컬러"로 하면 JPEG 압축 노이즈만으로 진짜 흑백 사진까지 컬러로 오판하게 됨 — 실측(컬러 30장 vs 그레이스케일→JPEG 재인코딩 30장)으로 진짜 흑백은 색 픽셀 비율 0%, 실제 컬러는 대부분 60~94%로 명확히 갈리는 걸 확인하고, 픽셀별 채널표준편차 > 8인 픽셀이 전체의 2% 이상이면 컬러로 판정하는 절충 로직으로 교체. `reclassify_photos.py`로 기존 26장 재반영, 흑백→컬러로 뒤집힌 사진들 확인됨
- **주간/야간 3중 다수결(EXIF+CNN+밝기 휴리스틱) 시도했다가 되돌림**: 카메라 시계 오류로 EXIF가 틀린 사례(화보 사진, 실제 대낮인데 EXIF는 00:39로 기록됨 — OffsetTimeOriginal +00:00 vs OffsetTime +08:00 불일치로 시간대 오설정 추정)를 발견하고 다수결로 보완하려 시도했으나, **CNN과 밝기 휴리스틱이 둘 다 픽셀 밝기 기반이라 독립적이지 않다는 게 실측으로 확인됨**(그림자 짙은 대낮 숲 사진 250장 중 118장(47%)에서 CNN이 야간으로 오판, 그중 다수는 휴리스틱도 동일하게 오판 — 이 경우 다수결이 2:1로 정확한 EXIF를 뒤집어버림). 카메라 시계 오류는 극소수 사례인 반면 이 역효과가 더 크다고 판단해 원래 EXIF 우선 방식으로 복구(`ai/daynight_predict.py`의 `resolve_daynight()`로 app.py/reclassify_photos.py 공유 로직은 통합해둠 — 흑백 스킵 체크가 EXIF 분기에서 빠질 뻔한 것도 이 통합 과정에서 재확인). 재설계는 보류(아래 "다음 할 일" 참고)
- **daynight CNN이 숲/자연 사진에 광범위하게 약하다는 것 신규 발견**: 위 다수결 검증 과정에서 발견 — `dataset/train/자연`의 숲 사진 250장 중 47%가 CNN에 의해 야간으로 오판됨(23장은 85% 이상 확신도로 자신있게 틀림, 최대 99.5%). 기존엔 "가끔 틀리는 사례 하나" 정도로만 알고 있었는데 실제로는 훨씬 광범위한 약점이었음 — EXIF 우선 로직 덕분에 EXIF 있는 사진은 영향 없지만, EXIF 없는 숲/나무 사진은 여전히 이 문제에 노출됨
- **`database.find_similar()`의 "임베딩 없으면 해시로 폴백" 로직이 실제로는 죽어있던 버그 발견 및 정리**: `app.py`가 `photo.get("embedding") or photo.get("hash")`로 해시를 유사도 검색에 넘기고 있었는데, `find_similar()`는 해시 문자열(64자리 0/1)을 JSON 정수로 파싱해버려서(예: `json.loads("110100...")`가 거대한 정수로 파싱됨) `np.array(..., dtype=float32)`가 오버플로로 `inf`가 되는 등 사실상 의도대로 동작한 적이 없었음. 지금은 모든 사진이 업로드 시점에 임베딩을 항상 갖게 돼서(오늘 이전 세션에 이미 기존 사진 전부 마이그레이션 완료) 이 폴백 경로 자체가 도달 불가능한 죽은 코드였음 — 해시 폴백 제거하고 `find_similar()`는 임베딩만 다루도록 정리, 0벡터 폴백 차원도 실제 임베딩 차원(1280)에 맞게 수정(기존엔 1024로 틀려있었음)
- **`update_classification()`이 is_bw/color_tone/brightness/saturation을 안 갱신하던 것 발견 및 수정**: `reclassify_photos.py`가 새 `is_bw_image()` 로직으로 태그는 새로 계산해서 반영하면서도, `photo` 테이블의 `is_bw`/`color_tone`/`brightness`/`saturation` 원본 컬럼은 그대로 둬서 태그와 DB 컬럼이 어긋나는 상태가 됨(지금은 이 컬럼들을 화면에 직접 보여주는 곳이 없어서 눈에 띄는 버그는 아니었음). `update_classification()`에 선택 인자로 추가해서 재분류 시 같이 갱신되도록 수정
- **전체 주석 검토**: `scripts/` 하위 13개 파일은 리팩터링 이후에도 stale 참조 없이 깨끗한 상태 확인. `reclassify_photos.py`의 모듈 docstring이 실제 갱신 항목(indoor/animal 등)을 다 담지 못하고 있던 것 발견해 수정
- **실내/실외 판별에도 저확신 스킵 로직 추가**: `predict_indoor()`가 장르 CNN(`LOW_CONFIDENCE_THRESHOLD`)과 달리 확신도 체크 없이 51% vs 49%인 애매한 경우도 무조건 실내/실외를 확정하고 있던 것 발견 → 동일한 임계값(0.6) 패턴 적용, 미만이면 `indoor=None`으로 태그 자체를 건너뜀. 검증 데이터셋(120장)에서 회귀 없음 확인(여전히 100% 정확도, 스킵 0건 — 학습 데이터와 비슷한 사진은 항상 고확신으로 나온다는 뜻). 단, 이 로직은 "애매해서 낮은 확신"만 걸러주고, 반사 유리 건물처럼 "확신 있게 틀린" 케이스(98%)는 못 거름 — 별개 문제로 남아있음
- **장르 CNN이 동물 클로즈업을 사람으로 오분류하는 문제 발견 및 보정**: 고양이 얼굴 클로즈업 사진이 96.6% 확신도로 "인물"로 분류돼 `#인물 #고양이`라는 모순된 태그 조합이 나온 사례 발견. 원인은 "인물" 학습 데이터가 전부 사람 얼굴 클로즈업(정면 응시·큰 눈·얼굴 중심 구도)이라, 같은 구도의 동물도 사람으로 일반화해버린 것(negative 예시 부재) — 사물/음식 경계 모호함(텀블러), 실내/실외(반사 유리) 문제와 같은 계열의 학습 데이터 다양성 부족. `ai/genre_predict.py`에 `resolve_genre_with_animal()` 추가: mediapipe가 사람 얼굴을 못 찾았는데(face_count==0) `animal_predict`가 동물을 감지했다면 "인물" 판정을 "기타"로 보정. "동물"을 별도 장르로 만들지 않은 이유는 genre_probs가 5-class CNN 그대로라 근거 없는 6번째 장르를 만들면 확률 표시와 모순되기 때문(`#고양이` 태그가 이미 "무엇인지" 알려주므로 `#기타 #고양이` 조합으로 충분). `app.py`/`reclassify_photos.py` 모두 animal 예측 시점을 genre 확정 직후로 옮겨서, 보정된 genre가 이후의 실내/실외 판별에도 일관되게 반영되도록 순서 조정. 실제 인물 사진(val 80장)으로 회귀 테스트해서 오탐(진짜 인물이 잘못 기타로 바뀌는 경우) 0건 확인 후 기존 DB에도 반영(고양이 사진 1장이 인물→기타로 정정됨)

### 다음 할 일

- **주간/야간 판별 재설계 필요**: EXIF 우선 방식은 카메라 시계 오류에 취약하고, CNN/휴리스틱은 그림자 짙은 대낮 사진(특히 숲)에 취약함 — 서로 다른 실패 모드라 단순 다수결로는 안 풀림. 카메라 시계 오류를 직접 감지하는 방향(예: `OffsetTimeOriginal` vs `OffsetTime` 불일치 체크)이 밝기 기반 휴리스틱보다 더 독립적인 신호일 수 있어 다음에 시도해볼 만함
- **daynight CNN의 숲/자연 사진 오분류(47%) 원인 파악 및 재학습 필요**: EXIF 없는 숲 사진에 실질적으로 영향. 학습 데이터에 나무 그늘/캐노피 도메인이 부족했을 가능성
- **동물 태그를 실제 강아지/고양이/말 사진으로 검증 필요했던 항목 — 완료**: 재분류 중 실제 업로드 사진에서 `#고양이` 태그가 정상적으로 붙는 것 확인됨
- 실내/실외: "하늘 안 보이는 실외(벽/간판 클로즈업)" 오분류 — 한 차례 보강 시도했으나 미해결(위 참고), 추가로 파고들지 이 정도 한계로 둘지 다음에 결정
- **텀블러/음료 애매 케이스(사물 vs 음식) 개선 방향 결정됨, 착수는 보류**: 원인은 `사물` 학습 데이터(Caltech-101)가 전부 흰 배경 스튜디오 제품샷이라 스타일/맥락 다양성이 부족한 것 — 카테고리를 더 늘리는 건 비효율적(장르 분류는 세분류/객체 인식과 달리 물건 종류 자체를 구분할 필요가 없음)이고, 대신 **COCO(Common Objects in Context)처럼 생활 맥락 속에서 찍힌 물건 사진**을 추가해 "이런 촬영 스타일/구도도 사물이다"를 가르치는 방향이 맞음. 실행은 다음 세션으로 보류
- 음식/사물: 재학습 후에도 여전히 안 좋으면 confusion matrix로 정확한 오분류 패턴 확인 → 추가 데이터 보강 방향 결정
- 오늘 작업 전체 git 커밋 (아직 미완료 — 43개 이상 변경 사항 누적)
- **보류**: 계층형 태그 체계 확장(예: 장르/피사체/환경/시각요소로 나눠 건물·간판·타이포그래피·미니멀·기하·색상 등 세분화). 난이도가 태그마다 완전히 다름:
  - 색상(예: 빨간색): 쉬움 — 이미 있는 색온도/채도 분석(`opencv/feature_analyzer.py`)에서 지배적 색상 추출로 확장 가능, CNN 불필요
  - 텍스트 유무/타이포그래피: 중간 — OCR/텍스트 검출기(EAST, Tesseract 등) 필요, 지금까지 쓴 분류 CNN과는 다른 종류의 모델
  - 건물/간판: 어려움 — "사진 전체 장르"를 맞추는 지금의 분류(classification)가 아니라 사진 안에서 위치까지 찾는 객체 탐지(object detection, YOLO류)가 필요. 지금 프로젝트에 없는 종류의 모델+데이터
  - 미니멀/기하: 가장 어려움 — 객관적 정답이 없는 미적 판단이라 특징 기반 추정은 신뢰도가 낮고, 제대로 하려면 사람이 라벨링한 데이터로 별도 학습 필요
  - 착수한다면 색상 → 텍스트 유무 순으로 먼저, 객체 탐지·미적 판단류는 훨씬 큰 별도 작업으로 나중에

### 다음 세션 시작 프롬프트

다음 세션 시작할 때 이 내용을 그대로 붙여넣으면 됨:

```
tag_lens 이어서 하자. DEVLOG.md의 2026-07-08 항목 읽고 시작해줘.

오늘 남은 것들:
1. 주간/야간 판별 재설계 — EXIF 우선 방식은 카메라 시계 오류에 취약하고
   CNN/밝기휴리스틱은 그림자 짙은 대낮 숲 사진에 취약해서 단순 다수결로는
   안 풀림. OffsetTimeOriginal vs OffsetTime EXIF 태그 불일치로 카메라 시계
   오류를 직접 감지하는 방향 검토해보자.
2. daynight CNN이 숲/자연 사진 47%를 야간으로 오분류하는 문제 원인 파악
   (재학습 데이터에 나무 그늘/캐노피 도메인 보강 필요할 수 있음)
3. 실내/실외 "하늘 안 보이는 실외(벽/간판 클로즈업, 반사 유리 건물)" 오분류
   — 계속 파고들지 알려진 한계로 남길지 결정
4. 텀블러/음료 사물 vs 음식 애매 케이스 — COCO 스타일 데이터 추가하는
   방향으로 결정은 됐는데 착수는 안 함
5. git 커밋 아직 안 함 — 오늘 변경사항 전부(44개 파일) 미커밋 상태로 남아있음,
   커밋 메시지/범위 상의 필요

먼저 현재 상태 확인하고 뭐부터 할지 정하자.
```

### 환경 메모

- 실제 패키지(tensorflow/opencv/kagglehub/mediapipe/datasets)는 conda 환경 **`firstvenv`**(Python 3.12)에 설치돼 있음. 프로젝트 루트의 `.venv`는 비어있는 미사용 환경이므로 혼동 주의
- 로컬 GPU 없음 — CPU 학습은 MobileNetV2 기준 에폭당 100~250초(데이터 규모에 비례). 규모가 크면 Google Colab(GPU T4)이 훨씬 빠르지만, `ImageDataGenerator` 기반 파이프라인은 데이터 로딩이 CPU 싱글스레드라 GPU 이점이 상당 부분 상쇄될 수 있음(확인됨) — 다음에 다시 쓸 일이 있으면 `workers`/`use_multiprocessing` 병렬화나 `tf.data` 파이프라인으로 교체 고려
