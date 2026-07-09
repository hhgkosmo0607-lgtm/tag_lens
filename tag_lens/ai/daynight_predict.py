from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from opencv.feature_analyzer import is_bw_image

CLASS_NAMES = ["주간", "야간"]

DAY_START_HOUR = 6   # 06:00
DAY_END_HOUR = 18    # 18:00 (미만)


def get_exif_datetime_and_offsets(image_path: str) -> tuple[str | None, str | None, str | None]:
    """
    EXIF에서 촬영 시각 문자열과 관련 시간대 오프셋 태그를 함께 읽는다.
    반환: (datetime_str, offset_time_original, offset_time)

    DateTimeOriginal(36867)은 최상위 IFD0가 아니라 **Exif SubIFD**(태그 34665로
    가리키는 중첩 구조)에 들어있다. image.getexif()는 IFD0만 반환하므로 거기서
    36867을 찾으면 항상 실패하고, 최상위의 DateTime(306)으로 폴백하게 되는데
    — 이 306은 Lightroom/포토샵 같은 편집 프로그램이 "내보내기한 시각"으로
    덮어써버리는 경우가 많다(실제 촬영 시각이 아님). 그래서 SubIFD를 먼저 봐야 한다.

    우선순위: SubIFD의 DateTimeOriginal → SubIFD의 DateTimeDigitized → IFD0의 DateTime
    app.py(업로드)와 reclassify_photos.py(재분류)가 공유하는 단일 진입점.
    """
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()

        try:
            from PIL.ExifTags import IFD

            exif_ifd = exif_data.get_ifd(IFD.Exif)
        except Exception:
            exif_ifd = {}

        offset_original = exif_ifd.get(36881)  # OffsetTimeOriginal
        offset_time = exif_ifd.get(36880)      # OffsetTime

        if 36867 in exif_ifd:  # DateTimeOriginal (SubIFD)
            return exif_ifd[36867], offset_original, offset_time
        if 36868 in exif_ifd:  # DateTimeDigitized (SubIFD)
            return exif_ifd[36868], offset_original, offset_time
        if 306 in exif_data:  # DateTime (IFD0) — 편집 프로그램이 덮어썼을 수 있는 최후 폴백
            return exif_data[306], offset_original, offset_time
    except Exception:
        pass

    return None, None, None


def _parse_offset_hours(offset_str: str | None) -> float | None:
    """'+08:00' 같은 EXIF 시간대 오프셋 문자열을 시간 단위 실수로 변환한다. 파싱 실패 시 None."""
    if not offset_str:
        return None
    try:
        sign = -1 if offset_str[0] == "-" else 1
        hh, mm = offset_str.lstrip("+-").split(":")
        return sign * (int(hh) + int(mm) / 60)
    except Exception:
        return None


def extract_exif_hour(image_path: str) -> tuple[int | None, int | None]:
    """
    EXIF 촬영 시각(0~23시)을 추출한다. 반환: (보정된 시각, 보정 전 원본 시각).
    EXIF가 없으면 (None, None). 보정이 적용되지 않았으면 두 값이 같다.

    raw_hour을 따로 반환하는 이유: `resolve_daynight()`가 "보정 전 시각이 원래
    경계 근처였는지"도 함께 판단해야 하기 때문이다(아래 실측 사례 참고 — 보정이
    시각을 경계에서 멀리 옮겨놔도, 애초에 경계 근처였던 값을 보정한 거라 여전히
    불확실할 수 있음).

    카메라 시계 오류 보정: `OffsetTimeOriginal`이 "+00:00"(카메라가 시간대를 모를 때
    흔히 쓰는 기본값)인데 `OffsetTime`은 다른 실제 시간대를 가리키는 경우,
    DateTimeOriginal이 로컬 시각이 아니라 시간대 보정 없이 그대로 기록된 것으로 보고
    OffsetTime 값만큼 보정한다.

    실측(uploads/ 26장 중 12장이 이 패턴): 대부분 대낮 사진인데 00~06시대로 잘못
    기록돼 있었고, OffsetTime 오프셋(+08:00)만큼 보정하면 실제 촬영 상황(맑은 대낮)과
    일치함을 육안으로 확인함. 단, 이 패턴 중 하나(원본 06:17, 경계 06시 바로 근처)는
    보정 후(14:17)에도 실제로는 명백한 야간 사진이었음 — 시간대가 아니라 카메라 시계
    자체가 신뢰할 수 없었던 사례로, `resolve_daynight()`의 경계 안전장치가 이런 경우를
    잡아낸다.

    단, 두 오프셋이 다르기만 하면 보정하는 건 안 된다 — 이 프로젝트 사진 중
    "+08:00 vs +09:00"처럼 둘 다 0이 아닌 실제 시간대끼리 다른 경우도 흔한데, 이건
    여행/편집으로 인한 정상적 차이라 DateTimeOriginal 자체가 이미 촬영지 로컬 시각임
    (보정하면 오히려 틀어짐). 그래서 "OffsetTimeOriginal == +00:00"이라는 구체적
    신호가 있을 때만 보정 대상으로 좁힘.
    """
    date_str, offset_original, offset_time = get_exif_datetime_and_offsets(image_path)
    if not date_str:
        return None, None

    try:
        raw_hour = int(date_str.split(" ")[1].split(":")[0])
    except Exception:
        return None, None

    hour = raw_hour
    if offset_original == "+00:00" and offset_time and offset_time != "+00:00":
        correction = _parse_offset_hours(offset_time)
        if correction is not None:
            hour = int(round(hour + correction)) % 24

    return hour, raw_hour


def daynight_from_hour(hour: int) -> tuple[str, dict[str, float]]:
    """EXIF 촬영 시각(0~23시)으로 주간/야간을 확정 판별한다.
    픽셀 밝기를 추정하는 CNN/휴리스틱보다 신뢰도가 훨씬 높은 실측 신호다."""
    label = "주간" if DAY_START_HOUR <= hour < DAY_END_HOUR else "야간"
    return label, {"주간": 1.0 if label == "주간" else 0.0, "야간": 1.0 if label == "야간" else 0.0}


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.percentile(gray, 50))
    night_score = max(0.0, min(1.0, (120 - brightness) / 120))
    probs = {"주간": 1.0 - night_score, "야간": night_score}
    label = max(probs, key=probs.get)
    return label, probs


# 2026-07-09: 자체 학습 CNN(도로/도시 웹캠 데이터로 학습)을 CLIP 제로샷으로 교체함.
#
# 경위: `자연` 장르 사진의 EXIF vs CNN 불일치를 조사하다가, 그림자 짙은 숲 캐노피
# (하늘 안 보임)뿐 아니라 특이한 색보정(보라빛 하늘) 사진에서도 CNN이 야간으로
# 오판하는 걸 확인함 — 공통 원인은 CNN의 학습 데이터(CCTV 도로 400장 + DNIM 웹캠
# 17개 지점)가 전부 도시/도로 장면이라 자연광/그늘진 자연 사진 도메인을 아예 학습한
# 적이 없다는 것. 같은 실패 사진 5장으로 CLIP 제로샷("a photo taken during the
# daytime" vs "a photo taken at night")을 테스트했더니 4/5 정확(CNN은 5/5 전부 오답)
# 해서 교체함. ai/indoor_predict.py에 CLIP을 먼저 적용해 검증된 것과 같은 패턴.
# 모델 로더는 ai/clip_utils.py 공유(indoor_predict.py, animal_predict.py와 같은 CLIP
# 인스턴스를 재사용해 중복 로드 방지).
_DAY_PROMPTS = ["a photo taken during the daytime", "a bright photo taken in daylight"]
_NIGHT_PROMPTS = ["a photo taken at night", "a dark photo taken at nighttime"]


def predict_daynight(image, model_path: str | None = None) -> tuple[str | None, dict[str, float] | None]:
    """주간/야간을 판별한다. 흑백 사진은 색온도·밝기 신호가 왜곡되므로 판별하지 않고 (None, None)을 반환한다.

    model_path는 기존 호출부와의 인터페이스 호환을 위해 남겨뒀지만 CLIP은 로컬 .h5
    경로가 필요 없어 쓰지 않는다.
    """
    if is_bw_image(image):
        return None, None

    try:
        import torch

        from ai.clip_utils import get_clip, image_to_pil

        model, processor = get_clip()
        pil_image = image_to_pil(image)

        prompts = _DAY_PROMPTS + _NIGHT_PROMPTS
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            scores = model(**inputs).logits_per_image.softmax(dim=1)[0]

        day_score = float(scores[: len(_DAY_PROMPTS)].sum())
        night_score = float(scores[len(_DAY_PROMPTS) :].sum())
        probs = {"주간": day_score, "야간": night_score}
        label = max(probs, key=probs.get)
        return label, probs
    except Exception as e:
        print(f"[ERROR] Failed to predict daynight via CLIP: {type(e).__name__}: {e}")
        return _heuristic_predict(image)


DAYNIGHT_BOUNDARY_MARGIN = 1  # 시간 단위 — 이 안쪽이면 "경계 근처"로 보고 CNN 교차검증


def resolve_daynight(
    image, exif_hour: int | None, model_path: str | None = None, raw_exif_hour: int | None = None
) -> tuple[str | None, dict[str, float] | None]:
    """
    주간/야간 최종 판정 — app.py(업로드)와 reclassify_photos.py(재분류)가 공유하는 단일 진입점.
    흑백 사진은 판별하지 않는다(색/밝기 신호가 왜곡되어 (None, None) 반환).

    EXIF 촬영시각이 있으면 그대로 확정 판정(daynight_from_hour), 없으면 CLIP 제로샷
    (실패 시 밝기 휴리스틱)으로 폴백한다.

    [설계 보류] EXIF·CNN·밝기 휴리스틱 세 신호를 다수결로 종합하는 방식을 시도했었으나,
    (당시) CNN과 밝기 휴리스틱이 둘 다 픽셀 밝기에 의존해 독립적이지 않다는 게 실측으로
    확인됨(그림자 짙은 대낮 숲 사진 250장 중 118장에서 CNN이 야간으로 오판했고, 그중
    다수는 휴리스틱도 똑같이 야간으로 오판 — 이 경우 다수결이 2:1로 정확한 EXIF를
    뒤집어버림). 카메라 시계 오류(EXIF가 실제와 다른 극소수 사례)보다 이 역효과가 더
    크다고 판단해 일단 EXIF 우선 방식으로 되돌려둠. (2026-07-09 추가: 이후 자체 학습
    CNN을 CLIP으로 교체했지만, 다수결 자체를 재검토하진 않음 — 아래 경계 안전장치로
    범위를 좁혀 쓰는 지금 방식을 유지)

    [카메라 시계 오류 감지 추가] `extract_exif_hour()`가 EXIF의 `OffsetTimeOriginal`이
    "+00:00"(시간대 미설정 기본값)인데 `OffsetTime`은 다른 실제 시간대를 가리키는
    경우를 감지해 이미 보정한 시각을 넘겨준다 — 이 함수는 이미 보정된 exif_hour를
    그대로 확정 판정에 쓰면 된다. 실측(uploads/ 26장 중 12장)으로 이 패턴 전부가
    실제로는 대낮인데 00~06시대로 잘못 기록된 것이었고 보정 시 정확히 맞아떨어짐을
    확인함(자세한 내용은 extract_exif_hour 문서 참고).

    [경계 안전장치 추가] EXIF 시각(보정 여부 무관)이 주간/야간 경계(06시, 18시)에서
    `DAYNIGHT_BOUNDARY_MARGIN`시간 이내면 CLIP으로 교차검증해서, CLIP이 70% 이상
    확신도로 다른 판정을 내리면 CLIP을 따른다(2026-07-09 이전엔 CNN이었음 — 아래
    predict_daynight()가 CLIP 제로샷으로 교체됨). 카메라 시계가 정확한 시간대를
    기록해도 시각 자체가(분 단위로) 살짝 어긋나 있으면 경계에서 판정이 뒤집힐 수 있기 때문.

    실측: `OffsetTimeOriginal "+00:00"` 보정을 거친 사진 중 하나(DTO 06:17:29, 실제로는
    명백한 야경/수면 반사 사진)가 경계(06시)에 걸려 EXIF만으로는 "주간"으로 오판됐는데,
    CNN(당시)은 87.6% 확신도로 정확히 "야간"이라고 판정함 — 이 케이스를 잡기 위해 추가.

    경계에서 먼 EXIF 시각(예: 낮 2시, 밤 11시)은 이 교차검증을 타지 않고 그대로 확정
    판정한다 — 범위를 넓히면 위에서 되돌렸던 3중 다수결의 역효과(그림자 짙은 대낮 숲
    사진에서 CNN이 정확한 EXIF를 뒤집는 문제)가 재발하기 때문에 경계 근처로만 좁혔다.

    `raw_exif_hour`(보정 전 원본 시각, `extract_exif_hour()`가 함께 반환)도 경계
    근처인지 같이 확인한다 — 보정이 시각을 경계에서 멀리 옮겨놔도(06시→14시처럼)
    애초에 경계 근처였던 원본을 보정한 거라 여전히 불확실하기 때문(위 06:17 사례가
    정확히 이 경우: 보정 후 14시라 경계에서 멀어 보이지만, 원본 06:17이 경계 코앞이라
    보정 자체가 신뢰도 낮은 상황이었음).
    """
    if is_bw_image(image):
        return None, None

    if exif_hour is not None:
        label, probs = daynight_from_hour(exif_hour)

        def _near_boundary(hour: int) -> bool:
            return (
                abs(hour - DAY_START_HOUR) <= DAYNIGHT_BOUNDARY_MARGIN
                or abs(hour - DAY_END_HOUR) <= DAYNIGHT_BOUNDARY_MARGIN
            )

        near_boundary = _near_boundary(exif_hour) or (raw_exif_hour is not None and _near_boundary(raw_exif_hour))
        if near_boundary:
            cnn_label, cnn_probs = predict_daynight(image, model_path)
            if cnn_probs is not None and cnn_label != label and cnn_probs[cnn_label] > 0.7:
                return cnn_label, cnn_probs
        return label, probs

    return predict_daynight(image, model_path)
