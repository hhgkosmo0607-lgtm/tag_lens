from __future__ import annotations


def generate_tags(
    genre: str,
    daynight: str | None,
    is_bw: bool = False,
    indoor: str | None = None,
    color_tone: str | None = None,
    animal: str | None = None,
) -> list[str]:
    """
    장르 태그 + 주간/야간 태그 + 흑백 태그 + 실내 태그 + 색온도 태그 + 동물 태그를 생성한다.
    app.py(업로드)와 reclassify_photos.py(재분류)가 공유하는 단일 소스.
    daynight가 None이면(흑백 사진 등 판별 불가) 주간/야간 태그를 붙이지 않는다.
    indoor는 "실내"일 때만 태그를 붙인다 — "실외"는 이미 자연/도시 장르 태그가
    커버하므로 붙이지 않고(중복 정보), indoor가 None이면(판별 대상 장르가 아님) 역시 안 붙인다.
    흑백 사진은 색 정보 자체가 없으므로 color_tone(따뜻한/차가운) 태그를 붙이지 않는다.
    animal은 "강아지"/"고양이"/"말"/"동물" 중 하나거나 None(동물 아님/확신도 낮음).
    """
    tags: set[str] = set()

    genre_map = {
        "자연": ["#자연"],
        "도시": ["#도시"],
        "음식": ["#음식"],
        "사물": ["#사물"],
        "인물": ["#인물"],
        "기타": ["#기타"],
    }
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
