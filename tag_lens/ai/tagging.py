from __future__ import annotations


def generate_tags(genre: str, daynight: str) -> list[str]:
    """
    장르 태그 + 주간/야간 태그를 생성한다.
    app.py(업로드)와 reclassify_photos.py(재분류)가 공유하는 단일 소스.
    """
    tags: set[str] = set()

    genre_map = {
        "풍경": ["#풍경"],
        "도시": ["#도시"],
        "음식": ["#음식"],
        "사물": ["#사물"],
        "인물": ["#인물"],
    }
    tags.update(genre_map.get(genre, []))

    tags.add(f"#{daynight}")

    if genre == "도시" and daynight == "야간":
        tags.add("#야경")

    return sorted(tags)
