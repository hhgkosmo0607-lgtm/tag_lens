from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "tag_lens.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS photo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                genre TEXT NOT NULL,
                brightness REAL NOT NULL,
                color_tone TEXT NOT NULL,
                saturation REAL NOT NULL,
                is_bw INTEGER NOT NULL DEFAULT 0,
                rating INTEGER NOT NULL DEFAULT 0,
                hash TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                embedding TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS tag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS photo_tag (
                photo_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (photo_id, tag_id),
                FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
            );
            """
        )
        
        # 마이그레이션: embedding 칼럼이 없으면 추가
        cursor = conn.execute("PRAGMA table_info(photo)")
        columns = {row[1] for row in cursor.fetchall()}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE photo ADD COLUMN embedding TEXT DEFAULT NULL")
            conn.commit()


def insert_photo(data: dict[str, object]) -> int:
    with get_connection() as conn:
        embedding_json = None
        if "embedding" in data and data["embedding"] is not None:
            embedding_json = json.dumps(data["embedding"].tolist() if isinstance(data["embedding"], np.ndarray) else data["embedding"])
        
        cursor = conn.execute(
            """
            INSERT INTO photo (
                filename, upload_date, genre, brightness, color_tone,
                saturation, is_bw, rating, hash, face_count, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["filename"],
                data["upload_date"],
                data["genre"],
                data["brightness"],
                data["color_tone"],
                data["saturation"],
                int(bool(data["is_bw"])),
                data.get("rating", 0),
                data["hash"],
                data.get("face_count", 0),
                embedding_json,
            ),
        )
        return int(cursor.lastrowid)


def link_tags(photo_id: int, tags: list[str]) -> None:
    unique_tags = sorted(set(tags))
    with get_connection() as conn:
        for tag_name in unique_tags:
            conn.execute("INSERT OR IGNORE INTO tag(name) VALUES (?)", (tag_name,))
            tag_row = conn.execute("SELECT id FROM tag WHERE name = ?", (tag_name,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO photo_tag(photo_id, tag_id) VALUES (?, ?)",
                (photo_id, int(tag_row["id"])),
            )


def replace_tags(photo_id: int, tags: list[str]) -> None:
    """사진의 기존 태그 연결을 모두 지우고 새 태그로 교체 (재분류용)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM photo_tag WHERE photo_id = ?", (photo_id,))
    link_tags(photo_id, tags)


def _build_filter_query(tags: list[str], rating_min: int, date_from: str, date_to: str) -> tuple[str, list[object]]:
    where = ["1=1"]
    params: list[object] = []

    if rating_min > 0:
        where.append("p.rating >= ?")
        params.append(rating_min)

    if date_from:
        where.append("p.upload_date >= ?")
        params.append(date_from)

    if date_to:
        where.append("p.upload_date <= ?")
        params.append(date_to)

    base = f"""
        SELECT
            p.*, GROUP_CONCAT(t.name, ' ') AS tags
        FROM photo p
        LEFT JOIN photo_tag pt ON p.id = pt.photo_id
        LEFT JOIN tag t ON t.id = pt.tag_id
        WHERE {' AND '.join(where)}
    """

    if tags:
        placeholders = ",".join("?" for _ in tags)
        base += f"""
            AND p.id IN (
                SELECT pt2.photo_id
                FROM photo_tag pt2
                JOIN tag t2 ON t2.id = pt2.tag_id
                WHERE t2.name IN ({placeholders})
                GROUP BY pt2.photo_id
                HAVING COUNT(DISTINCT t2.name) = ?
            )
        """
        params.extend(tags)
        params.append(len(tags))

    base += " GROUP BY p.id ORDER BY p.id DESC"
    return base, params


def get_photos(tags: list[str] | None = None, rating_min: int = 0, date_from: str = "", date_to: str = "") -> list[sqlite3.Row]:
    tags = tags or []
    query, params = _build_filter_query(tags, rating_min, date_from, date_to)
    with get_connection() as conn:
        return list(conn.execute(query, params).fetchall())


def get_photo(photo_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM photo WHERE id = ?", (photo_id,)).fetchone()
        if row is None:
            return None
        tags = conn.execute(
            """
            SELECT t.name FROM tag t
            JOIN photo_tag pt ON t.id = pt.tag_id
            WHERE pt.photo_id = ?
            ORDER BY t.name
            """,
            (photo_id,),
        ).fetchall()
        row_dict = dict(row)
        row_dict["tags"] = [t["name"] for t in tags]
        return row_dict


def get_tag_counts() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return list(
            conn.execute(
                """
                SELECT t.name, COUNT(pt.photo_id) AS photo_count
                FROM tag t
                LEFT JOIN photo_tag pt ON pt.tag_id = t.id
                GROUP BY t.id, t.name
                ORDER BY photo_count DESC, t.name ASC
                """
            ).fetchall()
        )


def find_duplicate_by_hash(image_hash: str) -> sqlite3.Row | None:
    """
    중복 사진 확인 (hash 기반)
    같은 해시를 가진 사진이 있으면 반환, 없으면 None
    """
    with get_connection() as conn:
        return conn.execute("SELECT id, filename, genre FROM photo WHERE hash = ?", (image_hash,)).fetchone()


def update_rating(photo_id: int, rating: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE photo SET rating = ? WHERE id = ?", (rating, photo_id))


def update_classification(photo_id: int, genre: str, face_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE photo SET genre = ?, face_count = ? WHERE id = ?",
            (genre, face_count, photo_id),
        )


def delete_photo(photo_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM photo WHERE id = ?", (photo_id,))


def find_similar(embedding: np.ndarray | str, exclude_photo_id: int, top_k: int = 5) -> list[dict[str, object]]:
    """CNN 특징 벡터 기반 유사도 검색 (코사인 유사도)"""
    # embedding이 문자열이면 JSON으로부터 변환
    if isinstance(embedding, str):
        try:
            embedding = np.array(json.loads(embedding), dtype=np.float32)
        except (json.JSONDecodeError, TypeError):
            embedding = np.zeros(1024, dtype=np.float32)
    
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, genre, rating, embedding FROM photo WHERE id != ? AND embedding IS NOT NULL",
            (exclude_photo_id,),
        ).fetchall()

    scored: list[dict[str, object]] = []
    for row in rows:
        try:
            target_embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)
            # 코사인 유사도: (A·B) / (||A|| * ||B||)
            dot_product = np.dot(embedding, target_embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(target_embedding)
            
            if norm_a > 0 and norm_b > 0:
                similarity = float(dot_product / (norm_a * norm_b))
                # [-1, 1] 범위를 [0, 100]으로 정규화
                similarity_pct = max(0.0, (similarity + 1) / 2 * 100)
            else:
                similarity_pct = 0.0
            
            scored.append({
                "id": row["id"],
                "filename": row["filename"],
                "genre": row["genre"],
                "rating": row["rating"],
                "similarity": round(similarity_pct, 2),
            })
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
