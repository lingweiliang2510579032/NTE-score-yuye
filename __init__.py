"""yuye 评分包。权重数据来自异环工坊开放 API（data/weights.json），使用时标注来源。"""

from __future__ import annotations

import json
import time
import asyncio
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from collections.abc import Sequence

import httpx

from ...contract import GradeSpec, BaseScorer, ScorerMeta
from ...registry import register_scorer
from ....utils.sdk.tajiduo_model import CharacterDetail, CharacterProperty, CharacterSuitItem
from ....utils.resource.RESOURCE_PATH import SCORING_PATH

_DATA = Path(__file__).parent / "data"
_ASSETS = Path(__file__).parent / "assets"

# 权重数据源：异环工坊开放接口（4 小时 TTL 缓存，接口不可用时回退本地 weights.json）
_API_URL = "https://yh.zzzmap.com/api/open/game-character/weight-configs"
_API_TTL_SECONDS = 4 * 3600
_API_TIMEOUT = 8.0

# 远程权重内存缓存
_REMOTE_WEIGHTS: dict | None = None
_REMOTE_FETCHED_AT: float = 0.0
_FETCH_LOCK: asyncio.Lock | None = None

WORKSHOP_TOTAL_AREA = 35
WORKSHOP_RATING_TOTAL = 280.0

WORKSHOP_GRADES = (
    (1.0, "ACE"),
    (0.9288, "SSS"),
    (0.8571, "SS"),
    (0.7857, "S"),
    (0.71143, "A+"),
    (0.55, "A"),
    (0.38, "B"),
    (0.28, "C"),
    (0.0, "D"),
)

PIECE_GRADES = (
    (0.8, "ACE"),
    (0.7, "SSS"),
    (0.6, "SS"),
    (0.5, "S"),
    (0.4, "A"),
    (0.3, "B"),
    (0.2, "C"),
    (0.0, "D"),
)

# 塔吉多装备数据用双 p（damageuppsychebase），工坊权重接口/weights.json 用单 p（damageupsychebase），
# 两者是同一属性「魂属性异能伤害增强」，按游戏侧拼写统一。
_PROP_ID_ALIASES = {"damageupsychebase": "damageuppsychebase"}


def _canon_prop_id(prop_id: str) -> str:
    pid = (prop_id or "").lower()
    return _PROP_ID_ALIASES.get(pid, pid)


SHOW_DRIVE_BADGE = True


@dataclass(frozen=True, slots=True, kw_only=True)
class _EquipmentView:
    item_id: str
    display: str
    grade: str | None
    unlocked_subs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Result:
    score: float
    display: str
    grade: str
    equipment: tuple[_EquipmentView, ...]
    main_weights: dict[str, float] = field(default_factory=dict)
    sub_weights: dict[str, float] = field(default_factory=dict)
    main_ids: frozenset[str] = frozenset()
    sub_ids: frozenset[str] = frozenset()
    effective_ids: frozenset[str] = frozenset()
    effective_names: frozenset[str] = frozenset()

    def is_role_prop_effective(self, prop: CharacterProperty) -> bool:
        ids = _attr_name_ids().get(prop.name, ())
        return any(
            self.main_weights.get(i, 0.0) >= 0.4 or self.sub_weights.get(i, 0.0) >= 0.4
            for i in ids
        )

    def is_main_prop_counted(self, prop: CharacterProperty) -> bool:
        return self.main_weights.get(_canon_prop_id(prop.id), 0.0) >= 0.4

    def is_sub_prop_recommended(self, prop: CharacterProperty) -> bool:
        return self.sub_weights.get(_canon_prop_id(prop.id), 0.0) >= 0.4

    def highlight_color(self, prop: CharacterProperty, locked: bool) -> tuple[int, int, int] | None:
        return (255, 176, 74) if not locked else (255, 200, 130)


def _normalize_weights(raw: dict) -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    result: dict[str, dict[str, dict[str, float] | frozenset[str]]] = {}
    for char_id, data in raw.items():
        result[str(char_id)] = {
            "main": {_canon_prop_id(attr): float(weight) for attr, weight in (data.get("main") or {}).items()},
            "sub": {_canon_prop_id(attr): float(weight) for attr, weight in (data.get("sub") or {}).items()},
            "highlight": frozenset(_canon_prop_id(attr) for attr in (data.get("highlight") or [])),
        }
    return result


@lru_cache(maxsize=1)
def _local_weights() -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    """本地兜底权重（data/weights.json，异环工坊接口快照）。"""
    path = _DATA / "weights.json"
    if not path.exists():
        raise ValueError(f"评分数据缺失: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_weights(raw)


def _weights() -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    """生效权重：远程接口成功优先，否则本地兜底。"""
    if _REMOTE_WEIGHTS is not None:
        return _REMOTE_WEIGHTS
    return _local_weights()


def _parse_api_payload(payload: dict) -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    """异环工坊权重接口响应 -> 包内权重格式。"""
    raw: dict[str, dict] = {}
    for char in payload.get("data") or []:
        char_id = str(char.get("itemId") or "").strip()
        if not char_id:
            continue
        wc = char.get("weightConfig") or {}
        weights = wc.get("weights") if isinstance(wc, dict) else (wc if isinstance(wc, list) else None)
        main: dict[str, float] = {}
        sub: dict[str, float] = {}
        highlight: list[str] = []
        for item in weights or []:
            key = _canon_prop_id(str(item.get("key") or ""))
            if not key:
                continue
            main_value = float(item.get("main_value") or 0)
            sub_value = float(item.get("value") or 0)
            if main_value > 0:
                main[key] = main_value
            if sub_value > 0:
                sub[key] = sub_value
            if item.get("highlight"):
                highlight.append(key)
        raw[char_id] = {
            "name": str(char.get("name") or ""),
            "main": main,
            "sub": sub,
            "highlight": highlight,
        }
    if not raw:
        raise ValueError("权重接口未解析到角色数据")
    return _normalize_weights(raw)


def _fetch_lock() -> asyncio.Lock:
    global _FETCH_LOCK
    if _FETCH_LOCK is None:
        _FETCH_LOCK = asyncio.Lock()
    return _FETCH_LOCK


async def _maybe_refresh_weights(force: bool = False) -> None:
    """按 TTL 拉取远程权重；失败静默，保留现有数据（远程或本地兜底）。"""
    global _REMOTE_WEIGHTS, _REMOTE_FETCHED_AT
    now = time.time()
    if not force and _REMOTE_WEIGHTS is not None and now - _REMOTE_FETCHED_AT < _API_TTL_SECONDS:
        return
    async with _fetch_lock():
        now = time.time()
        if not force and _REMOTE_WEIGHTS is not None and now - _REMOTE_FETCHED_AT < _API_TTL_SECONDS:
            return
        try:
            async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
                resp = await client.get(_API_URL)
                resp.raise_for_status()
                payload = resp.json()
            _REMOTE_WEIGHTS = _parse_api_payload(payload)
            _REMOTE_FETCHED_AT = time.time()
        except Exception:
            # 接口不可用：保留本地兜底，下次刷新面板再试
            pass


@lru_cache(maxsize=1)
def _attr_names() -> dict[str, str]:
    path = SCORING_PATH / "attributes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(attr_id).lower(): str(info.get("name", "")) for attr_id, info in raw.items()}


@lru_cache(maxsize=1)
def _attr_flags() -> dict[str, dict]:
    path = SCORING_PATH / "attributes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(attr_id).lower(): info for attr_id, info in raw.items()}


@lru_cache(maxsize=1)
def _attr_name_ids() -> dict[str, tuple[str, ...]]:
    path = SCORING_PATH / "attributes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for attr_id, info in raw.items():
        name = str(info.get("name", ""))
        if name:
            result.setdefault(name, []).append(str(attr_id).lower())
    return {name: tuple(ids) for name, ids in result.items()}


def _piece_grid(item: CharacterSuitItem) -> str | None:
    if not item.id.startswith("cell"):
        return None
    try:
        return str(int(item.id.split("_", 1)[0][4:]))
    except ValueError:
        return None


def _quality_factor(item_id: str) -> float:
    if not item_id:
        return 0.6
    i = item_id.lower()
    if "orange" in i or "gold" in i:
        return 1.0
    if "purple" in i:
        return 0.8
    return 0.6


def _weight_for(
    prop_id: str,
    weights: dict[str, float],
    name: str | None = None,
    name_map: dict[str, float] | None = None,
) -> float:
    pid = _canon_prop_id(prop_id)
    v = weights.get(pid, 0.0)
    if not v:
        v = weights.get(pid.replace("base", ""), 0.0)
    if not v:
        v = weights.get(pid + "base", 0.0)
    if not v and name and name_map:
        v = name_map.get(name, 0.0)
    return v


def _workshop_grade(total: float) -> str:
    ratio = total / WORKSHOP_RATING_TOTAL
    for threshold, grade in WORKSHOP_GRADES:
        if ratio >= threshold:
            return grade
    return WORKSHOP_GRADES[-1][1]


def _piece_grade(score: float, max_score: float) -> str:
    if max_score <= 0:
        return PIECE_GRADES[-1][1]
    ratio = score / max_score
    for threshold, grade in PIECE_GRADES:
        if ratio >= threshold:
            return grade
    return PIECE_GRADES[-1][1]


class YuyeScorer(BaseScorer):
    scorer_id = "yuye"
    meta = ScorerMeta(
        name="yuye",
        author="雨夜",
        version="1.4.0",
        description="权重数据来自异环工坊",
    )

    def grades(self) -> Sequence[GradeSpec]:
        return (
            GradeSpec(id="ACE", color=(255, 208, 96), icon=_ASSETS / "rank_ACE.png"),
            GradeSpec(id="SSS", color=(255, 112, 112), icon=_ASSETS / "rank_SSS.png"),
            GradeSpec(id="SS", color=(255, 170, 96), icon=_ASSETS / "rank_SS.png"),
            GradeSpec(id="S", color=(255, 208, 96), icon=_ASSETS / "rank_S.png"),
            GradeSpec(id="A+", color=(200, 140, 250), icon=_ASSETS / "rank_A+.png"),
            GradeSpec(id="A", color=(170, 165, 240), icon=_ASSETS / "rank_A.png"),
            GradeSpec(id="B", color=(176, 182, 214), icon=_ASSETS / "rank_B.png"),
            GradeSpec(id="C", color=(120, 214, 162), icon=_ASSETS / "rank_C.png"),
            GradeSpec(id="D", color=(140, 150, 170), icon=_ASSETS / "rank_D.png"),
        )

    def describe_char(self, char_id: str) -> str:
        weights = _weights().get(str(char_id))
        if not weights:
            return ""
        sub = weights.get("sub") or {}
        main = weights.get("main") or {}
        lines = ["评分标准：异环工坊小程序（350 原始分，按 280 评级）"]
        if sub:
            top = sorted(sub.items(), key=lambda item: -item[1])[:5]
            lines.append("有效副词条：" + "、".join(f"{attr}×{weight:g}" for attr, weight in top))
        if main:
            lines.append("核心主词条：" + "、".join(sorted(main)))
        return "\n".join(lines)

    async def prepare(self) -> None:
        await _maybe_refresh_weights(force=True)
        _weights()

    async def close(self) -> None:
        global _REMOTE_WEIGHTS, _REMOTE_FETCHED_AT
        _REMOTE_WEIGHTS = None
        _REMOTE_FETCHED_AT = 0.0
        _local_weights.cache_clear()

    async def score_character(self, character: CharacterDetail) -> _Result | None:
        await _maybe_refresh_weights()
        weights = _weights().get(str(character.id))
        if not weights:
            return None
        items = (*character.suit.core, *character.suit.pie) if character.suit.id else ()
        if not items:
            return None

        sub_weights = weights.get("sub") or {}
        main_weights = weights.get("main") or {}
        max_weight = sum(sorted(sub_weights.values(), reverse=True)[:4])
        if max_weight <= 0:
            return None

        effective_ids = frozenset(weights.get("highlight") or ())
        main_ids = frozenset(attr for attr, weight in main_weights.items() if weight > 0)
        sub_ids = frozenset(attr for attr, weight in sub_weights.items() if weight > 0)
        names = _attr_names()
        effective_names = frozenset(names.get(attr_id, "") for attr_id in effective_ids if names.get(attr_id))

        main_name_map: dict[str, float] = {}
        for attr_id, weight in main_weights.items():
            display = names.get(attr_id, "")
            if display and weight:
                main_name_map.setdefault(display, weight)

        total_main = 0.0
        total_sub = 0.0
        total_pie = 0.0
        equipment: list[_EquipmentView] = []
        for item in items:
            grid = _piece_grid(item)
            unlocked = item.lev // 5
            q = _quality_factor(item.id)
            if grid is None:
                main_w = max(
                    (
                        _weight_for(prop.id, main_weights, prop.name, main_name_map)
                        for prop in item.main_properties
                    ),
                    default=0.0,
                )
                sub_w = sum(
                    _weight_for(prop.id, sub_weights)
                    for prop in item.properties
                    if prop.value.strip()
                )
                piece_score = main_w * 50.0 * q + (10.0 / max_weight) * sub_w * 10.0 * q
                total_main += min(main_w * 50.0 * q, 50.0)
                total_sub += min((10.0 / max_weight) * sub_w * 10.0 * q, 100.0)
                piece_grade = _piece_grade(piece_score, 150.0)
                equipment.append(
                    _EquipmentView(
                        item_id=item.id,
                        display=f"{piece_score:.1f}分",
                        grade=piece_grade if SHOW_DRIVE_BADGE else None,
                        unlocked_subs=unlocked,
                    )
                )
            else:
                area = int(grid)
                actual = sum(
                    _weight_for(prop.id, sub_weights)
                    for prop in item.properties
                    if prop.value.strip()
                )
                piece_score = (10.0 / max_weight) * actual * area * q
                total_pie += piece_score
                piece_grade = _piece_grade(piece_score, 10.0 * area)
                equipment.append(
                    _EquipmentView(
                        item_id=item.id,
                        display=f"{piece_score:.1f}分",
                        grade=piece_grade if SHOW_DRIVE_BADGE else None,
                        unlocked_subs=unlocked,
                    )
                )

        total_pie = min(total_pie, 200.0)
        total = total_main + total_sub + total_pie
        grade = _workshop_grade(total)
        total = round(total, 1)
        return _Result(
            score=total,
            display=f"{total:g}",
            grade=grade,
            equipment=tuple(equipment),
            main_weights=dict(main_weights),
            sub_weights=dict(sub_weights),
            main_ids=main_ids,
            sub_ids=sub_ids,
            effective_ids=effective_ids,
            effective_names=effective_names,
        )


register_scorer(YuyeScorer())
