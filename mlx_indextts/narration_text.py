"""Conservative text cleanup and speaker assignment for spoken narration."""

from __future__ import annotations

import html
import re
import unicodedata


_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_NUMERIC_REFERENCE = re.compile(
    r"(?:\(|\[|【)\s*(?:第\s*)?"
    r"[0-9一二三四五六七八九十百千万零〇]+"
    r"(?:\s*[-–—,，、./至到]\s*[0-9一二三四五六七八九十百千万零〇]+)*"
    r"\s*(?:\)|\]|】)"
)
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def _remove_symbol_emoji(value: str) -> str:
    """Drop decorative Unicode symbols while retaining letters and punctuation."""
    return "".join(
        character
        for character in value
        if unicodedata.category(character) not in {"So", "Sk"}
    )


def clean_narration_text(text: str | None) -> str:
    """Prepare pasted prose for TTS without rewriting its meaning.

    The cleanup deliberately leaves dates, money, percentages and ordinary
    numbers intact for the model's own text normalizer. Bracketed numeric
    references are removed, while explanatory bracket contents are retained.
    """
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    value = _INVISIBLE.sub("", value)
    value = _MARKDOWN_IMAGE.sub(lambda match: match.group(1), value)
    value = _MARKDOWN_LINK.sub(lambda match: match.group(1), value)
    value = re.sub(r"```(?:[^\n]*)\n?|```", "", value)
    value = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|>|[-+*]\s+)\s*", "", value)
    value = _URL.sub("", value)
    value = _NUMERIC_REFERENCE.sub("", value)
    value = re.sub(r"(?m)^\s*(?:第\s*)?\d+\s*$", "", value)
    value = re.sub(r"[\[\]【】()（）{}《》〈〉]", "", value)
    value = re.sub(r"[*#_=~^|]{2,}", "", value)
    value = _remove_symbol_emoji(value)
    value = re.sub(r"([。！？!?；;，,、])\1+", r"\1", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *([，。！？；：、,.!?;:]) *", r"\1", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def split_narration_units(text: str | None, mode: str = "paragraph") -> list[str]:
    """Split narration into natural paragraphs, non-empty lines, or sentences."""
    value = str(text or "").strip()
    if not value:
        return []
    if mode == "sentence":
        return [part.strip() for part in _SENTENCE_BOUNDARY.split(value) if part.strip()]
    if mode == "line":
        return [part.strip() for part in value.splitlines() if part.strip()]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", value) if part.strip()]
    return paragraphs or [value]


def assign_voices_to_units(
    units: list[str], voice_ids: list[str], switch_every: int = 1
) -> list[tuple[str, str]]:
    """Assign units to voices in a stable round-robin rotation."""
    voices = [str(voice_id) for voice_id in voice_ids if str(voice_id).strip()]
    if not voices:
        return []
    interval = max(1, int(switch_every))
    return [
        (unit, voices[(index // interval) % len(voices)])
        for index, unit in enumerate(units)
    ]
