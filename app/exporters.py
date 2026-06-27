import json
from pathlib import Path
from typing import List

from app.models import OutputPaths, TranscriptSegment


def export_segments(
    segments: List[TranscriptSegment],
    result_dir: Path,
    filename: str,
) -> OutputPaths:
    stem = Path(filename).stem
    text_path = result_dir / f"{stem}.txt"
    markdown_path = result_dir / f"{stem}.md"
    json_path = result_dir / f"{stem}.json"
    srt_path = result_dir / f"{stem}.srt"

    text_path.write_text(_render_text(segments))
    markdown_path.write_text(_render_markdown(segments, filename))
    json_path.write_text(json.dumps([segment.model_dump() for segment in segments], indent=2, ensure_ascii=False))
    srt_path.write_text(_render_srt(segments))

    return OutputPaths(
        text=str(text_path),
        markdown=str(markdown_path),
        json_path=str(json_path),
        srt=str(srt_path),
    )


def _render_text(segments: List[TranscriptSegment]) -> str:
    return "\n".join(f"[{_time(segment.start)} - {_time(segment.end)}] {segment.speaker}: {segment.text}" for segment in segments) + "\n"


def _render_markdown(segments: List[TranscriptSegment], filename: str) -> str:
    lines = [f"# {filename}", ""]
    for segment in segments:
        lines.append(f"## {segment.speaker}")
        lines.append(f"[{_time(segment.start)} - {_time(segment.end)}] {segment.text}")
        lines.append("")
    return "\n".join(lines)


def _render_srt(segments: List[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}",
                    f"{segment.speaker}: {segment.text}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def _time(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _srt_time(seconds: float) -> str:
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    hours, remainder = divmod(whole, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}"
