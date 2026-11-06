"""Build the teacher curriculum — render training/teacher_lessons.jsonl
into token sequences the existing training pipeline consumes.

Teacher provenance: the lessons were authored by Muse (the teacher)
at the owner's direction, as training DATA. Every lesson record and
every rendered corpus carries the source string below; the SHA-256 of
the lessons file and of the rendered sequences are recorded in the
Gen 1 checkpoint manifest/report by the run script, so what the model
was taught — and that it was taught, not self-originated — is always
auditable. No teacher weights are involved anywhere in this path.

Rendering (deterministic; same tokenization and >=3-token line rule
as training/corpus.py and the Gen 0 curriculum):

  * each fact            -> one line (facts[0] is the canonical answer
                            sentence for the lesson's probe)
  * each worked example  -> one line
  * the counterexample   -> one line
  * the probe            -> "<prompt> <facts[0]>" (a Q&A line, in the
                            same shape as the core probes)
  * the exercise         -> "exercise : <question> answer : <answer>"

Usage: PYTHONPATH=src:training python3 training/build_teacher_curriculum.py
Writes training/teacher_curriculum.txt (one rendered line per line)
and prints the provenance record as JSON.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from steelbound.model import sha256_text, tokenize  # noqa: E402

SOURCE = "teacher:Muse (owner-directed)"
LESSONS_PATH = ROOT / "training" / "teacher_lessons.jsonl"
RENDERED_PATH = ROOT / "training" / "teacher_curriculum.txt"


def load_lessons(path: Path = LESSONS_PATH) -> list[dict]:
    lessons = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lessons.append(json.loads(line))
    return lessons


def render_lines(lesson: dict) -> list[str]:
    lines = list(lesson["facts"])
    lines.extend(lesson["examples"])
    lines.append(lesson["counterexample"])
    lines.append(f"{lesson['probe']['prompt']} {lesson['facts'][0]}")
    ex = lesson["exercise"]
    lines.append(f"exercise : {ex['question']} answer : {ex['answer']}")
    return lines


def render_sequences(lessons: list[dict]) -> list[list[str]]:
    """Token sequences, corpus-style: tokenize each rendered line and
    keep lines with at least 3 tokens."""
    seqs = []
    for lesson in lessons:
        for line in render_lines(lesson):
            toks = tokenize(line)
            if len(toks) >= 3:
                seqs.append(toks)
    return seqs


def teacher_probes(lessons: list[dict]) -> list[tuple[str, str]]:
    return [(lesson["probe"]["prompt"], lesson["probe"]["expected"])
            for lesson in lessons]


def provenance(lessons: list[dict], seqs: list[list[str]],
               path: Path = LESSONS_PATH) -> dict:
    return {
        "source": SOURCE,
        "lessons_file": "training/teacher_lessons.jsonl",
        "lessons_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "lesson_count": len(lessons),
        "topics": sorted({lesson["topic"] for lesson in lessons}),
        "rendered_sequences": len(seqs),
        "rendered_sha256": sha256_text("\n".join(" ".join(s) for s in seqs)),
    }


def main() -> int:
    lessons = load_lessons()
    seqs = render_sequences(lessons)
    lines = [line for lesson in lessons for line in render_lines(lesson)]
    RENDERED_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(provenance(lessons, seqs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
