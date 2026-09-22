"""Per-decision capture for distillation/replay (design §3). Side-effect-only: it NEVER
raises and NEVER changes agent behavior — a decision layer calls `record(...)` with the
exact input it saw and its raw/parsed output; records buffer per step and flush to
`decisions.jsonl` beside the run recorder's `log.jsonl`.

Held on the loop like `on_event`; attached onto `planner`/`reasoner` so their methods can
reach it via `getattr(self, "capture", None)`."""
from __future__ import annotations

import json
from pathlib import Path


def _safe(obj):
    """Best-effort JSON-serializable copy — fall back to `repr` for exotic values so a
    single unserializable field can never break capture (zero-behavior-change contract)."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return json.loads(json.dumps(obj, default=repr))


class Capture:
    def __init__(self, run_dir: str | Path, *, mode: str = "off"):
        self.dir = Path(run_dir)
        self.mode = mode                       # "off" | "decisions" | "distill"
        self.enabled = mode in ("decisions", "distill")
        self.deterministic = mode == "distill"  # gate for deterministic-layer records (Task 9)
        self._step = 0
        self._anchor: str | None = None
        self._seq = 0
        self._buf: list[dict] = []
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.dir / "decisions.jsonl").open("a", encoding="utf-8")
        else:
            self._fh = None

    def begin_step(self, step: int, anchor: str | None) -> None:
        self._step, self._anchor, self._seq, self._buf = step, anchor, 0, []

    def record(self, layer: str, *, kind: str = "model", model: str | None = None,
               input=None, output_raw=None, parsed=None, confidence=None,
               tokens: int = 0, latency_ms: int = 0, group_id: str | None = None,
               extra: dict | None = None) -> None:
        if not self.enabled:
            return
        try:
            rec = {
                "run_id": self.dir.name, "step": self._step, "seq": self._seq,
                "layer": layer, "kind": kind, "model": model,
                "input": _safe(input), "output_raw": output_raw,
                "output_parsed": _safe(parsed), "confidence": confidence,
                "tokens": tokens, "latency_ms": latency_ms,
                "anchor": self._anchor,
            }
            if group_id:
                rec["group_id"] = group_id
            if extra:
                rec["extra"] = _safe(extra)
            self._buf.append(rec)
            self._seq += 1
        except Exception:
            pass

    def record_round(self, layer: str, *, group_id: str, **kw) -> None:
        """A single KB-search round inside a multi-round decision (kind='model_round')."""
        self.record(layer, kind="model_round", group_id=group_id, **kw)

    def flush(self) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            for rec in self._buf:
                self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        except Exception:
            pass
        finally:
            self._buf = []

    def write_run_meta(self, meta: dict) -> None:
        if not self.enabled:
            return
        try:
            (self.dir / "run_meta.json").write_text(json.dumps(_safe(meta), indent=2))
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
