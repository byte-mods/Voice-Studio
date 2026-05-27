# Current Tasks

_Single source of truth for in-flight work. Updated by the agent before starting and after completing every task. Read this first to see what is in progress, what is queued, and what is done._

## In progress

_(none)_

## Queued

- [B] Section B — All real type bugs to zero (arg-type×55, attr-defined×11, assignment×6, no-any-return×17, no-untyped-call×5, misc×16)
- [C] Section C — Worst-3 deep clean (hf_import.py 56, architectures/factory.py 24, s2s/session.py 19)
- [D] Section D — Remaining annotation sweep (no-untyped-def remainder)
- [E] Section E — Unused-ignore (42) + final polish + fix MINOR carryovers from A (set[Task] type param)

## Completed (this session)

- [A-T1] Auto-fix ruff (115 fixes) + add `extend-immutable-calls` for FastAPI helpers — completed 2026-05-23T18:10Z — pytest 49/49, lint 181→51
- [A-T2] Fix F821×4 (TYPE_CHECKING numpy) + F841×3 (unused vars) — completed 2026-05-23T18:15Z — pytest 49/49, lint 51→44
- [A-T3] Sweep SIM105×21 (contextlib.suppress) + UP042×9 (StrEnum) + 14 manual fixes (RUF006 dangling-task fix, ContextVar None default, noqas) — completed 2026-05-23T18:30Z — pytest 49/49, lint 44→0
- [A-Tfinal] Section-level super-qa VERDICT PASS — completed 2026-05-23T18:35Z — 0 BLOCKER, 0 MAJOR, 2 MINOR (carryovers to E)
