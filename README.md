# mp-comfyui-lab

`mp-comfyui-lab` is an executor-neutral command-line workflow experimentation
tool. RunningHub is the first discovery and execution adapter; local ComfyUI and
other executors can implement the same executor contract.

All examples below are PowerShell commands run from the repository root. They
are noninteractive: every command emits one JSON document and never asks for
terminal input.

## Setup

Use this repository's virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$Lab = ".\.venv\Scripts\mp-comfy-lab.exe"
& $Lab version
```

`config.local.yaml` holds local paths only and is Git-ignored. A minimal
configuration is:

```yaml
experiments_root: experiments
vaults:
  characters: Z:/fixtures/characters
```

`experiments_root` is resolved from the repository root. A brief may refer to a
local fixture as `portrait.png`, or to a configured vault alias as
`vault://characters/hero.png`. Vault aliases keep machine-specific absolute
paths out of briefs and durable approval records.

Fixture arrays are ordered. Each reference image also has a unique
`binding_index` that binds it to the corresponding graph input. Keep the order
and indexes intentional because fixture bytes and bindings are part of the
exact run hash.

Both `config.local.yaml` and runtime contents under `experiments/` are ignored.
Only `experiments/.gitkeep` is tracked.

## Discover

Search RunningHub's public workflow catalog:

```powershell
$search = & $Lab search "qwen character" --sort REPUTATION --page 1 --size 30 | ConvertFrom-Json
$search.data.results | Select-Object workflow_id, url, name, score
```

`--include-graphs` adds graph evidence for selected duplicate-group
representatives. Without it, search remains detail-oriented and lighter.

Inspect a workflow by canonical or accepted RunningHub post URL:

```powershell
$url = "https://www.runninghub.ai/post/1980860923768426497"
$inspection = & $Lab inspect $url | ConvertFrom-Json
$inspection.data
```

Find workflows similar to a known post using public evidence derived from its
model inventory, tags, prompt structure, and normalized capability roles:

```powershell
$similar = & $Lab similar $url --sort REPUTATION --page 1 --size 30 --include-graphs | ConvertFrom-Json
$similar.data.results
```

Discovery labels content as `SFW`, `NSFW`, or `UNKNOWN`. `UNKNOWN` means the
public metadata did not provide enough explicit evidence; it does not mean
safe. Classification reasons travel with each candidate.

RunningHub search and detail web endpoints are observed public contracts. They
are guarded by sanitized offline fixtures and the opt-in read-only integration
test, but they are not a guaranteed RunningHub SDK. Run the live contract probe
only when explicitly intended:

```powershell
$env:MP_COMFYUI_LAB_LIVE_DISCOVERY = "1"
& .\.venv\Scripts\python.exe -m pytest tests/test_runninghub_readonly_integration.py -q
Remove-Item Env:MP_COMFYUI_LAB_LIVE_DISCOVERY
```

That test calls public search and detail only. It does not upload a fixture,
submit a graph, poll a paid task, or mutate remote state.

## Plan

Create a brief such as `examples/qwen-character.yaml`, then resolve its public
sources and materialize a deterministic approval plan:

```powershell
$plan = & $Lab plan examples/qwen-character.yaml | ConvertFrom-Json
$ExperimentId = $plan.data.experiment_id
$PlanHash = $plan.data.plan_hash
$VariantId = $plan.data.variant_ids[0]
$plan.data
```

Planning is read-only. The plan records sanitized source evidence, normalized
graphs, fixture fingerprints, expected outputs, exact run hashes, capability
coverage, and warnings. Review the emitted plan artifact before approving it.

Experimentation proceeds in capability waves. There is no arbitrary fixed task
count or budget ceiling in the planner. Exploration stops when prior run
evidence shows capability saturation, meaning no approved, non-dominated,
novel capability signature remains. Paid execution is still explicit for every
CLI run and always requires the current exact plan hash.

## Run And Recover

Run one approved variant with the exact hash returned by `plan`:

```powershell
$run = & $Lab run $ExperimentId $VariantId --approve-plan-hash $PlanHash | ConvertFrom-Json
$RunId = $run.data.run_id
$run.data
```

`run` is approval-gated and can submit paid remote work. A changed, stale, or
mistyped plan hash is rejected before submission. Search, inspect, similar, and
plan are public read-only discovery operations; run, rerun, resume, and
reconcile enter the paid execution lifecycle and may mutate provider state.

An exact fingerprint is not silently repeated. Use `--rerun` with a durable,
nonempty reason:

```powershell
$rerun = & $Lab run $ExperimentId --rerun $VariantId --reason "measure seed variance after visual review" --approve-plan-hash $PlanHash | ConvertFrom-Json
```

Inspect the durable local state without starting another run:

```powershell
$status = & $Lab status $ExperimentId $RunId | ConvertFrom-Json
$status.data
```

Resume a known submitted provider task after an interrupted or incomplete
lifecycle:

```powershell
$resumed = & $Lab resume $ExperimentId $RunId | ConvertFrom-Json
```

If submission was ambiguous and a human has independently recovered the real
provider task ID, reconcile that evidence before continuing:

```powershell
$reconciled = & $Lab reconcile $ExperimentId $RunId --task-id "provider-task-id" | ConvertFrom-Json
```

Do not use `reconcile` to guess a task ID. The stored recovery disposition and
provider evidence determine whether resume or reconciliation is valid.

## Compare And Select

Generate deterministic facts, duplicate groups, contact sheets, a judgment
template, finalists, and an absolute HTML gallery path:

```powershell
$comparison = & $Lab compare $ExperimentId | ConvertFrom-Json
$comparison.data.gallery
Start-Process $comparison.data.gallery
```

To validate a completed judgment document against the current comparison facts:

```powershell
$comparison = & $Lab compare $ExperimentId --judgments .\judgments.json | ConvertFrom-Json
$finalists = & $Lab finalists $ExperimentId --judgments .\judgments.json | ConvertFrom-Json
```

Without a judgment file, `finalists` uses current objective evidence and any
valid stored judgments:

```powershell
$finalists = & $Lab finalists $ExperimentId | ConvertFrom-Json
$finalists.data.variant_ids
```

Finalist selection is deliberately human. Review the gallery and evidence,
then select exactly one current finalist:

```powershell
$SelectedVariantId = $finalists.data.variant_ids[0]
$selection = & $Lab select $ExperimentId $SelectedVariantId | ConvertFrom-Json
```

There is no automatic winner promotion. A stale review generation, changed
artifact, or non-finalist selection fails closed.

## Export And Promote

Export the selected finalist:

```powershell
$export = & $Lab export $ExperimentId | ConvertFrom-Json
$Bundle = $export.data.bundle
Get-ChildItem $Bundle
```

The bundle is immutable, self-contained, and keyed by the selected evidence.
It includes the source and selected API graphs, semantic patch, model stack,
ordered fixture copies and manifest, prompts, selected outputs, execution and
comparison evidence, review artifacts, selection, and `handoff.json` with
artifact hashes. Export never performs production promotion.

Promote through the existing `port-comfy-workflow` skill. Its workflow is
strip, audit, register, smoke, and promote. The handoff instructions start from
`selected-workflow-api.json`; production registration and smoke testing remain
separate controlled operations.

## Schemas

Committed JSON Schemas are generated from the current Pydantic brief, plan, and
handoff models. Export intentional model changes atomically:

```powershell
& $Lab schemas export --output schemas
```

Check in memory that committed files exist and match without modifying them:

```powershell
& $Lab schemas check --output schemas
```

Schema JSON has sorted keys, stable two-space formatting, and one trailing
newline.

## Output Contract

Every invocation writes exactly one compact JSON document to stdout, including
help and errors. Automation should parse stdout and inspect `ok`, `command`,
`data` or `error`, and `artifacts`. Human-readable progress or diagnostics may
be written to stderr; do not merge stderr into the JSON stream. Durable
provider diagnostics also appear in structured run data and artifact files.

Stable process exit categories are:

| Code | Category | Typical error kinds |
| ---: | --- | --- |
| 0 | success | none |
| 2 | command or manifest usage | `usage` |
| 3 | missing local or public record | `not_found` |
| 4 | provider contract or transport | `provider` |
| 5 | exact approval rejected | `approval` |
| 6 | execution or artifact conflict | `execution`, `conflict` |
| 7 | recoverable incomplete work | `incomplete` |
| 8 | finalist or selection state | `selection`, `selection_required`, `stale_selection` |
| 9 | safety or integrity rejection | `safety`, `integrity` |
| 10 | missing or stale committed schema | `schema` |

## Artifact Layout

An experiment uses an ignored directory under `experiments/<experiment-id>/`.
Depending on completed phases it contains:

```text
brief.yaml
search-results.json
attempts.jsonl
sources/
evidence/
variants/
plans/
fixtures/
runs/
outputs/
comparisons/
reviews/
finalists.json
selection.json
export/<variant-id>/<selection-hash>/
```

Commands return authoritative absolute artifact paths. Treat approved plans,
runs, review generations, selections, and exported bundles as evidence records;
create a new plan, run, review, or export generation instead of editing them.
