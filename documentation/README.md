# Documentation index

Start with the [root README](../README.md) for current setup and commands. This
index connects project background, integration evidence, proposed work, and code
reviews. Historical documents describe their recorded source state; they are not
automatically current specifications.

## Reading order

1. [Project plan, revision 3](kernel/pre_trained_fast_lab_plan_v3.md): research
   question, architecture, and the overall K/U-track design.
2. [K-track guide](kernel/k_track_guide.md) and
   [findings](kernel/k_track_findings.md): what was built and measured before
   the workspace migration.
3. [Adversarial review K-REVIEW-001](kernel/k_track_adversarial_review_2026-10-21.md):
   numerical, validation, measurement, and integration concerns.
4. [Integration records](#integration-records): the shared workspace and packaging
   changes that supersede some older repository-layout claims.
5. [Kernel phase 1 draft](plans/2026-10-25-kernel-phase-1.md): evidence mapped to
   current code and a proposed correctness-first CUDA milestone. **Not approved.**

## Layout

```text
documentation/
  README.md       navigation, status, and documentation conventions
  integration/    baseline and migration validation records
  kernel/         related historical kernel plan, guide, findings, and review
  plans/          new proposals across backend, quant, or the whole workspace
  reviews/        new dated code and design reviews
  mockups/        design images
  artifacts/      generated logs, builds, and profiler outputs; ignored by Git
```

The four historical kernel documents stay together because they describe the
same development effort. New plans and reviews use the shared directories and
link to that background rather than duplicating it.

## Document register

Dates below are the dates stated by the documents, not inferred execution dates.

| Document | Date / revision | Type and status |
| --- | --- | --- |
| [Project plan](kernel/pre_trained_fast_lab_plan_v3.md) | Revision 3; date unspecified | Historical design; partly implemented |
| [K-track findings](kernel/k_track_findings.md) | 2026-10-14 | Historical measurements and findings |
| [K-track guide](kernel/k_track_guide.md) | 2026-10-20 | Historical implementation/status snapshot |
| [K-REVIEW-001](kernel/k_track_adversarial_review_2026-10-21.md) | 2026-10-21 | Historical review; findings have mixed current status |
| [Independent baselines](integration/independent-baselines.md) | 2026-10-23 | Integration evidence record |
| [Workspace validation](integration/workspace-validation.md) | 2026-10-24 | Integration evidence record |
| [Quant resource validation](integration/quant-resource-validation.md) | 2026-10-24 | Packaging evidence record |
| [K-PLAN-001: kernel phase 1](plans/2026-10-25-kernel-phase-1.md) | 2026-10-25 | **Draft — awaiting agreement; not implemented** |
| [Homepage mockup](mockups/Homepage.png) | Date unspecified | UI design reference |

### Integration records

These three records form one sequence:

- [Independent baselines](integration/independent-baselines.md): test outcomes
  before combining dependency environments.
- [Workspace validation](integration/workspace-validation.md): reconciled
  dependencies, shared-environment test results, and execution issues.
- [Quant resource validation](integration/quant-resource-validation.md): source
  distribution/wheel contents, resource access, and regression checks.

Read their remaining-work sections as statements made at the time. The nested
lockfiles were subsequently retired, backend became installable, and quant's
runtime resources moved inside its package. The root README describes current
commands. Kernel performance and quantized API serving remain further work.

## Adding plans and reviews

Use descriptive lowercase filenames with the initial date:

- `plans/YYYY-MM-DD-topic.md`, using the [plan template](plans/_template.md).
- `reviews/YYYY-MM-DD-topic-review.md`, using the [review template](reviews/_template.md).

Keep the filename stable while discussing the same proposal; Git records its
revisions. A new phase or materially different replacement gets a new document
and links to the previous one. Keep IDs stable: the existing kernel review is
`K-REVIEW-001`, and the first new kernel plan is `K-PLAN-001`. Allocate the next
unused number within the relevant scope and document type.

For each new document:

1. Record status, dates, scope, source commit, related documents, and any source
   changes relevant to the analysis. A commit hash alone does not describe a
   dirty checkout.
2. Add a row to this register and link directly to relevant code and evidence.
   Use repository-relative Markdown links so navigation works on GitHub and in
   a local checkout.
3. Keep plan discussion and decisions in its decision log. Use statuses
   **Draft → Approved → In progress → Implemented**, or **Superseded**. Only mark
   approval after explicit agreement, and implementation after validation.
4. Keep review observations as dated snapshots. A later review references
   finding IDs and records **open**, **partially addressed**, **closed**, or
   **superseded**, with evidence. Do not erase an original finding to imply it
   was never present.

## Git and evidence policy

- Commit Markdown, templates, and intentional design assets. `.gitignore` excludes
  generated `documentation/artifacts/`, not the whole documentation tree.
- Use normal Git commits for draft revisions; avoid `final`, `final-v2`, and
  timestamped copies of the same evolving proposal. Historical filenames already
  present here are retained.
- Commit documentation organization separately from kernel implementation. Plan
  approval and a code commit are different events; saving a draft approves no code.
- Preserve reported results, dates, and caveats in historical evidence. Add a new
  record for new execution rather than replacing baseline numbers.
- Record the commands, environment, source identity, input identity, and durable
  artifact location for validation. Distinguish newly executed checks from
  results supplied by someone else or read from older artifacts.
- Generated logs, wheels, and profiler captures belong in `artifacts/` or durable
  external storage. A `/tmp` path is temporary; ignored local files are not a
  shared archive. Put links or checksums in the tracked summary.

## Current review context

The first kernel plan maps K-REVIEW-001 to the current implementation. In
particular, runtime resource packaging is addressed, some backend criticism is
superseded, and the quantized-reference activation-coverage gap remains present.
The historical guide's claims about separate lockfiles, root-level quant
resources, and ignored documentation describe its earlier repository state.
