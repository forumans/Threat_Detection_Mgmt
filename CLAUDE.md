# Repository Instructions

## Prompt Logging

Every substantive prompt the user gives in this repository must be saved, verbatim, under a `prompts/` directory — organized by category, then grouped into topically-named files.

**Location.** `prompts/` is scoped per-project, matching how `docs/` and other project folders already work:
- `threat_detection_system/prompts/<category>/<topic>.md`
- `synthetic_threat_data_generator/prompts/<category>/<topic>.md`
- A root-level `prompts/<category>/<topic>.md` holds prompts that concern the repo as a whole or span both projects (e.g. the original request that produced both projects' architecture docs).

**Categories.** architecture, coding, testing, debugging, security, database, frontend, backend, agent, deployment, documentation, process (prompts about the prompt-logging system itself). Add a new category directory when a prompt doesn't fit an existing one — this list is a starting point, not a fixed enum.

**Grouping.** Within a category, group related prompts into one topically-named file rather than one file per prompt (e.g. every database-schema-related prompt goes into `database/db-creation-prompts.md`). Append to an existing file when a new prompt continues that topic; create a new file when it starts a different one.

**What to log.** Substantive requests, questions, or instructions that drove work or a decision. Skip pure acknowledgments/confirmations with no new instructional content (e.g. "looks good", "that worked").

**File format.** Each log file has a short summary, the prompts verbatim in order, and an outcomes table mapping each prompt to what it produced. See `threat_detection_system/prompts/database/db-creation-prompts.md` for the reference format.

**Git.** `prompts/` is gitignored in every project (local reference only, never pushed).

**When to log.** Immediately, as part of doing the work the prompt requested — not deferred, not batched, not left for the user to ask for again. Before ending a turn that involved a substantive prompt, check: has this prompt been filed under `prompts/` yet? A large implementation task is exactly when this is easiest to forget (the actual work crowds it out) and most valuable to have (it's the prompt worth remembering later). If a prompt was missed, backfill it as soon as it's noticed rather than leaving the log incomplete.
