# Shared GameWorld assets

```text
core/                         # GameWorld code (modified), Astray + Minecraft, model profiles
tasks/
  train/
    astray/                   # 20 training maps selected by train_seed
    minecraft/                # 20 quota variants and 6 additional hard-task YAMLs
  holdout/
    astray/                   # Original Astray task YAMLs
    minecraft/                # Original Minecraft task YAMLs
    <other original game IDs>/
```

Only the `03_astray` and `18_minecraft-clone-glm` game assets are bundled, with
some modifications relative to GameWorld-Games; see `../README.md` for source,
citation and the research-only disclaimer. Training YAMLs
are moved without content changes. The Minecraft overlay's unrelated older
Astray API is discarded; shared Astray uses the seed-parameterized API/page.
No `train_seed` means seed 42. Scoring rules and Minecraft core behavior are
unchanged.

`gameworld_root` selects `core`; `task_assets_roots.train` and
`task_assets_roots.holdout` select the phase's YAML directory. These settings
live under `game_options.gameworld`. A task entry can override the selected
directory with `task_assets_root`. Folder aliases `astray` and `minecraft`
retain the original game IDs `03_astray` and `18_minecraft-clone-glm`.

The GameWorld entry point accepts `task_assets_root` (CLI: `--task-assets-root`).
Without it, the bundled core loads holdout tasks. The checkpoint helper accepts
the same CLI option when checking a training task. Both execution and checkpoint
identity resolve the same YAML, and step identity includes both page/API hashes.

This changes paths and source identity. Existing checkpoints are not converted;
use a fresh run directory for the reorganized assets. No browser or API validation
was performed as part of the move.
