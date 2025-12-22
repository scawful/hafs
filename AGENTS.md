# Agent Instructions (hafs)

This repo is the shared core. Keep it generic and free of personal paths, hostnames, and project-specific training data.

- Put user-specific training scripts, Windows/GPU workflows, and domain generators (Zelda/Asar/Oracle/Gigaleak/YAZE) in `hafs_scawful`.
- Prefer placeholders and templates in `hafs` docs/configs; keep real values in plugin configs.
- If a change only affects the plugin, edit `~/Code/hafs_scawful` instead of this repo.
