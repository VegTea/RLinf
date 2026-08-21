# IsaacLab local assets

The IsaacLab installation and generated assets are intentionally kept outside
Git. In particular, do not commit `isaac_sim/`, `rlinf/assets.zip`, or
`rlinf/assets_isaaclab/` to the source repository.

Place the Isaac Sim installation at the path expected by the local environment
setup, and place the task assets under `rlinf/assets_isaaclab/`. These files
are machine-local and may be restored from the project asset archive or the
team's shared artifact storage. Keep the asset version and source recorded in
experiment notes when reproducing results.
