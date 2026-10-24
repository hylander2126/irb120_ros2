# Repository collaboration

Use README.md as the documentation index. Current package READMEs and the contact
policy own supported behavior; docs/archive contains historical evidence, not
runtime instructions. Workspace-level collaboration preferences also apply when
this repo is used inside irb120_ws.

The supported perception path is fresh offline FFS capture/reconstruction and
observed-surface contact selection. Retired live detector/RealSense TSDF code is
recoverable from Git; do not silently restore it as a fallback. Preserve capture
identity and invalidate frozen targets after motion.

Keep operating guides concise, update actual CLI/launch examples when wiring
changes, and archive dated investigations rather than appending contradictory
“current” sections. Preserve validation limits, units, named frames and source
provenance. Calibration/reference assets and experiment variants are distinct
from disposable development output.

Use fake services and recorded data for software checks. Report separately what
was verified on hardware. Local hardware-interface guards and pinned dependency
ownership are intentional; avoid replacing them during unrelated cleanup.
