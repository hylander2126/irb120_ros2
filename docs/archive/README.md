# Development archive

These dated source snapshots preserve experiment evidence and superseded design
history. Current instructions live in the package READMEs and
[contact selection](../../irb120_perception/CONTACT_SELECTION.md).

Live DBSCAN, standalone live SAM detectors, streaming stereo prototype, robot
cloud mask, live digital-twin/confidence tools, and RealSense-only TSDF CLI were
retired on 2026-10-24. Their source and dedicated tests are recoverable from Git
commit `893f151461f1efb95c13dfefdf6df6fb4ce5a69e` (before this cleanup):

```bash
git show 893f151:irb120_perception/irb120_perception/object_detector_dbscan.py
```

SAM segmentation and TSDF fusion inside the supported offline FFS pipeline remain
active. Preserve active calibrations, object registries and experiment meshes.
