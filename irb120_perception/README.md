# irb120_perception

Object detection for the IRB120 robot workspace. Subscribes to RealSense
camera streams, isolates objects on the workspace surface, and publishes
their 3D convex hulls, centroids, and orientations.

The independent [contact point selector](CONTACT_SELECTION.md) selects planar
push, forward-tip and press contacts and estimates tipping axes from object clouds.

Three segmentation backends exist, sharing plumbing (TF, ROI crop, PCA
orientation, convex hull, Detection3D/marker publishing, EMA smoothing) via
`perception_common.py` (`ObjectDetectorBase`) so neither duplicates that logic:

- `object_detector_dbscan` — pure geometry, no GPU required, fast.
- `object_detector_sam` — full automatic MobileSAM, useful for visual
  instance-segmentation experiments but expensive on CPU.
- `object_detector_dbscan_sam_cull` — an offline DBSCAN refinement: one
  prompted MobileSAM mask from cam1 removes DBSCAN points outside the visual
  object silhouette.

---

## Backend

### DBSCAN

Clusters the 3D pointcloud spatially using DBSCAN. Works well when objects
are clearly separated by a gap in 3D space. Requires no GPU and runs in real
time on CPU.

**Three-camera fusion:** `object_detector_dbscan` transforms each raw camera
cloud into `base_link`, caches the latest cloud per camera, then concatenates
them before ROI cropping. The offline workflow assumes the arm is clear, so
`robot_mask_filter` is not part of the default pipeline. This reduces
occlusion (each camera's viewpoint covers extremities the others miss) and
increases point density. See [Launching](#launching) to disable fusion for an
individual camera.

**Limitations:** Fails when two objects touch or have similar depth profiles,
because their points merge into a single cluster with no spatial gap to split
on — see the SAM backend below for an alternative that doesn't share this
failure mode.

### Automatic SAM (MobileSAM)

Segments the color image with a promptable vision model (Segment Anything,
"segment everything" mode) instead of clustering 3D points, then back-projects
each mask through the aligned depth image into 3D. Since it splits on visual
appearance/boundary rather than spatial gap, two touching objects with a
visible seam aren't a special case for it — DBSCAN's core limitation above.
Added for side-by-side comparison against DBSCAN, not as a replacement; both
can run at once (see [Launching](#launching)), on separate topic namespaces.

Model is MobileSAM (`sam_model_type: vit_t`, the default) — same API as
Meta's original SAM, but a ~40M-param TinyViT encoder instead of ViT-H's
632M, because this machine is CPU-only and full SAM is impractically slow
there. Full SAM checkpoints load through the same code path if this ever
runs on a GPU. No realtime requirement, so it re-segments at most once every
`min_reseg_interval_sec` (default 5s) while active, rather than on every
incoming frame — see [`object_detector_sam.py`](irb120_perception/object_detector_sam.py)'s
docstring for the full design rationale and current limitations (single
camera only, class-agnostic masks).

The automatic backend runs independently on each configured camera and merges
matching 3-D instances. It is intentionally an experimental comparison
backend: automatic point-grid mask generation is slow on CPU.

### DBSCAN → prompted SAM cull

This is the preferred offline path when DBSCAN gets the object broadly right
but retains a depth-edge/flying-pixel blob. DBSCAN first produces its normal
object cloud. The culler projects that cloud into cam1, uses its 2-D bounds as
one MobileSAM box prompt, and retains only DBSCAN points inside the resulting
mask. It avoids full-image "segment everything" inference; a live run
measured about 300 ms for DBSCAN plus about 2 s for the cull on CPU.

It is comparison-safe: it never modifies `/object_detector/*` or contact
selection automatically. Review `/object_detector_dbscan_sam_cull/*` in RViz
before choosing it for a consumer. It cannot remove a bad depth point that
falls inside the object's visible 2-D silhouette from cam1.

SAM variants require pip packages not declared in `package.xml` (not rosdep-known) and a
downloaded checkpoint — see that module's docstring for exact commands
before trying to run it.

**Always launch this via `perception_sam.launch.py`, never a bare `ros2 run`.**
irb120_perception is built with the workspace's normal system Python — on
purpose, because the primary DBSCAN detector runs near a live, 250 Hz EGM
control loop and must not depend on a pip-heavy venv's Python. The launch file
routes only this node through `~/irb_venv`'s interpreter and caps BLAS/torch
to eight threads. Skipping the launch file
either fails the torch import (system Python) or, if you improvise your own
venv invocation without the same thread caps, risks the exact failure this
setup exists to avoid: a pip-installed numpy's bundled OpenBLAS defaults to
one thread pool per core *per call*, and this package's nodes call it at
~90 Hz — building the whole package under venv Python once did this to
the continuously-running perception stack and knocked out EGM on live hardware.

---

## Pipeline

```
/realsense/depth/color/points   (PointCloud2, ~30 Hz)  ─┐
/realsense2/depth/color/points  (PointCloud2, ~30 Hz)  ─┤  fused in object_detector:
/realsense3/depth/color/points  (PointCloud2, ~30 Hz)  ─┤  TF → base_link (per camera),
                                                          │  concatenate latest clouds
                                                          ▼
                        raw fused cloud  (base_link, all configured cameras)
        │
        ▼
┌─────────────────────┐
│  TF transform       │  camera frame → base_link (per camera)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ROI crop           │  discard points outside the workspace box
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Table-plane reject  │  fitted horizontal plane removes support-surface points
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Temporal consensus  │  retain voxels seen in 12 of the last 20 observations
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Voxel downsample   │  one point per voxel cell → uniform density;
└─────────────────────┘  also collapses cam1/cam2 overlap into single points
        │
        ▼
┌─────────────────────┐
│  DBSCAN clustering  │  group points by spatial proximity
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Per cluster                                │
│   • 3D convex hull  (scipy)                 │
│   • Centroid        (mean of cluster pts)   │
│   • Orientation     (PCA principal axes)    │
└─────────────────────────────────────────────┘
        │
        ├──▶  ~/detections   (vision_msgs/Detection3DArray)
        └──▶  ~/markers      (visualization_msgs/MarkerArray)
```

The ROI provides a first coarse crop; a fitted near-horizontal plane is then
removed before temporal accumulation and clustering.

---

## Launching

```bash
# Fuses all three raw camera clouds; no robot mask is required offline
ros2 launch irb120_perception perception.launch.py

# Camera 1 only (disable fusion for cameras 2 and 3)
ros2 launch irb120_perception perception.launch.py cam2_cloud_topic:='' cam3_cloud_topic:=''
```

Or via the full bringup:

```bash
ros2 launch irb120_control bringup_irb120_moveit.launch.py
```

`ros2 run irb120_perception object_detector_dbscan` also works directly if
you want the backend running standalone outside the launch file (e.g.
against a bag).

### Prompted SAM cull (recommended comparison)

```bash
ros2 launch irb120_perception perception.launch.py
ros2 launch irb120_perception perception_dbscan_sam_cull.launch.py \
  sam_checkpoint:=/path/to/mobile_sam.pt
```

The second launch consumes DBSCAN's `/object_detector/object_points` and cam1
RGB/camera-info. With both running, compare:

| | DBSCAN | DBSCAN → SAM cull |
|---|---|---|
| Detections | `/object_detector/detections` | `/object_detector_dbscan_sam_cull/detections` |
| Markers | `/object_detector/markers` | `/object_detector_dbscan_sam_cull/markers` |
| Object points | `/object_detector/object_points` | `/object_detector_dbscan_sam_cull/object_points` |

---

## Parameters

All parameters are set in [`launch/perception.launch.py`](launch/perception.launch.py).

### Region of Interest (ROI)

Defined in `base_link` frame (metres). Points outside this box are discarded
before any processing.

| Parameter   | Default   | Effect |
|-------------|-----------|--------|
| `roi_x_min` | `0.15`    | Near edge (toward robot). Raise to ~0.3 to exclude the robot body. |
| `roi_x_max` | `0.80`    | Far edge of workspace. |
| `roi_y_min` | `-0.25`   | Left edge. |
| `roi_y_max` | `0.25`    | Right edge. |
| `roi_z_min` | `-0.015`  | Height floor. Set just above the table surface in base_link Z. |
| `roi_z_max` | `0.50`    | Height ceiling. |

### Voxel Downsampling

| Parameter    | Default | Effect |
|--------------|---------|--------|
| `voxel_size` | `0.005` | Grid cell size in metres. Larger = faster but coarser hull. `0.01` is a good trade-off for speed. |

### Temporal Accumulation

The robot is stationary and out of frame while this runs, so a point that
doesn't recur across frames is noise, not signal — this denoises using that
directly, rather than only a single frame's local density. See
`FrameAccumulator`'s docstring in `perception_common.py`.

**Enabled by default in `perception.launch.py`: 20 observations, 12 hits.**
The default launch now consumes direct camera clouds rather than the slow
robot-mask pipeline, so this warm-up is short during offline operation and
reliably rejects flickering edge/free-space depth samples. The scene must be
stationary while the window fills.

| Parameter          | Default | Effect |
|---------------------|---------|--------|
| `accum_frames`      | `20` | Sliding-window length. `1` disables accumulation. |
| `accum_min_hits`    | `12` | Minimum observations in which a voxel must recur. Raise for stronger denoising (may trim faint/thin edges); lower to retain marginal edges at the cost of more noise. |

While the window is filling after activation, `object_detector` publishes
nothing at all (not `~/detections`, not `~/object_points`) — this matters
because `press_point_check.check_press_point()` waits for and uses the
*first* message it receives after activating; publishing an empty result
during warm-up would look identical to "no object detected" to that caller.
The same silence is indistinguishable from a hang if the window never fills
in a reasonable time, which is exactly what happened with the old default.

### DBSCAN

| Parameter          | Default | Effect |
|--------------------|---------|--------|
| `dbscan_eps`       | `0.02`  | Neighbourhood radius (m). Raise if one object splits into multiple clusters. Lower if two objects merge. |
| `dbscan_min_pts`   | `20`    | Minimum points to form a cluster core. Raise to suppress noise clusters. |
| `min_cluster_pts`  | `30`    | Discard clusters with fewer points than this. |
| `max_cluster_pts`  | `50000` | Discard clusters larger than this (catches robot body leaking into ROI). |
| `single_object_mode` | `True` | Union every surviving cluster into one object. For a single rigid item that comes back as multiple disconnected clusters (e.g. a monitor's base+screen, joined only at a rear seam no camera can see). Only safe if the workspace holds one physical item per detection cycle — otherwise this wrongly fuses genuinely separate objects. |
| `outlier_k`        | `8`     | Neighbours sampled per point for the local-density check, run before clustering. Strips depth-camera "flying pixel" noise trails that DBSCAN's single-linkage chaining would otherwise absorb into a real object's cluster. See [contact_point_selector](CONTACT_SELECTION.md) — this is what was throwing off downstream contact selection. |
| `outlier_std_ratio` | `2.0`  | How many std-devs above the mean k-NN distance counts as "sparse" and gets dropped. Lower = more aggressive removal (risks trimming real sparse object edges); `0` disables the check entirely. |
| `table_plane_distance` | `0.008` | Distance from the fitted horizontal tabletop plane (m) considered table and removed before clustering. |
| `smooth_alpha`     | `0.3`   | EMA weight for temporal smoothing. `0` = frozen (previous frame), `1` = raw (no smoothing). Lower values reduce jitter but add lag. |

---

## Outputs

### `~/detections` — `vision_msgs/Detection3DArray`

One `Detection3D` per detected object, in `base_link` frame.

| Field | Content |
|-------|---------|
| `bbox.center.position` | Centroid (mean of cluster points) |
| `bbox.center.orientation` | PCA orientation — X axis = longest dimension |
| `bbox.size` | Axis-aligned bounding box extents |
| `results[0].pose` | Same centroid + orientation |
| `id` | Integer index assigned this frame (not persistent across frames) |

### `~/markers` — `visualization_msgs/MarkerArray`

Visualisation for RViz. Add a **MarkerArray** display subscribed to `/object_detector/markers`.

| Namespace  | Type       | Content |
|------------|------------|---------|
| `hull`     | LINE_LIST  | Convex hull wireframe (triangulated edges) |
| `centroid` | SPHERE     | Centroid position |
| `axes`     | ARROW ×3   | PCA principal axes — Red=X (longest), Green=Y, Blue=Z |

Markers expire after 3 s so they disappear cleanly if detection stops.

---

## All topics at a glance

| Topic | Type | Produced by | QoS | Notes |
|-------|------|-------------|-----|-------|
| `/realsense[2|3]/depth/color/points` | PointCloud2 | RealSense | Best Effort | Raw DBSCAN inputs; each is transformed/fused in `object_detector` |
| `/object_detector/detections` | Detection3DArray | `object_detector` | Reliable | |
| `/object_detector/markers` | MarkerArray | `object_detector` | Reliable | |
| `/object_detector/object_points` | PointCloud2 (x,y,z,label) | `object_detector` | Reliable | Input to `contact_point_selector` — see [CONTACT_SELECTION.md](CONTACT_SELECTION.md) |
| `/object_detector_dbscan_sam_cull/detections` | Detection3DArray | `object_detector_dbscan_sam_cull` | Reliable | Optional prompted-SAM refinement of DBSCAN |
| `/object_detector_dbscan_sam_cull/markers` | MarkerArray | `object_detector_dbscan_sam_cull` | Reliable | Optional prompted-SAM refinement visualisation |
| `/object_detector_dbscan_sam_cull/object_points` | PointCloud2 (x,y,z,label) | `object_detector_dbscan_sam_cull` | Reliable | SAM-culled DBSCAN cloud; comparison output, not selected by contact code by default |

### Troubleshooting: topic is in `ros2 topic list`, but RViz shows nothing

The most common cause: **QoS mismatch (Best Effort vs Reliable).** RViz's default PointCloud2/Image
   display requests `Reliable` unless you override it. If the publisher is
   `Best Effort`, the subscription is incompatible and silently receives
   nothing — no error dialog, just a permanently empty display. Check with:
   ```bash
   ros2 topic info <topic> --verbose   # look at "Reliability:" under QoS profile
   ```
   Fix: either set the display's **Topic → Reliability Policy** to `Best Effort`
   in RViz, or (better, so it's not a manual step every time) make the
   publisher `Reliable` if nothing about it needs Best Effort's tradeoffs.

**Reliable vs. Best Effort, briefly:** Reliable is like TCP — the publisher
keeps a message around and retries until the subscriber acks it, so nothing
is ever silently dropped. Best Effort is like UDP — fire-and-forget, no
retry; if a message doesn't make it, it's just gone. Best Effort exists for
high-rate sensor firehoses (e.g. the RealSense driver's raw 30 Hz streams)
where retrying a stale frame is pointless — the next frame is only ~33ms
away regardless, and retry/ack bookkeeping under load risks a growing
backlog, which is *worse* for latency than just dropping the occasional
frame. That's a real cost of Reliable, not a myth — but it only bites when
messages are large *and* frequent *and* something is otherwise struggling to
keep up. The RealSense raw streams stay Best Effort; detector outputs are
Reliable, so RViz can subscribe with its default QoS. A Best-Effort subscriber
can read a Reliable publisher; the incompatible direction is a Reliable
subscriber consuming a Best-Effort publisher.

If Reliable seemed "faster" when you tried it, that wasn't really a speed
comparison — Best Effort with a mismatched Reliable subscriber delivers
*zero* messages, so switching to Reliable went from "nothing arrives" to
"everything arrives," which will always look like a win over doing nothing.
On this same machine, at this data rate, there's no meaningful per-message
latency difference between the two once QoS is actually compatible.

---

## Press/contact point selection

Contact point selection (picking where on the detected object to press,
push, or tip from) now lives entirely in `contact_point_selector.py` — see
[CONTACT_SELECTION.md](CONTACT_SELECTION.md) for the selection strategy,
outputs, and parameters. It's invoked directly by
`irb120_control/util/press_point_check.py` (`select_contact_points()`), not
launched as a standalone persistent node.

`object_detector` publishes `~/object_points`
(`sensor_msgs/PointCloud2`, fields `x,y,z,label:int32`) — every detected
object's raw cluster points in one cloud, tagged with the same integer id
used for `Detection3D.id`. This lets `contact_point_selector` recover
per-object geometry from one topic without redoing segmentation.

(The older `press_point_selector.py` node, which used a simpler
nearest-X/top-most heuristic, has been removed — `contact_point_selector`
superseded it. See CONTACT_SELECTION.md for the migration notes.)

---

## RealSense configuration

Shared settings are in [`realsense_common.yaml`](../irb120_handeye/config/realsense_common.yaml),
loaded by each camera bringup wrapper.

| Setting | Value | Notes |
|---------|-------|-------|
| `pointcloud.enable` | `true` | Publishes the raw clouds consumed by DBSCAN. |
| `align_depth.enable` | `true` | Depth pixels aligned to color image frame |
| `decimation_filter.enable` | `false` | Disabled to preserve full native resolution |
| `spatial_filter.enable` | `true` | Magnitude 1, alpha 0.5, delta 5. |
| `temporal_filter.enable` | `true` | Alpha 0.7, delta 35, persistency "Valid in 2/last 4". |

The current configuration intentionally has `decimation_filter.enable: false`
and `temporal_filter.holes_fill: 0`, preserving native edge detail. These
camera-side settings do not fully remove stereo depth-edge artifacts; the
DBSCAN temporal consensus, table-plane rejection, and optional SAM cull are
the perception-side safeguards.

---

## Tuning guide

### DBSCAN

**One object splitting into multiple clusters:**
→ Increase `dbscan_eps` (try `0.03`, then `0.05`).

**Two adjacent objects merging into one cluster:**
→ Decrease `dbscan_eps` (try `0.015`). DBSCAN cannot separate touching
  objects without a spatial gap — use automatic SAM when visual instance
  separation is required.

**Too many small noise clusters:**
→ Increase `min_cluster_pts` and `dbscan_min_pts`.

**Table not fully excluded:**
→ The fitted table-plane rejection is enabled by default. Tune
  `table_plane_distance` before raising `roi_z_min`, which can trim an
  object's base.

**After enabling multi-camera fusion — visible seam/ghosting where clouds overlap:**
→ Extrinsic calibration error. Check `base -> realsense_link`, `base -> realsense2_link`,
  and `base -> realsense3_link` in RViz (same registration check the
  `record_depth_both.launch.py` RViz view was built for) before tuning anything else —
  a bad extrinsic can't be fixed by re-tuning DBSCAN params.

**After enabling multi-camera fusion — clusters merging or splitting differently than before:**
→ Point density increases in overlap regions, which can shift how `dbscan_eps`
  behaves there. Re-check `dbscan_eps`/`voxel_size` rather than assuming old values still hold.

**A single rigid object comes back as multiple detections (disconnected parts, e.g. a
monitor's base + screen joined only at an occluded rear seam):**
→ Set `single_object_mode: true` — only if the workspace holds one item at a time; it
  unions every surviving cluster into one object rather than trying to bridge the gap
  with `dbscan_eps` (which would risk merging genuinely separate objects elsewhere).

**A few stray points survive off to the side of the real object, throwing off
`contact_point_selector` downstream:**
→ Classic depth-camera "flying pixel" noise trailing off an object edge — DBSCAN's
  single-linkage chaining absorbs the trail into the real cluster one point-hop at a
  time even though it's sparse as a whole. Lower `outlier_std_ratio` (e.g. `1.5`) for
  more aggressive removal, or raise `outlier_k` (e.g. `12`) for a more stable density
  estimate. If real object points near a genuinely sparse edge start getting trimmed
  too, raise `outlier_std_ratio` back up instead.

**Processing too slow:**
→ Increase `voxel_size` to `0.01`.

**Noisy/spurious points survive even with `remove_sparse_outliers` tuned aggressively:**
→ That filter only sees one frame; a noise cluster that's locally dense
  *within* a frame passes it easily. Increase `accum_frames` (and/or
  `accum_min_hits`) instead — real geometry recurs across frames, most noise
  doesn't. Measure your actual input rate first (`ros2 topic hz
  <input_cloud_pc>`) and budget `accum_frames`/that rate seconds of latency
  before the first detection — see "Temporal Accumulation" above; do not
  assume camera rate.

---

## Dependencies

| Package | Used for |
|---------|----------|
| `numpy`, `scipy` | Point cloud math, convex hull (`scipy.spatial.ConvexHull`) |
| `scikit-learn` | DBSCAN clustering |
| `vision_msgs`, `visualization_msgs` | ROS 2 message types |
| `tf2_ros` | Point cloud transform to `base_link` |
| `torch` (CPU build), `mobile_sam` | SAM backend only — not in `package.xml` (pip, not rosdep-known); see [`object_detector_sam.py`](irb120_perception/object_detector_sam.py)'s docstring for install + checkpoint commands. DBSCAN and the rest of this package do not need either. |
