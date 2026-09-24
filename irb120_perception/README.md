# irb120_perception

Object detection for the IRB120 robot workspace. Subscribes to RealSense
camera streams, isolates objects on the workspace surface, and publishes
their 3D convex hulls, centroids, and orientations.

The independent [contact point selector](CONTACT_SELECTION.md) selects planar
push, forward-tip and press contacts and estimates tipping axes from object clouds.

Two segmentation backends exist, sharing plumbing (TF, ROI crop, PCA
orientation, convex hull, Detection3D/marker publishing, EMA smoothing) via
`perception_common.py` (`ObjectDetectorBase`) so neither duplicates that logic:

- `object_detector_dbscan` — pure geometry, no GPU required, fast.
- `object_detector_sam` — vision-based (Segment Anything), added to compare
  against DBSCAN on objects DBSCAN can't segment (touching objects — see
  below). CPU-only capable but slower; not a replacement for DBSCAN.

---

## Backend

### DBSCAN

Clusters the 3D pointcloud spatially using DBSCAN. Works well when objects
are clearly separated by a gap in 3D space. Requires no GPU and runs in real
time on CPU.

**Three-camera fusion:** `robot_mask_filter` transforms all three cameras'
clouds into `base_link` (using the existing eye-to-hand extrinsics),
robot-masks each, concatenates them, and publishes one fused cloud —
`object_detector_dbscan` itself has no camera-count awareness. This reduces
occlusion (each camera's viewpoint covers extremities the others miss) and
increases point density. See [robot_mask_filter's docstring](irb120_perception/robot_mask_filter.py)
for the fusion details, and [Launching](#launching) to disable fusion for any
one camera.

**Limitations:** Fails when two objects touch or have similar depth profiles,
because their points merge into a single cluster with no spatial gap to split
on — see the SAM backend below for an alternative that doesn't share this
failure mode.

### SAM (MobileSAM)

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

Requires pip packages not declared in `package.xml` (not rosdep-known) and a
downloaded checkpoint — see that module's docstring for exact commands
before trying to run it.

**Always launch this via `perception_sam.launch.py`, never a bare `ros2 run`.**
irb120_perception is built with the workspace's normal system Python — on
purpose, because `robot_mask_filter`/`object_detector` run near a live,
250 Hz real-time EGM control loop and must not depend on a pip-heavy venv's
Python. The launch file routes only this node through `~/irb_venv`'s
interpreter and forces single-threaded BLAS/torch. Skipping the launch file
either fails the torch import (system Python) or, if you improvise your own
venv invocation without the same thread caps, risks the exact failure this
setup exists to avoid: a pip-installed numpy's bundled OpenBLAS defaults to
one thread pool per core *per call*, and this package's nodes call it at
~90 Hz — building the whole package under venv Python once did this to
`robot_mask_filter` and knocked out EGM on live hardware.

---

## Pipeline

```
/realsense/depth/color/points   (PointCloud2, ~30 Hz)  ─┐
/realsense2/depth/color/points  (PointCloud2, ~30 Hz)  ─┤  fused in robot_mask_filter:
/realsense3/depth/color/points  (PointCloud2, ~30 Hz)  ─┤  TF → base_link (per camera),
                                                          │  robot-mask, concatenate
                                                          ▼
                        ~/points_masked_dbscan  (PointCloud2, base_link, all three cameras)
        │
        ▼
┌─────────────────────┐
│  TF transform       │  base_link → base_link (identity; cloud already fused into base_link)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ROI crop           │  discard points outside the workspace box
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

Table removal is handled by `roi_z_min` set just above the known table height.
No RANSAC is needed because the table height is fixed in the robot base frame.

---

## Launching

```bash
# Fuses all three cameras
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

### SAM, alongside DBSCAN for comparison

```bash
ros2 launch irb120_perception perception.launch.py           # robot_mask_filter + DBSCAN, as above
ros2 launch irb120_perception perception_sam.launch.py sam_checkpoint:=/path/to/mobile_sam.pt
```

The second launch file adds only `object_detector_sam` — it depends on
`robot_mask_filter` already running from the first (see
[`perception_sam.launch.py`](launch/perception_sam.launch.py) for why it
doesn't start its own copy). With both running, compare:

| | DBSCAN | SAM |
|---|---|---|
| Detections | `/object_detector/detections` | `/object_detector_sam/detections` |
| Markers | `/object_detector/markers` | `/object_detector_sam/markers` |
| Object points | `/object_detector/object_points` | `/object_detector_sam/object_points` |

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

**Off (`accum_frames: 1`) by default, including in `perception.launch.py`.**
It was briefly enabled there (`accum_frames: 10`) on the assumption that
`robot_mask_filter` sustains something near camera rate; measured live under
continuous operation (`active_at_start` default) it's actually ~1-1.5 Hz
with heavy jitter, since it's the most expensive node in the chain and was
designed to be toggled on briefly per check rather than run continuously
flat-out (see its own docstring's "On/off gate"). At that real rate,
`accum_frames=10` meant 10+ seconds of total silence on `~/object_points`
before the first detection ever appeared — indistinguishable from
segmentation being broken. Re-measure `ros2 topic hz <input_cloud_pc>` under
your actual run conditions before raising this above 1; it's much safer
around a single deliberate `press_point_check` activation (where a
several-second wait is already expected and budgeted) than for continuous
bringup viewing.

| Parameter          | Default | Effect |
|---------------------|---------|--------|
| `accum_frames`      | `1` (off) | Sliding window length. `1` disables accumulation (original single-frame behaviour). Naive latency estimate is `accum_frames`/publish-rate seconds before the first usable detection — but see the real-rate warning above before trusting that estimate. Keep well under `check_press_point`'s 5 s `timeout_sec` for whatever rate you actually measure. |
| `accum_min_hits`    | `0` (auto: ~60% of `accum_frames`) | Minimum distinct frames a voxel must be seen in to survive. Raise for more aggressive denoising (risks trimming a genuinely faint/thin object edge); lower to keep more of a marginal edge at the cost of more surviving noise. |

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
| `single_object_mode` | `False` | Union every surviving cluster into one object. For a single rigid item that comes back as multiple disconnected clusters (e.g. a monitor's base+screen, joined only at a rear seam no camera can see). Only safe if the workspace holds one physical item per detection cycle — otherwise this wrongly fuses genuinely separate objects. |
| `outlier_k`        | `8`     | Neighbours sampled per point for the local-density check, run before clustering. Strips depth-camera "flying pixel" noise trails that DBSCAN's single-linkage chaining would otherwise absorb into a real object's cluster. See [contact_point_selector](CONTACT_SELECTION.md) — this is what was throwing off downstream contact selection. |
| `outlier_std_ratio` | `2.0`  | How many std-devs above the mean k-NN distance counts as "sparse" and gets dropped. Lower = more aggressive removal (risks trimming real sparse object edges); `0` disables the check entirely. |
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
| `/robot_mask_filter/points_masked_dbscan` | PointCloud2 | `robot_mask_filter` | **Reliable** | DBSCAN input, robot body removed, all three cameras fused into `base_link` |
| `/object_detector/detections` | Detection3DArray | `object_detector` | Reliable | |
| `/object_detector/markers` | MarkerArray | `object_detector` | Reliable | |
| `/object_detector/object_points` | PointCloud2 (x,y,z,label) | `object_detector` | Reliable | Input to `contact_point_selector` — see [CONTACT_SELECTION.md](CONTACT_SELECTION.md) |

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
   publisher `Reliable` if nothing about it needs Best Effort's tradeoffs —
   that's exactly what was done for `robot_mask_filter`'s two output topics.

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
keep up. `robot_mask_filter`'s inputs stay Best Effort to match the camera
driver they subscribe to; its outputs (`points_masked_dbscan`,
`depth_masked_sam`) were switched to Reliable, which is safe here because a
Best-Effort subscriber (like `object_detector`) can always read from a
Reliable publisher — the incompatibility only runs the other direction
(Reliable subscriber vs. Best-Effort publisher).

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

Configured in [`bringup_stack.launch.py`](../irb120_control/launch/bringup_stack.launch.py).

| Setting | Value | Notes |
|---------|-------|-------|
| `depth_module.depth_profile` | `1280x720x30` | Max depth resolution on the D435 (top res caps at 30fps) |
| `rgb_camera.color_profile` | `1280x720x30` | Matched to depth resolution — avoids scaling artefacts in aligned depth |
| `align_depth.enable` | `true` | Depth pixels aligned to color image frame |
| `decimation_filter.enable` | `false` | Disabled to preserve full native resolution |
| `depth_module.hdr_enabled` / `hdr_merge.enable` | `true` | On-sensor HDR merge (alternating exposure/gain pairs) — see gotcha below |
| `disparity_filter.enable` | `true` | Wraps spatial/temporal in the disparity domain (Intel-recommended for filter quality) |
| `spatial_filter.enable` | `true` | Magnitude 2, smooth alpha 0.5, smooth delta 4, persistency disabled — tuned by hand in the RealSense Viewer |
| `temporal_filter.enable` | `true` | Smooth alpha 0.02, smooth delta 99, persistency "Valid in 2/last 4" — tuned by hand in the RealSense Viewer |

Fine-grained filter numbers live in [`realsense_filters.yaml`](../irb120_control/config/realsense_filters.yaml)
(passed via `rs_launch.py`'s `config_file` arg, since they aren't exposed as
top-level launch arguments in this realsense2_camera version).

Note: the D435 cannot exceed 30fps at this resolution — higher framerates (60/90fps) are only
available at 848x480 or lower. This config prioritizes resolution/accuracy over framerate.

**Gotcha (verified live against the D435):** HDR merge and a `visual_preset`
(e.g. "High Density") cannot both be active — the sensor throws "gain is
locked while HDR is active" and fails to start if you try. `visual_preset`
is intentionally omitted from `realsense_filters.yaml` for this reason.

Spatial and temporal filtering are now done on-device rather than in
software. `object_detector_dbscan` additionally applies EMA smoothing
(`smooth_alpha`) on top of this — worth checking whether that's now
double-filtering before tuning either side further.

---

## Tuning guide

### DBSCAN

**One object splitting into multiple clusters:**
→ Increase `dbscan_eps` (try `0.03`, then `0.05`).

**Two adjacent objects merging into one cluster:**
→ Decrease `dbscan_eps` (try `0.015`). DBSCAN cannot separate touching
  objects without a spatial gap — a vision-based backend would be needed
  for that case, and none is currently wired in.

**Too many small noise clusters:**
→ Increase `min_cluster_pts` and `dbscan_min_pts`.

**Table not fully excluded:**
→ Increase `roi_z_min` to sit clearly above the table surface.

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
