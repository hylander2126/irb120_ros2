# irb120_perception

Object detection for the IRB120 robot workspace. Subscribes to RealSense
camera streams, isolates objects on the workspace surface, and publishes
their 3D convex hulls, centroids, and orientations.

Two segmentation backends are available as **separate nodes/executables**
(different runtime deps, selected at launch time via the `method` arg, which
picks which one actually runs — see [Launching](#launching)):

- **`object_detector_dbscan`** — pure geometry, no GPU required, fast. Also
  the only backend that fuses both cameras (see below).
- **`object_detector_sam`** — vision-based using SAM 2, GPU required, handles
  adjacent/touching objects. Single camera only.

Both share plumbing (TF, ROI crop, PCA orientation, convex hull, Detection3D/
marker publishing, EMA smoothing) via `perception_common.py`, so the shared
logic isn't duplicated even though the backends are split into different files.

---

## Backends

### DBSCAN (default)

Clusters the 3D pointcloud spatially using DBSCAN. Works well when objects
are clearly separated by a gap in 3D space. Requires no GPU and runs in real
time on CPU.

**Two-camera fusion:** `robot_mask_filter` transforms both cameras' clouds
into `base_link` (using the existing eye-to-hand extrinsics), robot-masks
each, concatenates them, and publishes one fused cloud — `object_detector_dbscan`
itself has no camera-count awareness. This reduces occlusion (camera 2's
portrait orientation covers the extremities of tall objects camera 1 misses)
and increases point density. See [robot_mask_filter's docstring](irb120_perception/robot_mask_filter.py)
for the fusion details, and [Launching](#launching) to disable it.

**Limitations:** Fails when two objects touch or have similar depth profiles,
because their points merge into a single cluster with no spatial gap to split on.

### SAM (Segment Anything Model 2)

Segments the RGB image with SAM 2, then back-projects each mask into 3D
using the aligned depth image and camera intrinsics. Objects are distinguished
visually (colour, texture, edges) rather than spatially, so touching or
adjacent objects are handled correctly.

**When to prefer SAM over DBSCAN:**
- Objects are touching or have gaps smaller than `dbscan_eps`
- Objects share a similar depth profile (e.g. stacked or flat items)
- The scene is complex and spatial clustering produces too many false splits/merges
- Object identity matters more than raw speed

**Requirements:** CUDA GPU, SAM 2 weights (see [Setup](#setup)).

---

## Pipelines

### DBSCAN pipeline

```
/realsense/depth/color/points   (PointCloud2, ~30 Hz)  ─┐
/realsense2/depth/color/points  (PointCloud2, ~30 Hz)  ─┤  fused in robot_mask_filter:
                                                          │  TF → base_link (per camera),
                                                          │  robot-mask, concatenate
                                                          ▼
                              ~/points_masked_dbscan  (PointCloud2, base_link, both cameras)
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

### SAM pipeline

```
/realsense/color/image_raw          (Image, ~30 Hz)   ─┐
/realsense/aligned_depth_to_color/  (Image, ~30 Hz)   ─┤─▶  _depth_cloud_cb
/realsense/color/camera_info        (CameraInfo)      ─┘
        │
        ▼
┌──────────────────────────┐
│  Depth median blur       │  5×5 kernel — kills salt-and-pepper depth noise
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  SAM 2 (Hiera-T, GPU)    │  generates pixel-space segmentation masks on RGB
│  points_per_side = 8     │  (16×16 → 8×8 grid reduces prompt count ~4×)
│  iou_thresh = 0.85       │
│  min_mask_area = 1000 px │
└──────────────────────────┘
        │  N masks
        ▼
┌──────────────────────────┐
│  Back-projection         │  pixel (u,v) + depth → XYZ in camera frame
│  (pinhole model)         │  using camera intrinsics (fx, fy, cx, cy)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  TF transform            │  camera frame → base_link
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  ROI filter              │  discard points outside the workspace box
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Voxel downsample        │  uniform density, matches DBSCAN path
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Statistical outlier     │  remove points > 2σ from cluster centroid
│  removal                 │  prevents stray depth pixels from spiking the hull
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Prominent object select │  keep only the largest mask by pixel area
│  (sam_prominent_only)    │  (disable to publish all detected objects)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  EMA temporal smoothing  │  centroid position smoothed across frames
│  (smooth_alpha = 0.3)    │  point cloud shifted to match — kills jitter
└──────────────────────────┘
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

---

## Launching

`method` selects which backend node (`object_detector_dbscan` or
`object_detector_sam`) actually runs; the other stays defined but disabled
via a launch `IfCondition`.

```bash
# DBSCAN (default, no GPU needed) — fuses both cameras
ros2 launch irb120_perception perception.launch.py method:=dbscan

# DBSCAN, camera 1 only (disable fusion)
ros2 launch irb120_perception perception.launch.py method:=dbscan cam2_cloud_topic:=''

# SAM (requires CUDA GPU and weights) — camera 1 only, no fusion
ros2 launch irb120_perception perception.launch.py method:=sam
```

Or via the full bringup:

```bash
ros2 launch irb120_control bringup_irb120_moveit.launch.py perception_method:=sam
```

Both `ros2 run irb120_perception object_detector_dbscan` and
`... object_detector_sam` also work directly if you want a backend running
standalone outside the launch file (e.g. against a bag).

---

## Setup

### SAM 2 weights

```bash
mkdir -p ~/.local/share/irb120_perception/weights
# Download sam2.1_hiera_tiny.pt from the SAM 2 releases page and place it there
```

### Python venv (SAM only)

SAM requires torch + CUDA. A separate venv is used to keep these off the system Python:

```bash
~/.venvs/.venv_torch_SAM/bin/python3
```

The launch file injects this venv's site-packages via `PYTHONPATH` automatically.

---

## Parameters

All parameters are set in [`launch/perception.launch.py`](launch/perception.launch.py).

### Shared — Region of Interest (ROI)

Defined in `base_link` frame (metres). Points outside this box are discarded
before any processing in both backends.

| Parameter   | Default   | Effect |
|-------------|-----------|--------|
| `roi_x_min` | `0.15`    | Near edge (toward robot). Raise to ~0.3 to exclude the robot body. |
| `roi_x_max` | `0.80`    | Far edge of workspace. |
| `roi_y_min` | `-0.25`   | Left edge. |
| `roi_y_max` | `0.25`    | Right edge. |
| `roi_z_min` | `-0.015`  | Height floor. Set just above the table surface in base_link Z. |
| `roi_z_max` | `0.50`    | Height ceiling. |

### Shared — Voxel Downsampling

| Parameter    | Default | Effect |
|--------------|---------|--------|
| `voxel_size` | `0.005` | Grid cell size in metres. Larger = faster but coarser hull. `0.01` is a good trade-off for speed. |

### DBSCAN-specific

| Parameter          | Default | Effect |
|--------------------|---------|--------|
| `dbscan_eps`       | `0.02`  | Neighbourhood radius (m). Raise if one object splits into multiple clusters. Lower if two objects merge. |
| `dbscan_min_pts`   | `20`    | Minimum points to form a cluster core. Raise to suppress noise clusters. |
| `min_cluster_pts`  | `30`    | Discard clusters with fewer points than this. |
| `max_cluster_pts`  | `50000` | Discard clusters larger than this (catches robot body leaking into ROI). |
| `single_object_mode` | `False` | Union every surviving cluster into one object. For a single rigid item that comes back as multiple disconnected clusters (e.g. a monitor's base+screen, joined only at a rear seam no camera can see). Only safe if the workspace holds one physical item per detection cycle — otherwise this wrongly fuses genuinely separate objects. |

### SAM-specific

| Parameter              | Default | Effect |
|------------------------|---------|--------|
| `sam_weights`          | *(set by launch)* | Path to `sam2.1_hiera_tiny.pt`. |
| `sam_config`           | `configs/sam2.1/sam2.1_hiera_t.yaml` | SAM 2 model config. |
| `sam_points_per_side`  | `8`     | Grid density for SAM prompt generation. Lower = faster, fewer masks. `16` for better recall on small objects. |
| `sam_iou_thresh`       | `0.85`  | Minimum predicted IoU to keep a mask. Raise to reduce false positives. |
| `sam_min_mask_area`    | `1000`  | Minimum mask size in pixels. Filters noise and background texture. |
| `sam_min_cluster_pts`  | `30`    | Minimum 3D points after back-projection for a mask to be kept. |
| `sam_prominent_only`   | `True`  | Keep only the largest mask (by pixel area). Set `False` to detect all objects. |
| `depth_median_ksize`   | `5`     | Kernel size for depth image median blur. `0` to disable. Must be odd. |
| `outlier_std_ratio`    | `2.0`   | Remove 3D points further than this many standard deviations from the cluster centroid. Lower = more aggressive removal. |
| `smooth_alpha`         | `0.3`   | EMA weight for temporal smoothing. `0` = frozen (previous frame), `1` = raw (no smoothing). Lower values reduce jitter but add lag. |

---

## Outputs

### `~/detections` — `vision_msgs/Detection3DArray`

One `Detection3D` per detected object, in `base_link` frame.

| Field | Content |
|-------|---------|
| `bbox.center.position` | Centroid (mean of cluster points, EMA-smoothed in SAM mode) |
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
| `/robot_mask_filter/points_masked_dbscan` | PointCloud2 | `robot_mask_filter` | **Reliable** | DBSCAN input, robot body removed, both cameras fused into `base_link` |
| `/robot_mask_filter/depth_masked_sam` | Image | `robot_mask_filter` | **Reliable** | SAM input, robot body removed |
| `/object_detector/detections` | Detection3DArray | `object_detector` | Reliable | Both backends |
| `/object_detector/markers` | MarkerArray | `object_detector` | Reliable | Both backends |
| `/object_detector/object_points` | PointCloud2 (x,y,z,label) | `object_detector` | Reliable | Both backends; input to `press_point_selector` |
| `/press_point_selector/press_pose` | PoseStamped | `press_point_selector` | Reliable | On-demand, see [Press point selection](#press-point-selection) |
| `/press_point_selector/press_marker` | MarkerArray | `press_point_selector` | Reliable | On-demand |
| `/object_detector/debug/sam_mask_overlay` | Image | `object_detector` | Reliable | **SAM only**, on-demand |
| `/object_detector/debug/sam_pts_camera` | PointCloud2 | `object_detector` | Reliable | **SAM only**, on-demand |
| `/object_detector/debug/sam_pts_after_roi` | PointCloud2 | `object_detector` | Reliable | **SAM only**, on-demand |
| `/object_detector/debug/sam_pts_after_clean` | PointCloud2 | `object_detector` | Reliable | **SAM only**, on-demand |

The "SAM only" topics have a publisher registered from startup regardless of
backend, but nothing is ever actually published to them unless
`segmentation_method: 'sam'` is active **and** a snapshot is triggered:

```bash
ros2 topic pub --once /object_detector/sam_debug_snapshot std_msgs/msg/Empty '{}'
```

Under DBSCAN they show up in `ros2 topic list` (the publisher exists) but
will never carry data — that's expected, not broken.

### Troubleshooting: topic is in `ros2 topic list`, but RViz shows nothing

Two independent causes, both look identical from RViz:

1. **It's a SAM-only debug topic and you're running DBSCAN** — see above.
2. **QoS mismatch (Best Effort vs Reliable).** RViz's default PointCloud2/Image
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

## Press point selection

`press_point_selector` picks a single 3D "press point" on the detected
object: the highest, closest-to-camera point, biased toward the lateral
middle of the object rather than an extreme corner. Useful for a
single-finger poke/press action.

**Heuristic** (per point of the target object, higher = better):

```
score = w_height * height_score      (higher Z = better)
      + w_camera * camera_score      (closer to the camera = better)
      + w_center * center_score      (closer to the object's lateral
                                       centroid, measured in the plane
                                       perpendicular to the camera's
                                       view direction = better)
```

The final point is the mean of the top `top_fraction` scoring points (not
a single raw point) — this smooths sensor noise and lands near the middle
of the qualifying near-top region instead of on one noisy outlier point.

**On-demand only.** The node continuously caches the latest
`~/object_points` message but does **not** recompute every frame — it only
runs the heuristic when its service is called, so the target doesn't
drift while the arm is mid-approach:

```bash
ros2 service call /press_point_selector/compute_press_point std_srvs/srv/Trigger {}
```

The last computed result is re-published at `publish_rate_hz` (default
5 Hz) purely to keep RViz's preview fresh between triggers — that
republish does not touch the point cloud or recompute anything.

**Multi-object scenes:** `target_object_id` defaults to `-1` (auto-select
the object with the highest mean Z, i.e. the "prominent" object, same
convention as `sam_prominent_only`). Set it to a specific detection id to
target a different object instead.

### Outputs

| Topic/Frame | Type | Content |
|---|---|---|
| `~/press_pose` | `geometry_msgs/PoseStamped` | Position = press point in `base_frame`. Orientation Z axis = approach direction (camera → point). |
| `~/press_marker` | `visualization_msgs/MarkerArray` | Magenta sphere at the point + arrow along the approach direction. |
| TF: `base_frame` → `press_frame_id` | — | Broadcast continuously (from cached result) so it stays lookup-able in RViz/MoveIt between triggers. |

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `input_points` | `/object_detector/object_points` | Labeled per-object cloud consumed (see below). |
| `camera_frame` | `realsense_color_optical_frame` | TF frame used to compute "closest to camera". |
| `target_object_id` | `-1` | `-1` = auto-select prominent object; else a specific detection id. |
| `top_fraction` | `0.12` | Fraction of best-scoring points averaged into the final point. |
| `min_top_points` | `5` | Floor on how many points are averaged, even for small objects. |
| `w_height` / `w_camera` / `w_center` | `1.0` / `1.0` / `0.5` | Heuristic term weights. |
| `publish_rate_hz` | `5.0` | Rate at which the cached result is re-published for RViz. |

### New `object_detector` output this depends on

`object_detector` now also publishes `~/object_points`
(`sensor_msgs/PointCloud2`, fields `x,y,z,label:int32`) — every detected
object's raw cluster points in one cloud, tagged with the same integer id
used for `Detection3D.id`. This lets `press_point_selector` (or any other
consumer) recover per-object geometry from one topic without redoing
segmentation.

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
software. `object_detector_sam.py` still applies its own `cv2.medianBlur`
(`depth_median_ksize`) on the depth image, and both backends apply EMA
smoothing (`smooth_alpha`) on top of this — worth checking whether that's
now double-filtering before tuning either side further.

---

## Tuning guide

### DBSCAN

**One object splitting into multiple clusters:**
→ Increase `dbscan_eps` (try `0.03`, then `0.05`).

**Two adjacent objects merging into one cluster:**
→ Decrease `dbscan_eps` (try `0.015`).
→ Switch to SAM if the objects are touching — DBSCAN cannot separate them without a spatial gap.

**Too many small noise clusters:**
→ Increase `min_cluster_pts` and `dbscan_min_pts`.

**Table not fully excluded:**
→ Increase `roi_z_min` to sit clearly above the table surface.

**After enabling camera 2 fusion — visible seam/ghosting where the two clouds overlap:**
→ Extrinsic calibration error. Check `base -> realsense_link` and `base -> realsense2_link`
  in RViz (same registration check the `record_depth_both.launch.py` RViz view was built for)
  before tuning anything else — a bad extrinsic can't be fixed by re-tuning DBSCAN params.

**After enabling camera 2 fusion — clusters merging or splitting differently than before:**
→ Point density roughly doubles in the overlap region, which can shift how `dbscan_eps`
  behaves there. Re-check `dbscan_eps`/`voxel_size` rather than assuming old values still hold.

**A single rigid object comes back as multiple detections (disconnected parts, e.g. a
monitor's base + screen joined only at an occluded rear seam):**
→ Set `single_object_mode: true` — only if the workspace holds one item at a time; it
  unions every surviving cluster into one object rather than trying to bridge the gap
  with `dbscan_eps` (which would risk merging genuinely separate objects elsewhere).

### SAM

**Hull still jittery:**
→ Lower `smooth_alpha` (e.g. `0.15`) for stronger smoothing. Increases lag.
→ Lower `outlier_std_ratio` (e.g. `1.5`) to trim more fringe points from the mask edge.

**Hull does not fit the object well:**
→ Increase `sam_points_per_side` to `16` for denser mask generation.
→ Lower `outlier_std_ratio` if stray background points are inflating the hull.
→ Raise `roi_z_min` to exclude table-surface points leaking into the mask.

**Wrong object selected as prominent:**
→ Ensure the target object is the largest visible item in the scene.
→ Set `sam_prominent_only: False` and inspect all detected clusters in RViz to debug.

**Processing too slow:**
→ Reduce `sam_points_per_side` to `4` or `8`.
→ Raise `sam_min_mask_area` to skip small masks earlier.
→ A GPU upgrade has near-linear impact — SAM Hiera-T is fully GPU-bound.

### Both backends

**Processing too slow:**
→ Increase `voxel_size` to `0.01`.

---

## Dependencies

| Package | Used for |
|---------|----------|
| `numpy`, `scipy` | Point cloud math, convex hull (`scipy.spatial.ConvexHull`) |
| `scikit-learn` | DBSCAN clustering |
| `opencv-python` | Depth image median blur, colour conversion |
| `torch` + `sam2` | SAM 2 inference (SAM backend only, venv-installed) |
| `cv_bridge` | ROS Image ↔ numpy conversion |
| `vision_msgs`, `visualization_msgs` | ROS 2 message types |
| `tf2_ros` | Point cloud transform to `base_link` |
