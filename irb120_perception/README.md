# irb120_perception

Object detection for the IRB120 robot workspace. The node subscribes to
RealSense camera streams, isolates objects on the workspace surface, and
publishes their 3D convex hulls, centroids, and orientations.

Two segmentation backends are available and selected at launch time:

- **DBSCAN** — pure geometry, no GPU required, fast
- **SAM** — vision-based using SAM 2, GPU required, handles adjacent/touching objects

---

## Backends

### DBSCAN (default)

Clusters the 3D pointcloud spatially using DBSCAN. Works well when objects
are clearly separated by a gap in 3D space. Requires no GPU and runs in real
time on CPU.

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
/realsense/depth/color/points  (PointCloud2, ~30 Hz)
        │
        ▼
┌─────────────────────┐
│  TF transform       │  camera frame → base_link
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ROI crop           │  discard points outside the workspace box
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Voxel downsample   │  one point per voxel cell → uniform density
└─────────────────────┘
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

```bash
# DBSCAN (default, no GPU needed)
ros2 launch irb120_perception perception.launch.py method:=dbscan

# SAM (requires CUDA GPU and weights)
ros2 launch irb120_perception perception.launch.py method:=sam
```

Or via the full bringup:

```bash
ros2 launch irb120_control bringup_irb120_moveit.launch.py perception_method:=sam
```

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

## RealSense configuration

Configured in [`bringup_irb120_moveit.launch.py`](../irb120_control/launch/bringup_irb120_moveit.launch.py).

| Setting | Value | Notes |
|---------|-------|-------|
| `depth_module.depth_profile` | `640x480x30` | Matched to color resolution — avoids scaling artefacts in aligned depth |
| `rgb_camera.color_profile` | `640x480x30` | |
| `align_depth.enable` | `true` | Depth pixels aligned to color image frame |
| `decimation_filter.enable` | `true` | 2×2 hardware averaging before output — reduces depth noise with negligible latency |
| `spatial_filter.enable` | `false` | Would improve edge quality but adds pipeline lag; replaced by software median blur |
| `temporal_filter.enable` | `false` | Would reduce temporal noise but adds lag; replaced by software EMA smoothing |

The RealSense spatial and temporal post-processing filters were found to introduce
significant latency. Equivalent denoising is applied in software:
- **Spatial noise** → `cv2.medianBlur` on the raw depth image (configurable via `depth_median_ksize`)
- **Temporal noise** → EMA smoothing on the detected centroid across frames (configurable via `smooth_alpha`)

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
