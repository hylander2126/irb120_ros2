# irb120_perception

Pointcloud-based object detection for the IRB120 robot workspace.
Subscribes to the RealSense depth pointcloud, isolates objects on the
workspace surface, and publishes their 3D convex hulls, centroids, and
orientations.

---

## Pipeline

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
│  ROI crop           │  keep only points inside the workspace box
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Voxel downsample   │  reduce point density for speed
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  RANSAC plane fit   │  find and remove the dominant flat surface (table)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  DBSCAN clustering  │  group remaining points into individual objects
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Per cluster                                │
│   • 3D convex hull  (Open3D)                │
│   • Centroid        (mean of cluster pts)   │
│   • Orientation     (PCA principal axes)    │
└─────────────────────────────────────────────┘
        │
        ├──▶  ~/detections   (vision_msgs/Detection3DArray)
        └──▶  ~/markers      (visualization_msgs/MarkerArray)
```

---

## Parameters

All parameters are set in [`launch/perception.launch.py`](launch/perception.launch.py).

### Region of Interest (ROI)
Defined in `base_link` frame (metres). Points outside this box are
discarded before any processing.

| Parameter   | Default | Effect |
|-------------|---------|--------|
| `roi_x_min` | `0.15`  | Near edge of workspace (toward robot). **Raise to ~0.3–0.4 to exclude the robot body.** |
| `roi_x_max` | `0.80`  | Far edge of workspace. |
| `roi_y_min` | `-0.25` | Left edge. |
| `roi_y_max` | `0.25`  | Right edge. |
| `roi_z_min` | `0.005` | Height floor above table. **Raise to 0.02–0.05 to skip the table surface entirely.** |
| `roi_z_max` | `0.50`  | Height ceiling. |

### Voxel Downsampling
| Parameter    | Default | Effect |
|--------------|---------|--------|
| `voxel_size` | `0.005` | Grid cell size in metres. Larger = faster but coarser hull. `0.01` is a good trade-off for speed. |

### RANSAC Plane Removal
Fits a plane to the dominant flat surface (the table) and removes all
inlier points. Only one plane is removed per frame.

| Parameter          | Default | Effect |
|--------------------|---------|--------|
| `ransac_distance`  | `0.01`  | Max distance (m) from the plane for a point to be counted as an inlier. **Raise to 0.02–0.03 if the table is not fully removed.** |
| `ransac_n`         | `3`     | Points sampled per RANSAC hypothesis. Leave at 3 (minimum for a plane). |
| `ransac_iters`     | `1000`  | Iterations. Lower for speed, higher for robustness on noisy data. |

### DBSCAN Clustering
Groups the remaining (non-table) points into individual object clusters
by spatial proximity.

| Parameter        | Default | Effect |
|------------------|---------|--------|
| `dbscan_eps`     | `0.02`  | Neighbourhood radius (m). **Raise if one object splits into multiple clusters. Lower if two objects merge into one.** |
| `dbscan_min_pts` | `20`    | Minimum points to form a cluster core. Raise to suppress noise clusters. |

### Cluster Size Filter
Clusters outside this point count range are discarded entirely.

| Parameter          | Default | Effect |
|--------------------|---------|--------|
| `min_cluster_pts`  | `30`    | Discard clusters smaller than this (noise, reflections). |
| `max_cluster_pts`  | `50000` | Discard clusters larger than this (catches the robot body if it leaks into the ROI). |

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
Visualisation for RViz. Add a **MarkerArray** display subscribed to
`/object_detector/markers`.

| Namespace  | Type       | Content |
|------------|------------|---------|
| `hull`     | LINE_LIST  | Convex hull wireframe (triangulated edges) |
| `centroid` | SPHERE     | Centroid position |
| `axes`     | ARROW ×3   | PCA principal axes — Red=X (longest), Green=Y, Blue=Z |

Markers expire after 0.5 s so they disappear cleanly if detection stops.

---

## Quick-start tuning guide

**Robot body appearing as a detected object:**
→ Increase `roi_x_min` until the robot base is outside the box.
→ Alternatively, increase `max_cluster_pts` to filter it out by size.

**Table not fully removed:**
→ Increase `ransac_distance` (try `0.02`, then `0.03`).
→ Increase `roi_z_min` to start the ROI above the table surface.

**One object splitting into multiple clusters:**
→ Increase `dbscan_eps` (try `0.03`, then `0.05`).

**Two adjacent objects merging into one cluster:**
→ Decrease `dbscan_eps` (try `0.015`).

**Too many small noise clusters:**
→ Increase `min_cluster_pts` and `dbscan_min_pts`.

**Processing too slow:**
→ Increase `voxel_size` to `0.01`.
→ Decrease `ransac_iters` to `500`.

---

## Dependencies

- `open3d` — plane segmentation, DBSCAN, convex hull (`pip install open3d`)
- `numpy`, `scipy` — already present in the workspace
- `vision_msgs`, `visualization_msgs` — standard ROS 2 Jazzy packages
- `tf2_ros` — pointcloud frame transform to `base_link`
