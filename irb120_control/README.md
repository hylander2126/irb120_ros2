# irb120_control

This package contains the ROS 2 bringup and control wiring for the real ABB IRB120 setup in this workspace.

## Important: local copy of vendor hardware interface

This setup depends on a local package named `irb120_abb_hardware_interface`, which is a copy/fork of the vendor package `abb_hardware_interface`.

The local copy is required for this project-specific behavior:

- Custom RWS credential pass-through from launch and xacro parameters.
- Local plugin identity used by this package (`irb120_abb_hardware_interface/ABBSystemHardware`).
- Avoiding direct edits to vendor packages in `src/abb_ros2/...`.

If `irb120_abb_hardware_interface` is missing, not built, or not sourced, real robot bringup from this package will fail.

## Notes

- This package intentionally keeps vendor packages unmodified.
- Bringup launch files in this package are tailored to this robot cell configuration.
