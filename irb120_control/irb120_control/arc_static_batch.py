#!/usr/bin/env python3
"""Automated batch of N arc_static trials in one episode — for objects that
settle back into a good pose on their own after each squash-arc-unarc-retract
cycle, so re-perceiving and re-confirming before every trial is pure overhead.

Differences from a normal `arc_static` run:
  - One perception snapshot up front (skipped if the episode's last stage
    already is one); every trial uses its press contact and pivot.
  - One operator confirmation, before trial 1. Between trials the arm returns
    to pre-squash (a real MoveIt move), not home, so there are no snapshots
    between trials -- the arm stays in the cameras' view.
  - Each trial still uses arc_static's retry-on-slip force escalation
    (run_adaptive_press), but the starting force CARRIES FORWARD from wherever
    the previous trial ended up. For an object like the monitor, whose nominal
    force is deliberately too low, only trial 1 pays for the escalation search.
  - Home once at the end, then a final snapshot.

Each trial is its own 'press_pull_tip' stage (own logs + videos, trial number
in the stage entry), so estimate_params sees N trials.

Usage:
    ros2 run irb120_control arc_static_batch box --trials 10
    ros2 run irb120_control arc_static_batch box --trials 10 --episode last --quality lossless
"""

import argparse
import sys
from datetime import datetime

import rclpy

from irb120_control.arc_static import ArcStatic, run_adaptive_press
from irb120_control.util.egm_client import deactivate_egm, ensure_egm_active
from irb120_control.util.episode import Episode
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.perception_snapshot import latest_contacts, take_snapshot
from irb120_control.util.runtime_log_dir import (
    VALID_OBJECTS,
    module_constants,
    save_ft_pose_log,
    save_run_metadata,
    start_recording,
    stop_recording,
)

LOG_PREFIX = "arc_static_batch"
DEFAULT_TRIALS = 10


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    parser = argparse.ArgumentParser(description="Run N arc_static trials back to back in one episode")
    parser.add_argument("object", choices=sorted(VALID_OBJECTS))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--quality", choices=["h264", "lossless"], default="h264")
    parser.add_argument("--episode", default="new", metavar="new|last|PATH",
                        help="Episode to record into (default: a new one). See util/episode.py.")
    parsed = parser.parse_args(raw_args[:ros_args_index])

    rclpy.init(args=raw_args[ros_args_index:])
    node = ArcStatic(parsed.object)
    episode = Episode.resolve(parsed.episode, parsed.object, node)
    stage = None
    motion_started = False
    home_ok = None
    try:
        if episode.last_stage_kind() != "perceive":
            take_snapshot(node, episode, label="before batch")
        contacts = latest_contacts(episode)
        if contacts is None:
            node.get_logger().error("No usable perception snapshot -- not moving.")
            return 1
        if not node.set_targets_from_contacts(contacts):
            return 1

        if not tare_netft(node):
            return 1
        if not ensure_egm_active(node):
            return 1
        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error("MoveIt Servo is not ready. Launch the stack with start_servo:=true.")
            return 1

        motion_started = True
        if not node.move_to_pre_squash():
            node.get_logger().error("Initial approach failed. Aborting.")
            return 1
        if not node._operator_confirm(
            f"At pre-squash pose. Confirm clear contact conditions before starting {parsed.trials} trials."
        ):
            return 0
        node.resume_servo()

        nominal_force_ref = node._force_ctrl.reference
        force_ref = nominal_force_ref
        for trial in range(1, parsed.trials + 1):
            node.get_logger().info(f"=== Batch trial {trial}/{parsed.trials} (starting force {force_ref:.2f} N) ===")
            # Each trial's logs cover only that trial.
            node._ft_transformed_log, node._pose_log, node._obj_pose_log, node._command_log = [], [], [], []
            stage = episode.begin_stage("press_pull_tip", trial=trial, trials_total=parsed.trials)
            if not start_recording(node, str(stage.path), quality=parsed.quality):
                node.get_logger().error(f"Trial {trial}: cameras failed to start recording — aborting batch")
                return 1

            start_force_ref = force_ref
            _, force_ref, attempts = run_adaptive_press(node, force_ref, log_prefix=f"Trial {trial} attempt")
            stop_recording(node)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_ft_pose_log(node._ft_transformed_log, node._pose_log, str(stage.path), LOG_PREFIX,
                             node._obj_pose_log, command_log=node._command_log, timestamp=ts)
            save_run_metadata(str(stage.path), LOG_PREFIX, ts, {
                **module_constants(globals()),
                "object": node._object, "trial": trial, "trials_total": parsed.trials,
                "pre_squash": node._pre_squash_pos, "arc_center": node._arc_center,
                "object_nominal_force_ref_n": nominal_force_ref, "trial_start_force_ref_n": start_force_ref,
                "trial_converged_force_ref_n": force_ref, "escalation_attempts": attempts,
                "completed": node._completed,
            })
            stage.end("ok" if node._completed else "failed", completed=node._completed, attempts=attempts,
                      start_force_ref_n=start_force_ref, converged_force_ref_n=force_ref)
            stage = None

            if trial < parsed.trials:
                node.pause_servo()
                if not node.move_to_pre_squash():
                    node.get_logger().error(f"Trial {trial}: return to pre-squash failed — aborting batch")
                    return 1
                node.resume_servo()

        node.get_logger().info(f"Batch complete: {parsed.trials} trials.")

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(node)  # in case an abort left a trial's recording running
        if stage is not None:  # a trial aborted mid-way
            stage.end("aborted", completed=node._completed)
        node._servo_cmd.publish_zero(node._state, node._force_z)
        if motion_started and rclpy.ok():
            node.pause_servo()
            home_ok = node.move_to_home()
            if not home_ok:
                node.get_logger().error("MoveIt return to home failed; not retrying automatically.")
        node._servo_cmd.close()
        deactivate_egm(node)
        if home_ok and rclpy.ok():
            take_snapshot(node, episode, label="after batch")
        print(episode.summary())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
