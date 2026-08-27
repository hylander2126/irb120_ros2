#!/usr/bin/env python3
"""Automated batch of N identical, non-adaptive arc_static trials — for objects
that (per manual testing) settle back into a good pose on their own after each
squash-arc-unarc-retract cycle, so the per-trial overhead in a normal
arc_static run is pure waste.

Differences from a normal `arc_static` run, all aimed at that overhead:
  - check_press_point() runs ONCE at the very start (sanity check against the
    hardcoded pre_squash pose), not before every trial.
  - move_to_pre_squash() likewise runs ONCE up front. Between trials, the only
    return-to-pre-squash is the same MoveIt call again (still a real,
    verified return — not just trusting the FSM's open-loop RETRACT phase to
    have landed exactly on-pose), but there's no repeated press-point check
    and no repeated operator confirmation.
  - Each trial still uses arc_static's normal retry-on-slip force-escalation
    ladder (run_adaptive_press, shared with plain arc_static) — but the
    starting force for each trial CARRIES FORWARD from wherever the previous
    trial ended up, rather than resetting to the object's nominal
    object_params.json value every time. For an object like the monitor,
    where the nominal force is deliberately too low to showcase the adaptive
    mechanism, this means only trial 1 pays for the escalation search; trials
    2..N start from the already-discovered working force and typically
    complete on their first attempt.
  - No home return between trials — only once at the very end (or on abort).
  - The operator confirmation gate fires once, before trial 1, not per trial.

Video and F/T+pose+metadata logs stay one-per-trial, same filename
conventions as a normal arc_static run (just tagged "arc_static_batch" and
with a trial number in the metadata) — so 10 trials still produce 10 separate
recordings/logs, not one long one.

Usage:
    ros2 run irb120_control arc_static_batch box --trials 10
    ros2 run irb120_control arc_static_batch box --trials 10 --quality lossless
"""

import argparse
import sys
from datetime import datetime

import rclpy

from irb120_control.arc_static import ArcStatic, FORCE_HARD_LIMIT_N, run_adaptive_press
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.press_point_check import check_press_point
from irb120_control.util.runtime_log_dir import (
    module_constants,
    save_ft_pose_log,
    save_run_metadata,
    start_recording,
    stop_recording,
    VALID_OBJECTS,
)

LOG_PREFIX = "arc_static_batch"
DEFAULT_TRIALS = 10


def _reset_trial_logs(node: ArcStatic) -> None:
    """Clear the per-trial F/T + pose buffers so each trial's saved log covers
    only that trial, not a running concatenation of the whole batch."""
    node._ft_transformed_log = []
    node._pose_log = []
    node._obj_pose_log = []
    node._command_log = []


def _save_trial_log(
    node: ArcStatic,
    trial: int,
    trials_total: int,
    object_nominal_force_ref_n: float,
    trial_start_force_ref_n: float,
    trial_converged_force_ref_n: float,
    escalation_attempts: int,
) -> None:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_ft_pose_log(
            node._ft_transformed_log,
            node._pose_log,
            node._log_subdir,
            LOG_PREFIX,
            node._obj_pose_log,
            command_log=node._command_log,
            timestamp=ts,
        )
        save_run_metadata(
            node._log_subdir,
            LOG_PREFIX,
            ts,
            {
                **module_constants(globals()),
                "object": node._object,
                "trial": trial,
                "trials_total": trials_total,
                "object_nominal_force_ref_n": object_nominal_force_ref_n,
                "trial_start_force_ref_n": trial_start_force_ref_n,
                "trial_converged_force_ref_n": trial_converged_force_ref_n,
                "escalation_attempts": escalation_attempts,
                "completed": node._completed,
                "press_point_check": getattr(node, "_press_point_check_result", None),
            },
        )
    except Exception as exc:
        node.get_logger().error(f"Trial {trial}: failed to save F/T+pose log: {exc}")


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    app_args = raw_args[:ros_args_index]
    ros_args = raw_args[ros_args_index:]

    parser = argparse.ArgumentParser(description="Run N identical non-adaptive arc_static trials back to back")
    parser.add_argument("object", nargs="?", choices=sorted(VALID_OBJECTS))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help=f"Number of trials (default: {DEFAULT_TRIALS})")
    parser.add_argument(
        "--quality", choices=["h264", "lossless"], default="h264",
        help="Video quality for every trial in this batch (default: h264). Same knob as arc_static.",
    )
    parser.add_argument(
        "--ignore-press-sanity", action="store_true",
        help="Continue even if the one-time press-point sanity check fails. Same knob as "
             "arc_static — only skip it when you already know why it's failing.",
    )
    parsed = parser.parse_args(app_args)

    rclpy.init(args=ros_args)
    node = ArcStatic(object_name=parsed.object)
    try:
        if not tare_netft(node):
            return 1

        if not ensure_egm_active(node):
            return 1

        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error(
                "MoveIt Servo is not ready (/servo_node/delta_twist_cmds has no subscribers). "
                "Launch stack with start_servo:=true before running arc_static_batch."
            )
            return 1

        # One-time sanity check — same safety gate as a normal arc_static run,
        # just not repeated before every trial.
        if not check_press_point(node, node._pre_squash_pos, label=f"arc_static_batch/{node._object}"):
            msg = (
                "Press-point sanity check failed — perception disagrees with the calibrated "
                "pre_squash pose (or no detection at all)."
            )
            if parsed.ignore_press_sanity:
                node.get_logger().warn(f"{msg} Continuing anyway (--ignore-press-sanity).")
            else:
                node.get_logger().error(f"{msg} Aborting before any motion.")
                return 1

        if not node.move_to_pre_squash():
            node.get_logger().error("Initial approach failed. Aborting.")
            return 1

        node.get_logger().info(
            f"[PARAMS] object={node._object}  trials={parsed.trials}  quality={parsed.quality}  "
            f"force_ref={node._force_ctrl.reference:.2f}N  hard_limit={FORCE_HARD_LIMIT_N:.1f}N  "
            f"pre_squash_pos={node._pre_squash_pos}"
        )

        if not node._operator_confirm(
            f"At pre-squash pose. Confirm clear contact conditions before starting {parsed.trials} trials."
        ):
            return 0

        node.resume_servo()

        # Object's as-configured force_ref (object_params.json), kept around only
        # for the log — never fed back into reset_attempt() after trial 1. The
        # value actually driving each trial is current_force_ref below, which
        # carries forward from wherever the previous trial's ladder converged
        # (see run_adaptive_press) — that's the whole point for an object like
        # the monitor, where the nominal value is deliberately too low.
        object_nominal_force_ref = node._force_ctrl.reference
        current_force_ref = object_nominal_force_ref

        for trial in range(1, parsed.trials + 1):
            node.get_logger().info(f"=== Batch trial {trial}/{parsed.trials} (starting force {current_force_ref:.2f} N) ===")
            _reset_trial_logs(node)

            if not start_recording(node, node._log_subdir, quality=parsed.quality):
                node.get_logger().error(f"Trial {trial}: one or more cameras failed to start recording — aborting batch")
                return 1

            trial_start_force_ref = current_force_ref
            _, current_force_ref, escalation_attempts = run_adaptive_press(
                node, current_force_ref, log_prefix=f"Trial {trial} attempt"
            )

            stop_recording(node)
            _save_trial_log(
                node, trial, parsed.trials,
                object_nominal_force_ref, trial_start_force_ref, current_force_ref, escalation_attempts,
            )

            if trial < parsed.trials:
                node.pause_servo()
                node.get_logger().info("Returning to pre-squash for next trial...")
                if not node.move_to_pre_squash():
                    node.get_logger().error(f"Trial {trial}: return to pre-squash failed — aborting batch")
                    return 1
                node.resume_servo()

        node.get_logger().info(f"Batch complete: {parsed.trials} trials.")

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(node)  # safety net in case an abort left a trial's recording running
        node._servo_cmd.publish_zero(node._state, node._force_z)
        if rclpy.ok():
            node.pause_servo()
            node.get_logger().info(
                "Returning to home position (all joints zero) via MoveIt on the existing EGM session..."
            )
            if not node.move_to_home():
                node.get_logger().error(
                    "MoveIt return to home failed. EGM may have ended; "
                    "not retrying automatically to avoid an unsafe snap."
                )
        node._servo_cmd.close()
        deactivate_egm(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
