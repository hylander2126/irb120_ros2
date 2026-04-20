#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Empty
from std_msgs.msg import Bool
import numpy as np

# ----------------------- force monitor -----------------------
class ForceWatcher:
    """
    Pseudo-state-machine to monitor force to stop robot motion at threshold during tipping.
        - 'baseline': estimate noise floor from rolling window
        - 'contact': once a sustained rise above baseline is detected
        - 'falling': track peak and trigger when force falls below
                        (1 - k_safe) * peak for several samples
    """
    def __init__(
            self,
            n_safety=0.9, # n_safety = 0 is NO safety and =1 is FULL safety (stop push upon contact)
            debug=False,
            initial_state=None # (i.e. for starting baseline after motion)
        ):
        self.n_safety = n_safety
        self.debug = debug
        self.F_STOP_SAFETY = 15.0 # N absolute max stop (regardless of n_safety and peak) FINGER IS CAUGHT ON SOMETHING....

        self.is_active = False

        # Baseline / noise estimation
        self.baseline_buf = [0.0] * 50 # prepare empty buffer (50 samples is good)
        self.baseline_ready = False
        self.noise_floor = 0.0

        self.median_buf = [0.0] * 5 # Median smoothing (5-sample window is good, but still responsive)

        # Contact detection parameters
        self.contact_delta = 0.09  # N above baseline to declare contact (this is our resolution!)
        self.contact_required = 4  # consecutive samples needed for contact
        self.contact_count = 0
        
        self.release_delta = 0.03  # N below baseline to declare release (smaller than contact so we don't unlatch while pushing)
        self.release_required = 8  # consecutive samples needed for release (no more contact)
        self.release_count = 0

        # Falling / trigger parameters
        self.min_contact_time = 0.25 # Seconds of contact before we can trigger (backup to fall samples)
        self.contact_time = None
        self.fall_samples = 5     # consecutive samples needed for falling trigger
        self.fall_count = 0

        # State
        self.STATE = initial_state
        self.trigger_latched = False
        self.contact_latched = False # True upon contact; latched until reset
        self.peak = 0.0

        # We use the following to keep contact latched after trigger for a set period. This avoids False Positive 'un-contact' during retract.
        self.contact_instant = None # Time of initial contact in seconds
        self.trigger_instant = None # Time of trigger in seconds
        self.retracting_start_time = None # Time when we entered RETRACTING state (for release guard window)
        self.release_guard_time = 0.5 # Seconds to wait before checking for release after entering RETRACTING (avoids false positives during stop→retract transition)

        self.sub_ft = rospy.Subscriber("/netft_data_transformed", WrenchStamped, self.ft_cb, queue_size=50)

        # The following publishers are so that logging can record when contact and trigger occurs
        self.pub_contact = rospy.Publisher('/com_3d/fw_contact_status', Bool, queue_size=1, latch=True)
        self.pub_retract = rospy.Publisher("/com_3d/retract_phase",     Bool, queue_size=1, latch=False)
        self.pub_trigger = rospy.Publisher('/com_3d/fw_trigger_status', Bool, queue_size=1, latch=True)
        
        rospy.loginfo(f"[ForceWatcher] Armed with n_safety={self.n_safety}.")


    def reset(self, force_state="BASELINE"):
        """Full reset of all state variables and publishers."""
        self.STATE = force_state # Or MONITOR if you prefer
        self.trigger_latched = False
        self.contact_latched = False
        self.contact_count = 0
        self.fall_count = 0
        self.peak = 0.0
        self.release_count = 0
        self.contact_time = None
        self.retracting_start_time = None # Reset guard timer on new run

        # baseline reset
        self.baseline_ready = False
        self.baseline_buf = [0.0] * len(self.baseline_buf)

        # clear history
        self.median_buf = [0.0] * len(self.median_buf)
        
        # Publish the cleared state immediately to update latched topics
        self.pub_contact.publish(False)
        self.pub_trigger.publish(False)
        self.pub_retract.publish(False)
        # rospy.loginfo("[ForceWatcher] State reset for new run.")


    def ft_cb(self, msg):
        # Bail out if shutdown or not active
        if rospy.is_shutdown():
            return
        
        # Publish contact and trigger events continuously.
        try:
            self.pub_contact.publish(self.STATE == "CONTACT" or self.contact_latched)
            self.pub_trigger.publish(self.STATE == "RETRACTING" or self.trigger_latched) # HMMM they are kinda the same...
            self.pub_retract.publish(self.STATE == "RETRACTING" or self.trigger_latched)
        except rospy.ROSException:
            return
        
        # If not active, allow publishing (above) but do not process further
        if not self.is_active:
            # Don't reset the whole thing here, just unlatch after retracting finished
            self.STATE = None
            self.trigger_latched = False
            self.contact_latched = False
            return

        # Read force components
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        f_mag = float(np.linalg.norm([fx, fy, fz]))

        # Bail out COMPLETELY if force (components or norm) exceeds absolute max
        if f_mag > self.F_STOP_SAFETY:
            rospy.logerr(f"Absolute force limit exceeded! |F|: {f_mag:.3f} N")
            self.STATE = "RETRACTING" # NEW, this was commented before
            self.trigger_latched = True
            return

        # Median smoothing (rolling window)
        self.median_buf.pop(0)
        self.median_buf.append(f_mag)
        f_med = np.median(self.median_buf)

        # ------------ 1) BASELINE / SEARCHING STATE ------------
        if self.STATE == "BASELINE":
            # Build baseline buffer
            self.baseline_buf.pop(0)
            self.baseline_buf.append(f_med)

            if not self.baseline_ready:
                # Wait until enough samples collected
                if self.baseline_buf.count(0.0) == 0:
                    self.baseline_ready = True
                    # Estimate noise floor as median of baseline buffer/window
                    self.noise_floor = float(np.median(self.baseline_buf))
            else:
                rospy.loginfo_once(f"Noise floor: {self.noise_floor:.3f} N  Contact delta: {self.contact_delta:.3f} N\n Combined: {self.noise_floor + self.contact_delta:.3f} N")
                self.STATE = "MONITOR"
        
        # ------------ 2) MONITORING FOR CONTACT ------------
        if self.STATE == "MONITOR":
            # Check for sustained rise above baseline
            if f_med > (self.noise_floor + self.contact_delta):
                self.contact_count += 1
                self.debug_msg(f"Contact magnitude met: {f_med:.3f} N, Count: {self.contact_count}/{self.contact_required}")
                
                # Once enough samples detected, latch contact and move to PEAK state
                if self.contact_count >= self.contact_required:
                    self.STATE = "CONTACT"
                    self.contact_latched = True
                    self.release_count = 0
                    self.contact_time = rospy.Time.now().to_sec() # NEW
            else:
                self.contact_count = 0

        # ------------ 3) CONTACT / PEAK TRACING ------------
        if self.STATE == "CONTACT":
            
            # UPDATE: Continue Tracing Peak if F rises
            old_peak = self.peak
            self.peak = max(self.peak, f_med)
            if self.peak > old_peak:
                rospy.loginfo(f"Tracing new peak: {self.peak:.3f} N")

            f_safe = self.n_safety * self.peak
            thresh = max(f_safe, self.noise_floor)

            self.debug_msg(f"Current: {f_med:.3f}N  Peak: {self.peak:.3f} N  f_safe: {f_safe:.3f} N", 0.1)
            
            # Only check f_safe if peak is non-zero
            if self.peak > 0.0:
                if self.contact_time is not None:
                    if (rospy.Time.now().to_sec() - self.contact_time) < self.min_contact_time:
                        self.fall_count = 0  # reset below count if min contact time not met
                        return  # skip further processing until min contact time met
                if f_med < thresh:
                    self.fall_count += 1
                    if self.fall_count >= self.fall_samples:
                        self.STATE = "RETRACTING"
                        self.retracting_start_time = rospy.Time.now().to_sec()  # Start guard timer
                        self.trigger_latched = True
                        self.debug_msg(f"STOP! peak: {self.peak:.3f} N, f_safe: {thresh:.3f}, last: {f_med:.3f}")
                else:
                    self.fall_count = 0

        # ---------------- Release detection (un-contact) ----------------
        if self.STATE == "RETRACTING":
            # Check if we're still within the guard window (skip release detection during stop→retract transition)
            if self.retracting_start_time is not None:
                time_in_retracting = rospy.Time.now().to_sec() - self.retracting_start_time
                if time_in_retracting < self.release_guard_time:
                    # Still in guard window; don't check for release yet
                    return

            # Only attempt release if we already latched contact and baseline exists
            release_thresh = self.noise_floor + self.release_delta

            if f_med < release_thresh:
                self.release_count += 1
                if self.release_count >= self.release_required:
                    self.contact_latched = False
                    self.release_count = 0

                    rospy.loginfo_once(
                        f"CONTACT RELEASED (f_med={f_med:.3f} < {release_thresh:.3f} for {self.release_required} samples).")
            else:
                self.release_count = 0

        # NOTE: We STAY in RETRACTING until reset (in push.py) This is so that we can capture the retract phase for fitting.


    def debug_msg(self, msg, throttle=None):
        if self.debug:
            if throttle is None:
                rospy.loginfo(f"[ForceWatcher] {msg}")
            elif throttle == 0:
                rospy.loginfo_throttle(5.0, f"[ForceWatcher] {msg}") # large 5s throttle
            else:
                rospy.loginfo_throttle(throttle, f"[ForceWatcher] {msg}")