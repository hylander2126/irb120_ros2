#pragma once

#include <abb_egm_rws_managers/data_containers.h>
#include <cmath>
#include <cstdint>

namespace abb_hardware_interface
{
// This driver controls one IRB120 EGM channel. Keep the gate independent of ROS
// so packet loss/restarts and initial zero defaults can be tested without a robot.
class EGMCommandGuard
{
public:
  void reset() { initialized_ = tripped_ = check_commands_ = false; }
  bool initialized() const { return initialized_; }

  bool initialize(abb::robot::MotionData& data, bool updated, size_t joint_count)
  {
    reset();
    if (!updated || !valid(data, joint_count)) return false;
    for (auto& joint : data.groups[0].units[0].joints)
    {
      joint.command.position = joint.state.position;
      joint.command.velocity = 0.0;
    }
    sequence_ = data.groups[0].egm_channel_data.input.header().sequence_number();
    running_ = data.groups[0].egm_channel_data.input.status().egm_state() ==
      abb::egm::wrapper::Status::EGM_RUNNING;
    initialized_ = true;
    return true;
  }

  bool check(const abb::robot::MotionData& data, size_t joint_count)
  {
    if (!initialized_) return false;
    if (!valid(data, joint_count))
    {
      trip();
      return false;
    }
    const auto sequence = data.groups[0].egm_channel_data.input.header().sequence_number();
    const bool running = data.groups[0].egm_channel_data.input.status().egm_state() ==
      abb::egm::wrapper::Status::EGM_RUNNING;
    // Fail closed on a new EGM session, including a counter wrap. Never let an
    // active controller replay a previous session's trajectory after reconnect.
    if (sequence < sequence_ || (running_ && !running))
    {
      trip();
      return false;
    }
    sequence_ = sequence;
    running_ = running;
    return true;
  }

  // After a trip, resume once the operator has restarted EGM (new session,
  // RUNNING, valid feedback). Re-seeds commands from measured position.
  bool rearm(abb::robot::MotionData& data, bool updated, size_t joint_count)
  {
    if (!tripped_ || !updated || !valid(data, joint_count) ||
        data.groups[0].egm_channel_data.input.status().egm_state() !=
          abb::egm::wrapper::Status::EGM_RUNNING)
      return false;
    if (!initialize(data, updated, joint_count)) return false;
    check_commands_ = true;
    return true;
  }

  // After a rearm the controller may still hold a stale pre-outage target.
  // Returns false (do not send) until its command is near the measured position.
  bool commands_ok(const abb::robot::MotionData& data, double tolerance = 0.1)
  {
    if (!check_commands_) return true;
    for (const auto& joint : data.groups[0].units[0].joints)
      if (!(std::abs(joint.command.position - joint.state.position) <= tolerance)) return false;
    check_commands_ = false;
    return true;
  }

private:
  void trip()
  {
    initialized_ = check_commands_ = false;
    tripped_ = true;
  }

  static bool valid(const abb::robot::MotionData& data, size_t joint_count)
  {
    if (joint_count != 6 || data.groups.size() != 1) return false;
    const auto& group = data.groups[0];
    const auto& channel = group.egm_channel_data;
    if (!channel.is_active || !channel.input.header().has_sequence_number() ||
        group.units.size() != 1) return false;
    const auto& unit = group.units[0];
    if (!unit.active || unit.type != abb::robot::MechanicalUnit_Type_TCP_ROBOT ||
        unit.joints.size() != joint_count) return false;
    const auto& feedback = channel.input.feedback().robot().joints();
    if (feedback.position().values_size() != static_cast<int>(joint_count) ||
        feedback.velocity().values_size() != static_cast<int>(joint_count)) return false;
    for (size_t i = 0; i < joint_count; ++i)
    {
      const auto& joint = unit.joints[i];
      if (!std::isfinite(joint.state.position) || !std::isfinite(joint.state.velocity) ||
          !std::isfinite(feedback.position().values(i)) ||
          !std::isfinite(feedback.velocity().values(i)) ||
          joint.state.position < joint.lower_limit || joint.state.position > joint.upper_limit)
        return false;
    }
    return true;
  }

  bool initialized_ = false;
  bool tripped_ = false;
  bool check_commands_ = false;
  uint32_t sequence_ = 0;
  bool running_ = false;
};
}  // namespace abb_hardware_interface
