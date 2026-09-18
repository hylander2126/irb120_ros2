#include <abb_hardware_interface/egm_command_guard.hpp>
#include <gtest/gtest.h>
#include <limits>

using abb_hardware_interface::EGMCommandGuard;

static abb::robot::MotionData feedback(unsigned int sequence = 1)
{
  abb::robot::MotionData data{};
  data.groups.resize(1);
  auto& group = data.groups[0];
  group.egm_channel_data.is_active = true;
  auto& input = group.egm_channel_data.input;
  input.mutable_header()->set_sequence_number(sequence);
  input.mutable_status()->set_egm_state(abb::egm::wrapper::Status::EGM_RUNNING);
  group.units.resize(1);
  auto& unit = group.units[0];
  unit.active = true;
  unit.type = abb::robot::MechanicalUnit_Type_TCP_ROBOT;
  unit.joints.resize(6);
  for (auto& joint : unit.joints)
  {
    joint.state.position = 0.5;
    joint.state.velocity = 0.0;
    joint.command.position = -1.0;  // stale previous target
    joint.command.velocity = 1.0;
    joint.lower_limit = -3.;
    joint.upper_limit = 3.;
    input.mutable_feedback()->mutable_robot()->mutable_joints()->mutable_position()->add_values(28.6479);
    input.mutable_feedback()->mutable_robot()->mutable_joints()->mutable_velocity()->add_values(0.0);
  }
  return data;
}

TEST(EGMGuard, RejectedFirstPacketCannotEnableWrites)
{
  EGMCommandGuard guard;
  auto data = feedback(0);
  data.groups[0].egm_channel_data.is_active = false;
  for (auto& joint : data.groups[0].units[0].joints) joint.state.position = 0.;
  EXPECT_FALSE(guard.initialize(data, false, 6));
  EXPECT_FALSE(guard.initialized());
  data = feedback(1);
  ASSERT_TRUE(guard.initialize(data, true, 6));
  for (const auto& joint : data.groups[0].units[0].joints)
  {
    EXPECT_DOUBLE_EQ(joint.command.position, 0.5);
    EXPECT_DOUBLE_EQ(joint.command.velocity, 0.0);
  }
}

TEST(EGMGuard, DelayedOrIncompleteOrNonfiniteFeedbackCannotInitialize)
{
  EGMCommandGuard guard;
  auto data = feedback();
  for (int i = 0; i < 100; ++i) EXPECT_FALSE(guard.initialize(data, false, 6));
  data.groups[0].units[0].joints.pop_back();
  EXPECT_FALSE(guard.initialize(data, true, 6));
  data = feedback();
  data.groups[0].units[0].joints[0].state.position = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(guard.initialize(data, true, 6));
  data = feedback();
  data.groups[0].egm_channel_data.input.mutable_feedback()->mutable_robot()->mutable_joints()->clear_position();
  EXPECT_FALSE(guard.initialize(data, true, 6));
  EXPECT_FALSE(guard.initialized());
}

TEST(EGMGuard, ValidZeroPoseIsAllowedButCannotBypassUpdatedRead)
{
  EGMCommandGuard guard;
  auto data = feedback(0);
  for (auto& joint : data.groups[0].units[0].joints) joint.state.position = 0.;
  EXPECT_FALSE(guard.initialize(data, false, 6));
  EXPECT_TRUE(guard.initialize(data, true, 6));
}

TEST(EGMGuard, LostChannelLatchesOffUntilExplicitActivation)
{
  EGMCommandGuard guard;
  auto data = feedback(100);
  ASSERT_TRUE(guard.initialize(data, true, 6));
  EXPECT_TRUE(guard.check(data, 6));  // repeated packet below vendor loss threshold
  data.groups[0].egm_channel_data.is_active = false;
  EXPECT_FALSE(guard.check(data, 6));
  data = feedback(101);
  EXPECT_FALSE(guard.check(data, 6));
  EXPECT_FALSE(guard.initialized());
  EXPECT_TRUE(guard.initialize(data, true, 6));
}

TEST(EGMGuard, SessionResetAndStopDisableCommands)
{
  EGMCommandGuard guard;
  auto data = feedback(100);
  ASSERT_TRUE(guard.initialize(data, true, 6));
  data = feedback(0);
  EXPECT_FALSE(guard.check(data, 6));
  data = feedback(100);
  ASSERT_TRUE(guard.initialize(data, true, 6));
  data.groups[0].egm_channel_data.input.mutable_status()->set_egm_state(abb::egm::wrapper::Status::EGM_STOPPED);
  EXPECT_FALSE(guard.check(data, 6));
  guard.reset();
  EXPECT_FALSE(guard.initialized());
}

TEST(EGMGuard, RearmsOnlyAfterTripAndRunningSession)
{
  EGMCommandGuard guard;
  auto data = feedback(100);
  EXPECT_FALSE(guard.rearm(data, true, 6));  // never tripped
  ASSERT_TRUE(guard.initialize(data, true, 6));
  data.groups[0].egm_channel_data.input.mutable_status()->set_egm_state(abb::egm::wrapper::Status::EGM_STOPPED);
  EXPECT_FALSE(guard.check(data, 6));
  EXPECT_FALSE(guard.rearm(data, true, 6));  // still stopped
  data = feedback(1);  // new RUNNING session
  ASSERT_TRUE(guard.rearm(data, true, 6));
  EXPECT_TRUE(guard.check(data, 6));
  data.groups[0].units[0].joints[0].command.position = 0.9;  // stale target
  EXPECT_FALSE(guard.commands_ok(data));
  data.groups[0].units[0].joints[0].command.position = 0.5;
  EXPECT_TRUE(guard.commands_ok(data));
}
