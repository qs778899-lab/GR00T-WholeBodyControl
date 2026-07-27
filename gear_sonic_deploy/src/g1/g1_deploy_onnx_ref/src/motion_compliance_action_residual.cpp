/**
 * @file motion_compliance_action_residual.cpp
 * @brief SONIC release identity, layout, and operator-control binding.
 */

#include "../include/motion_compliance_action_residual.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

using universal_tracker::compliance::ActionResidualHostContract;
using universal_tracker::compliance::ActionResidualOverlay;
using universal_tracker::compliance::ContextField;
using universal_tracker::compliance::ReleaseArtifactPin;

constexpr char kDecoderSha256[] =
    "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed";
constexpr char kEncoderSha256[] =
    "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3";
constexpr char kObservationConfigSha256[] =
    "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b";

std::vector<std::string> G1IsaacLabActionLayout() {
  return {
      "left_hip_pitch_joint",       "right_hip_pitch_joint",
      "waist_yaw_joint",            "left_hip_roll_joint",
      "right_hip_roll_joint",       "waist_roll_joint",
      "left_hip_yaw_joint",         "right_hip_yaw_joint",
      "waist_pitch_joint",          "left_knee_joint",
      "right_knee_joint",           "left_shoulder_pitch_joint",
      "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
      "right_ankle_pitch_joint",    "left_shoulder_roll_joint",
      "right_shoulder_roll_joint",  "left_ankle_roll_joint",
      "right_ankle_roll_joint",     "left_shoulder_yaw_joint",
      "right_shoulder_yaw_joint",   "left_elbow_joint",
      "right_elbow_joint",          "left_wrist_roll_joint",
      "right_wrist_roll_joint",     "left_wrist_pitch_joint",
      "right_wrist_pitch_joint",    "left_wrist_yaw_joint",
      "right_wrist_yaw_joint",
  };
}

ActionResidualHostContract MakeHostContract(
    const std::filesystem::path& decoder,
    const std::filesystem::path& encoder,
    const std::filesystem::path& observation_config) {
  return ActionResidualHostContract{
      .release_artifacts = {
          ReleaseArtifactPin{
              .name = "decoder",
              .path = decoder,
              .sha256 = kDecoderSha256,
          },
          ReleaseArtifactPin{
              .name = "encoder",
              .path = encoder,
              .sha256 = kEncoderSha256,
          },
          ReleaseArtifactPin{
              .name = "observation_config",
              .path = observation_config,
              .sha256 = kObservationConfigSha256,
          },
      },
      .context_layout = {
          ContextField{.name = "robot_motion_token", .offset = 0, .width = 64},
          ContextField{.name = "actor_observation", .offset = 64, .width = 930},
      },
      .site_layout = {"left_wrist_yaw_link", "right_wrist_yaw_link"},
      .action_layout = G1IsaacLabActionLayout(),
      .condition_width = 3,
  };
}

}  // namespace

SonicMotionComplianceActionResidual::SonicMotionComplianceActionResidual(
    std::unique_ptr<ActionResidualOverlay> overlay)
    : overlay_(std::move(overlay)) {
  if (overlay_ == nullptr) throw std::invalid_argument("overlay must not be null");
}

std::unique_ptr<SonicMotionComplianceActionResidual>
SonicMotionComplianceActionResidual::Load(
    const std::filesystem::path& overlay_config,
    const std::filesystem::path& release_decoder,
    const std::filesystem::path& release_encoder,
    const std::filesystem::path& release_observation_config) {
  auto overlay = ActionResidualOverlay::LoadFromConfig(
      overlay_config,
      MakeHostContract(
          release_decoder, release_encoder, release_observation_config));
  return std::unique_ptr<SonicMotionComplianceActionResidual>(
      new SonicMotionComplianceActionResidual(std::move(overlay)));
}

bool SonicMotionComplianceActionResidual::Compose(
    std::span<const float> release_context,
    std::span<const float> release_action,
    const std::array<double, 3>& vr_compliance,
    std::span<float> composed_action,
    std::string* error) noexcept {
  if (release_action.size() == composed_action.size() && !release_action.empty()) {
    std::memmove(
        composed_action.data(), release_action.data(),
        release_action.size() * sizeof(float));
  }
  // A supplied-but-disabled overlay is an exact host bypass.  In particular,
  // legacy operators may provide values outside the residual's validated
  // control range; those values must remain irrelevant until the overlay is
  // explicitly enabled in its own configuration.
  if (!overlay_->enabled()) {
    gate_[0] = 0U;
    condition_.fill(0.0F);
    return overlay_->Compose(
        1, 1, release_context, condition_, release_action, gate_,
        composed_action, error);
  }
  bool active = false;
  if (!ActionResidualOverlay::EvaluateAnyPositiveGate(
          vr_compliance, 2, 0.0, 0.5, &active, error)) {
    return false;
  }
  gate_[0] = static_cast<std::uint8_t>(active);
  condition_.fill(0.0F);
  if (active) {
    const auto configured = overlay_->default_enabled_condition();
    if (configured.size() != condition_.size()) {
      if (error != nullptr) {
        *error = "SONIC residual condition does not have three columns";
      }
      return false;
    }
    std::copy(configured.begin(), configured.end(), condition_.begin());
  }
  return overlay_->Compose(
      1, 1, release_context, condition_, release_action, gate_, composed_action,
      error);
}

bool SonicMotionComplianceActionResidual::enabled() const noexcept {
  return overlay_ != nullptr && overlay_->enabled();
}

std::uint64_t SonicMotionComplianceActionResidual::inference_calls() const noexcept {
  return overlay_ == nullptr ? 0 : overlay_->inference_calls();
}
