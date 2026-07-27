/**
 * Real system-ONNX-Runtime smoke for the portable residual and SONIC adapter.
 *
 * This executable deliberately avoids the Unitree/TensorRT runtime.  It tests
 * the deployable boundary that is new in Phase 5 while the production source
 * hook is checked separately by the Python runner.
 */

#include "action_residual_overlay.hpp"
#include "motion_compliance_action_residual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

template <typename T>
bool EqualBits(std::span<const T> left, std::span<const T> right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size_bytes()) == 0;
}

std::vector<std::string> ActionLayout() {
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

ActionResidualHostContract HostContract(
    const std::filesystem::path& decoder,
    const std::filesystem::path& encoder,
    const std::filesystem::path& observation_config) {
  return ActionResidualHostContract{
      .release_artifacts = {
          ReleaseArtifactPin{
              .name = "decoder", .path = decoder, .sha256 = kDecoderSha256},
          ReleaseArtifactPin{
              .name = "encoder", .path = encoder, .sha256 = kEncoderSha256},
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
      .action_layout = ActionLayout(),
      .condition_width = 3,
  };
}

void TestPortableArbitraryDisabledContracts(
    const std::filesystem::path& disabled_config,
    const std::filesystem::path& deliberately_missing) {
  const auto make_host = [&](std::vector<ContextField> context_layout,
                             std::size_t condition_width) {
    return ActionResidualHostContract{
        .release_artifacts = {
            ReleaseArtifactPin{
                .name = "policy_graph",
                .path = deliberately_missing,
                .sha256 = std::string(64, 'a'),
            },
        },
        .context_layout = std::move(context_layout),
        .site_layout = {"tool_a", "tool_b", "tool_c"},
        .action_layout = {"axis_0", "axis_1", "axis_2", "axis_3", "axis_4"},
        .condition_width = condition_width,
    };
  };
  auto one_field = ActionResidualOverlay::LoadFromConfig(
      disabled_config,
      make_host(
          {ContextField{.name = "state", .offset = 0, .width = 7}}, 4));
  Require(!one_field->enabled(), "one-field portable contract became enabled");
  Require(
      one_field->release_context_width() == 7 &&
          one_field->condition_width() == 4 && one_field->action_width() == 5,
      "one-field portable widths differ");

  auto three_fields = ActionResidualOverlay::LoadFromConfig(
      disabled_config,
      make_host(
          {
              ContextField{.name = "history", .offset = 0, .width = 2},
              ContextField{.name = "state", .offset = 2, .width = 3},
              ContextField{.name = "command", .offset = 5, .width = 4},
          },
          6));
  Require(!three_fields->enabled(), "three-field portable contract became enabled");
  Require(
      three_fields->release_context_width() == 9 &&
          three_fields->condition_width() == 6 &&
          three_fields->action_width() == 5,
      "three-field portable widths differ");
}

void TestDisabledAdapter(
    const std::filesystem::path& disabled_config,
    const std::filesystem::path& deliberately_missing) {
  auto adapter = SonicMotionComplianceActionResidual::Load(
      disabled_config, deliberately_missing, deliberately_missing,
      deliberately_missing);
  Require(!adapter->enabled(), "disabled adapter reported enabled");
  std::vector<float> release(29);
  for (std::size_t index = 0; index < release.size(); ++index) {
    release[index] = static_cast<float>(index) * 0.01F;
  }
  std::vector<float> composed(release.size(), -99.0F);
  const std::array<double, 3> irrelevant_controls{
      std::numeric_limits<double>::quiet_NaN(), 5.0, -5.0};
  std::string error;
  Require(
      adapter->Compose({}, release, irrelevant_controls, composed, &error),
      "disabled adapter did not bypass: " + error);
  Require(EqualBits<float>(release, composed), "disabled adapter changed action bytes");
  Require(adapter->inference_calls() == 0, "disabled adapter created an inference call");
}

void TestPortableDynamicRuntime(
    const std::filesystem::path& enabled_config,
    const ActionResidualHostContract& host) {
  auto overlay = ActionResidualOverlay::LoadFromConfig(enabled_config, host);
  Require(overlay->enabled(), "enabled portable overlay reported disabled");
  constexpr std::size_t kBatch = 2;
  constexpr std::size_t kSequence = 3;
  constexpr std::size_t kRows = kBatch * kSequence;
  constexpr std::size_t kContext = 994;
  constexpr std::size_t kCondition = 3;
  constexpr std::size_t kAction = 29;
  std::vector<float> context(kRows * kContext);
  std::vector<float> condition(kRows * kCondition, 0.0F);
  std::vector<float> release(kRows * kAction);
  std::vector<float> composed(release.size(), -77.0F);
  const std::array<std::uint8_t, kRows> gate{1, 0, 1, 0, 0, 1};
  for (std::size_t index = 0; index < context.size(); ++index) {
    context[index] = std::sin(static_cast<float>(index) * 0.001F);
  }
  for (std::size_t index = 0; index < release.size(); ++index) {
    release[index] = std::cos(static_cast<float>(index) * 0.03F) * 0.2F;
  }
  for (std::size_t row = 0; row < kRows; ++row) {
    if (gate[row] == 1U) {
      condition[row * kCondition] = 1.0F;
      condition[row * kCondition + 1] = 10.0F;
      condition[row * kCondition + 2] = 200.0F;
    } else {
      std::fill_n(
          context.begin() + static_cast<std::ptrdiff_t>(row * kContext),
          kContext, std::numeric_limits<float>::quiet_NaN());
      std::fill_n(
          condition.begin() + static_cast<std::ptrdiff_t>(row * kCondition),
          kCondition, std::numeric_limits<float>::quiet_NaN());
    }
  }
  std::string error;
  Require(
      overlay->Compose(
          kBatch, kSequence, context, condition, release, gate, composed,
          &error),
      "portable mixed compose failed: " + error);
  Require(overlay->inference_calls() == 1, "portable mixed compose call count differs");
  bool changed = false;
  for (std::size_t row = 0; row < kRows; ++row) {
    const auto offset = row * kAction;
    const auto release_row = std::span<const float>(release).subspan(offset, kAction);
    const auto composed_row = std::span<const float>(composed).subspan(offset, kAction);
    if (gate[row] == 0U) {
      Require(EqualBits<float>(release_row, composed_row), "inactive row changed bytes");
      continue;
    }
    for (std::size_t index = 0; index < kAction; ++index) {
      const auto delta = composed_row[index] - release_row[index];
      Require(std::isfinite(composed_row[index]), "active composed action is non-finite");
      Require(std::abs(delta) <= 0.250001F, "active action delta exceeded bound");
      changed = changed || delta != 0.0F;
    }
  }
  Require(changed, "trained residual produced no changed action in dynamic smoke");

  const auto calls_before_off = overlay->inference_calls();
  const std::array<std::uint8_t, kRows> all_off{};
  std::fill(composed.begin(), composed.end(), -55.0F);
  Require(
      overlay->Compose(
          kBatch, kSequence, {}, {}, release, all_off, composed, &error),
      "all-off portable bypass failed: " + error);
  Require(EqualBits<float>(release, composed), "all-off portable bypass changed bytes");
  Require(
      overlay->inference_calls() == calls_before_off,
      "all-off portable bypass called ORT");
}

void TestEnabledSonicAdapter(
    const std::filesystem::path& enabled_config,
    const std::filesystem::path& decoder,
    const std::filesystem::path& encoder,
    const std::filesystem::path& observation_config) {
  auto adapter = SonicMotionComplianceActionResidual::Load(
      enabled_config, decoder, encoder, observation_config);
  Require(adapter->enabled(), "enabled SONIC adapter reported disabled");
  std::vector<float> context(994);
  std::vector<float> release(29);
  std::vector<float> composed(29);
  for (std::size_t index = 0; index < context.size(); ++index) {
    context[index] = std::sin(static_cast<float>(index) * 0.007F);
  }
  for (std::size_t index = 0; index < release.size(); ++index) {
    release[index] = static_cast<float>(index) * 0.005F;
  }
  std::string error;
  Require(
      adapter->Compose(
          context, release, std::array<double, 3>{0.1, 0.0, 0.0}, composed,
          &error),
      "enabled SONIC adapter failed: " + error);
  Require(adapter->inference_calls() == 1, "enabled SONIC call count differs");
  for (std::size_t index = 0; index < release.size(); ++index) {
    Require(
        std::isfinite(composed[index]) &&
            std::abs(composed[index] - release[index]) <= 0.250001F,
        "enabled SONIC adapter violated finite/bounded composition");
  }

  const auto calls_before_off = adapter->inference_calls();
  Require(
      adapter->Compose(
          {}, release, std::array<double, 3>{0.0, 0.0, 0.0}, composed,
          &error),
      "SONIC all-off bypass failed: " + error);
  Require(EqualBits<float>(release, composed), "SONIC all-off changed action bytes");
  Require(adapter->inference_calls() == calls_before_off, "SONIC all-off called ORT");

  Require(
      !adapter->Compose(
          context, release, std::array<double, 3>{0.6, 0.0, 0.0}, composed,
          &error),
      "out-of-range SONIC control was accepted");
  Require(EqualBits<float>(release, composed), "failed SONIC compose did not fall back");
  Require(adapter->inference_calls() == calls_before_off, "failed gate called ORT");

  for (const double invalid : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity()}) {
    Require(
        !adapter->Compose(
            context, release, std::array<double, 3>{invalid, 0.0, 0.0},
            composed, &error),
        "non-finite SONIC control was accepted");
    Require(
        EqualBits<float>(release, composed),
        "non-finite SONIC control did not fall back");
    Require(
        adapter->inference_calls() == calls_before_off,
        "non-finite gate called ORT");
  }
}

void TestBaseIdentityRejection(
    const std::filesystem::path& enabled_config,
    const std::filesystem::path& wrong_decoder,
    const std::filesystem::path& encoder,
    const std::filesystem::path& observation_config) {
  try {
    (void)SonicMotionComplianceActionResidual::Load(
        enabled_config, wrong_decoder, encoder, observation_config);
  } catch (const std::exception&) {
    return;
  }
  Fail("enabled adapter accepted an incompatible release decoder");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8) {
      Fail(
          "usage: smoke DISABLED_YAML ENABLED_YAML DECODER ENCODER OBS_CONFIG "
          "WRONG_DECODER MISSING_PATH");
    }
    const std::filesystem::path disabled_config(argv[1]);
    const std::filesystem::path enabled_config(argv[2]);
    const std::filesystem::path decoder(argv[3]);
    const std::filesystem::path encoder(argv[4]);
    const std::filesystem::path observation_config(argv[5]);
    const std::filesystem::path wrong_decoder(argv[6]);
    const std::filesystem::path deliberately_missing(argv[7]);

    TestPortableArbitraryDisabledContracts(
        disabled_config, deliberately_missing);
    TestDisabledAdapter(disabled_config, deliberately_missing);
    const auto host = HostContract(decoder, encoder, observation_config);
    TestPortableDynamicRuntime(enabled_config, host);
    TestEnabledSonicAdapter(
        enabled_config, decoder, encoder, observation_config);
    TestBaseIdentityRejection(
        enabled_config, wrong_decoder, encoder, observation_config);
    std::cout
        << "MOTION_COMPLIANCE_PHASE5_CPP_ORT_PASS dynamic_shape=2x3 "
           "mixed_rows=6 hard_off_calls=0 action_width=29 context_width=994 "
           "portable_context_fields=1,3 nonfinite_controls=2"
        << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MOTION_COMPLIANCE_PHASE5_CPP_ORT_FAIL: " << error.what()
              << std::endl;
    return 1;
  }
}
