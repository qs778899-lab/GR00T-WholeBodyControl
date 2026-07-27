/**
 * @file motion_compliance_action_residual.hpp
 * @brief Thin SONIC/G1 adapter for the portable action-residual overlay.
 */

#pragma once

#include "action_residual_overlay.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

class SonicMotionComplianceActionResidual final {
 public:
  static std::unique_ptr<SonicMotionComplianceActionResidual> Load(
      const std::filesystem::path& overlay_config,
      const std::filesystem::path& release_decoder,
      const std::filesystem::path& release_encoder,
      const std::filesystem::path& release_observation_config);

  /**
   * Compose one 50 Hz action before the existing IsaacLab-to-MuJoCo remap.
   *
   * The first two VR controls are left/right wrist mode controls.  Because the
   * trained actor exposes one global three-value condition, either positive
   * wrist value enables the same global residual; this is not a per-wrist
   * condition interface.  All three controls are still range-validated.
   */
  bool Compose(
      std::span<const float> release_context,
      std::span<const float> release_action,
      const std::array<double, 3>& vr_compliance,
      std::span<float> composed_action,
      std::string* error = nullptr) noexcept;

  [[nodiscard]] bool enabled() const noexcept;
  [[nodiscard]] std::uint64_t inference_calls() const noexcept;

 private:
  explicit SonicMotionComplianceActionResidual(
      std::unique_ptr<universal_tracker::compliance::ActionResidualOverlay> overlay);

  std::unique_ptr<universal_tracker::compliance::ActionResidualOverlay> overlay_;
  std::array<float, 3> condition_{};
  std::array<std::uint8_t, 1> gate_{};
};
