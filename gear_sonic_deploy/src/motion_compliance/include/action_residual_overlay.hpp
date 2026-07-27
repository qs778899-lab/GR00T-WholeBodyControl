/**
 * @file action_residual_overlay.hpp
 * @brief Portable, opt-in ONNX action-residual composition.
 *
 * This module deliberately has no robot, tracker, TensorRT, or CUDA dependency.
 * A tracker adapter supplies its expected context/action layouts and the paths
 * of the release artifacts that the residual was trained against.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace universal_tracker::compliance {

struct ContextField {
  std::string name;
  std::size_t offset = 0;
  std::size_t width = 0;
};

struct ReleaseArtifactPin {
  std::string name;
  std::filesystem::path path;
  std::string sha256;
};

struct ActionResidualHostContract {
  // These pins belong to the host integration, not to artifact-authored
  // metadata.  A same-shaped but incompatible release policy must fail closed.
  // The portable runtime does not assume a decoder/encoder split: a host may
  // pin any non-empty set of named model, preprocessing, or schema files.
  std::vector<ReleaseArtifactPin> release_artifacts;
  std::vector<ContextField> context_layout;
  std::vector<std::string> site_layout;
  std::vector<std::string> action_layout;
  std::size_t condition_width = 0;
};

/**
 * Validated standalone action-residual runtime.
 *
 * A missing CLI overlay should mean this object is never constructed.  A
 * supplied config whose `enabled` field is false is parsed but does not read
 * release artifacts, metadata, the ONNX model, or create an ORT session.
 */
class ActionResidualOverlay final {
 public:
  static std::unique_ptr<ActionResidualOverlay> LoadFromConfig(
      const std::filesystem::path& config_path,
      const ActionResidualHostContract& host_contract);

  ~ActionResidualOverlay();
  ActionResidualOverlay(ActionResidualOverlay&&) noexcept;
  ActionResidualOverlay& operator=(ActionResidualOverlay&&) noexcept;
  ActionResidualOverlay(const ActionResidualOverlay&) = delete;
  ActionResidualOverlay& operator=(const ActionResidualOverlay&) = delete;

  [[nodiscard]] bool enabled() const noexcept;
  [[nodiscard]] std::size_t release_context_width() const noexcept;
  [[nodiscard]] std::size_t condition_width() const noexcept;
  [[nodiscard]] std::size_t action_width() const noexcept;
  [[nodiscard]] float max_abs_delta() const noexcept;
  [[nodiscard]] std::span<const float> default_enabled_condition() const noexcept;
  [[nodiscard]] std::uint64_t inference_calls() const noexcept;

  /**
   * Compose a `[batch, sequence, action]` residual.
   *
   * Rejected rows are copied bit-for-bit from `release_action`.  An all-off
   * gate performs no ORT call.  On every validation or inference failure the
   * method returns false and leaves the whole output as a release-action copy.
   */
  bool Compose(
      std::size_t batch,
      std::size_t sequence,
      std::span<const float> release_action_context,
      std::span<const float> condition,
      std::span<const float> release_action,
      std::span<const std::uint8_t> enabled_gate,
      std::span<float> composed_action,
      std::string* error = nullptr) noexcept;

  /**
   * Validate a caller-owned scalar control and derive one global gate.
   *
   * Every supplied value must be finite and lie in `[minimum, maximum]`.
   * Only the first `gate_value_count` values participate in the OR gate; the
   * remaining values are still validated.  This keeps site semantics in the
   * adapter instead of this portable runtime.
   */
  static bool EvaluateAnyPositiveGate(
      std::span<const double> values,
      std::size_t gate_value_count,
      double minimum,
      double maximum,
      bool* active,
      std::string* error = nullptr) noexcept;

 private:
  struct Impl;
  explicit ActionResidualOverlay(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace universal_tracker::compliance
