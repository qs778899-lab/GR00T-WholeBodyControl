#pragma once

#include <map>
#include <string>
#include <vector>

#include "../motion_data_reader.hpp"
#include "../policy_parameters.hpp"
#include "sim2sim_debug/source_frame_tracker.hpp"

namespace sim2sim_debug {

inline void PopulateRawReferenceOutputFields(
    std::map<std::string, std::vector<double>>& output_data_map,
    const MotionSequence* current_motion,
    int current_frame) {
  if (!SourceFrameTrackingEnabled() || current_motion == nullptr ||
      current_frame < 0 || current_frame >= current_motion->timesteps) {
    return;
  }
  if (current_motion->GetNumJoints() < 29 ||
      current_motion->GetNumBodies() < 1 ||
      current_motion->GetNumBodyQuaternions() < 1) {
    return;
  }

  std::vector<double> body_q_raw(29, 0.0);
  for (int i = 0; i < 29; ++i) {
    body_q_raw[i] = current_motion->JointPositions(current_frame)[isaaclab_to_mujoco[i]];
  }

  const auto& base_trans_raw = current_motion->BodyPositions(current_frame)[0];
  const auto& base_quat_raw = current_motion->BodyQuaternions(current_frame)[0];

  output_data_map["ref_base_trans_raw"].assign(base_trans_raw.begin(), base_trans_raw.end());
  output_data_map["ref_base_quat_raw"].assign(base_quat_raw.begin(), base_quat_raw.end());
  output_data_map["ref_body_q_raw"] = std::move(body_q_raw);
}

}  // namespace sim2sim_debug
