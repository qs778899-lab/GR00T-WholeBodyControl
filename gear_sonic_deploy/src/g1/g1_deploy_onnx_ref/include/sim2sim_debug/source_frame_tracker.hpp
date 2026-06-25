#pragma once

#include <cstdint>
#include <optional>

struct MotionSequence;

namespace sim2sim_debug {

void SetSourceFrameTrackingEnabled(bool enabled);

bool SourceFrameTrackingEnabled();

void RegisterSourceFrameWindow(const MotionSequence* motion,
                               int64_t window_start,
                               int frame_step,
                               int timesteps);

std::optional<int64_t> LookupSourceFrameIndex(const MotionSequence* motion,
                                              int current_frame);

void ClearSourceFrameWindows();

}  // namespace sim2sim_debug
