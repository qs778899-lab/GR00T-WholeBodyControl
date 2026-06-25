#include "sim2sim_debug/source_frame_tracker.hpp"

#include <mutex>
#include <unordered_map>

namespace sim2sim_debug {
namespace {

struct SourceFrameWindow {
  int64_t window_start = 0;
  int frame_step = 1;
  int timesteps = 0;
};

std::mutex g_mutex;
bool g_enabled = false;
std::unordered_map<const MotionSequence*, SourceFrameWindow> g_windows;

}  // namespace

void SetSourceFrameTrackingEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_enabled = enabled;
  if (!g_enabled) {
    g_windows.clear();
  }
}

bool SourceFrameTrackingEnabled() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_enabled;
}

void RegisterSourceFrameWindow(const MotionSequence* motion,
                               int64_t window_start,
                               int frame_step,
                               int timesteps) {
  if (motion == nullptr || timesteps <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled) {
    return;
  }
  g_windows[motion] = SourceFrameWindow{
      window_start,
      frame_step > 0 ? frame_step : 1,
      timesteps,
  };
}

std::optional<int64_t> LookupSourceFrameIndex(const MotionSequence* motion,
                                              int current_frame) {
  if (motion == nullptr || current_frame < 0) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled) {
    return std::nullopt;
  }
  auto it = g_windows.find(motion);
  if (it == g_windows.end()) {
    return std::nullopt;
  }
  const auto& window = it->second;
  if (current_frame >= window.timesteps) {
    return std::nullopt;
  }
  return window.window_start + static_cast<int64_t>(current_frame) * window.frame_step;
}

void ClearSourceFrameWindows() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_windows.clear();
}

}  // namespace sim2sim_debug
