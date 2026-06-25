#pragma once

#include <cstdint>
#include <string>

struct Sim2SimDebugConfig {
  bool enabled = false;
  bool write_csv = false;
  bool publish_zmq = false;
  std::string logs_dir;
};

class Sim2SimDebugHook {
 public:
  Sim2SimDebugHook() = default;
  explicit Sim2SimDebugHook(Sim2SimDebugConfig config);

  bool Enabled() const;
  bool WriteCsv() const;
  bool PublishZmq() const;
  const std::string& LogsDir() const;

  void OnStreamFrameDecoded(int64_t stream_frame_index,
                            int64_t source_frame_index);
  void OnControlTickObserved(uint64_t tick_index,
                             int motion_frame,
                             int64_t source_frame_index);
  void OnControlTickApplied(uint64_t tick_index,
                            int64_t applied_source_frame_index);

 private:
  Sim2SimDebugConfig config_;
};
