#include "sim2sim_debug/sim2sim_debug.hpp"

#include <utility>

Sim2SimDebugHook::Sim2SimDebugHook(Sim2SimDebugConfig config)
    : config_(std::move(config)) {}

bool Sim2SimDebugHook::Enabled() const {
  return config_.enabled;
}

bool Sim2SimDebugHook::WriteCsv() const {
  return config_.enabled && config_.write_csv;
}

bool Sim2SimDebugHook::PublishZmq() const {
  return config_.enabled && config_.publish_zmq;
}

const std::string& Sim2SimDebugHook::LogsDir() const {
  return config_.logs_dir;
}

void Sim2SimDebugHook::OnStreamFrameDecoded(int64_t, int64_t) {
  if (!config_.enabled) {
    return;
  }
}

void Sim2SimDebugHook::OnControlTickObserved(uint64_t, int, int64_t) {
  if (!config_.enabled) {
    return;
  }
}

void Sim2SimDebugHook::OnControlTickApplied(uint64_t, int64_t) {
  if (!config_.enabled) {
    return;
  }
}
