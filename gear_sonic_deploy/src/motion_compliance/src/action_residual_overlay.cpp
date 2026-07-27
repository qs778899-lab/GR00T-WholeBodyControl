/**
 * @file action_residual_overlay.cpp
 * @brief Validated ONNX Runtime implementation of the portable residual overlay.
 */

#include "action_residual_overlay.hpp"

#include <onnxruntime_cxx_api.h>
#include <sha256.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

namespace universal_tracker::compliance {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kRootKey = "motion_compliance_action_residual";
constexpr std::string_view kArtifactSchema =
    "universal-tracker.action-residual.onnx.v1";
constexpr std::string_view kReleaseInput = "release_action_context";
constexpr std::string_view kConditionInput = "motion_compliance_condition";
constexpr std::string_view kActionOutput = "action_delta";

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

bool IsLowerSha256(std::string_view value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

void RequireSha256(std::string_view value, std::string_view label) {
  if (!IsLowerSha256(value)) {
    Fail(std::string(label) +
         " must contain exactly 64 lowercase hexadecimal characters");
  }
}

std::string Sha256Bytes(std::string_view bytes) {
  char digest[SHA256_DIGEST_STRING_LENGTH] = {};
  if (SHA256Data(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(),
          digest) == nullptr) {
    Fail("SHA-256 byte digest failed");
  }
  return digest;
}

std::filesystem::path RequireRegularFile(
    const std::filesystem::path& path, std::string_view label) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status)) {
    Fail(std::string(label) + " must be a regular non-symlink file: " +
         path.string());
  }
  auto canonical = std::filesystem::canonical(path, error);
  if (error) {
    Fail(std::string(label) + " cannot be canonicalized: " + path.string());
  }
  return canonical;
}

std::filesystem::path RequireDirectory(
    const std::filesystem::path& path, std::string_view label) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_directory(status) ||
      std::filesystem::is_symlink(status)) {
    Fail(std::string(label) + " must be a real non-symlink directory: " +
         path.string());
  }
  auto canonical = std::filesystem::canonical(path, error);
  if (error) {
    Fail(std::string(label) + " cannot be canonicalized: " + path.string());
  }
  return canonical;
}

std::string Sha256File(
    const std::filesystem::path& path, std::string_view label) {
  const auto regular = RequireRegularFile(path, label);
  char digest[SHA256_DIGEST_STRING_LENGTH] = {};
  if (SHA256File(regular.c_str(), digest) == nullptr) {
    Fail(std::string("failed to hash ") + std::string(label) + ": " +
         regular.string());
  }
  return digest;
}

std::string ReadTextFile(
    const std::filesystem::path& path, std::string_view label) {
  const auto regular = RequireRegularFile(path, label);
  std::ifstream stream(regular, std::ios::binary);
  if (!stream) Fail(std::string("failed to open ") + std::string(label));
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    Fail(std::string("failed to read ") + std::string(label));
  }
  return contents.str();
}

template <typename T>
T RequiredYaml(const YAML::Node& mapping, const char* key, std::string_view label) {
  const auto value = mapping[key];
  if (!value || value.IsNull()) {
    Fail(std::string(label) + "." + key + " is required");
  }
  try {
    return value.as<T>();
  } catch (const YAML::Exception& error) {
    Fail(std::string(label) + "." + key + " is invalid: " + error.what());
  }
}

void RequireYamlMapping(const YAML::Node& value, std::string_view label) {
  if (!value || !value.IsMap()) {
    Fail(std::string(label) + " must be a mapping");
  }
}

void RequireYamlKeys(
    const YAML::Node& mapping, const std::set<std::string>& expected,
    std::string_view label) {
  RequireYamlMapping(mapping, label);
  std::set<std::string> actual;
  for (const auto& entry : mapping) {
    if (!entry.first.IsScalar()) {
      Fail(std::string(label) + " contains a non-scalar key");
    }
    actual.insert(entry.first.as<std::string>());
  }
  if (actual != expected) {
    std::ostringstream message;
    message << label << " keys differ";
    Fail(message.str());
  }
}

std::vector<std::string> YamlStringList(
    const YAML::Node& value, std::string_view label) {
  if (!value || !value.IsSequence() || value.size() == 0) {
    Fail(std::string(label) + " must be a non-empty sequence");
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  std::set<std::string> seen;
  for (const auto& item : value) {
    if (!item.IsScalar()) Fail(std::string(label) + " contains a non-string");
    const auto name = item.as<std::string>();
    if (name.empty() || !seen.insert(name).second) {
      Fail(std::string(label) + " contains an empty or duplicate identifier");
    }
    result.push_back(name);
  }
  return result;
}

std::vector<ContextField> YamlContextLayout(
    const YAML::Node& value, std::string_view label) {
  if (!value || !value.IsSequence() || value.size() == 0) {
    Fail(std::string(label) + " must be a non-empty sequence");
  }
  std::vector<ContextField> result;
  result.reserve(value.size());
  std::set<std::string> seen;
  std::size_t offset = 0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto entry = value[index];
    RequireYamlKeys(entry, {"name", "width"}, std::string(label) + " entry");
    const auto name = RequiredYaml<std::string>(entry, "name", label);
    const auto width = RequiredYaml<std::size_t>(entry, "width", label);
    if (name.empty() || width == 0 || !seen.insert(name).second) {
      Fail(std::string(label) +
           " entries must have unique names and positive widths");
    }
    if (offset > std::numeric_limits<std::size_t>::max() - width) {
      Fail(std::string(label) + " width overflow");
    }
    result.push_back(ContextField{.name = name, .offset = offset, .width = width});
    offset += width;
  }
  return result;
}

std::vector<float> YamlFiniteFloatList(
    const YAML::Node& value, std::string_view label) {
  if (!value || !value.IsSequence() || value.size() == 0) {
    Fail(std::string(label) + " must be a non-empty sequence");
  }
  std::vector<float> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    const auto parsed = item.as<double>();
    if (!std::isfinite(parsed) ||
        parsed < -std::numeric_limits<float>::max() ||
        parsed > std::numeric_limits<float>::max()) {
      Fail(std::string(label) + " must contain finite float32 values");
    }
    result.push_back(static_cast<float>(parsed));
  }
  return result;
}

struct DeclaredReleaseArtifact {
  std::string name;
  std::string sha256;
};

std::vector<DeclaredReleaseArtifact> YamlReleaseArtifacts(
    const YAML::Node& value, std::string_view label) {
  if (!value || !value.IsSequence() || value.size() == 0) {
    Fail(std::string(label) + " must be a non-empty sequence");
  }
  std::vector<DeclaredReleaseArtifact> result;
  result.reserve(value.size());
  std::set<std::string> seen;
  for (const auto& entry : value) {
    RequireYamlKeys(entry, {"name", "sha256"}, std::string(label) + " entry");
    auto name = RequiredYaml<std::string>(entry, "name", label);
    auto sha256 = RequiredYaml<std::string>(entry, "sha256", label);
    if (name.empty() || !seen.insert(name).second) {
      Fail(std::string(label) + " names must be unique and non-empty");
    }
    RequireSha256(sha256, std::string(label) + " digest");
    result.push_back(
        DeclaredReleaseArtifact{.name = std::move(name), .sha256 = std::move(sha256)});
  }
  return result;
}

void RequireFinitePositive(double value, std::string_view label) {
  if (!std::isfinite(value) || value <= 0.0) {
    Fail(std::string(label) + " must be finite and positive");
  }
}

std::size_t CheckedProduct(
    std::initializer_list<std::size_t> factors, std::string_view label) {
  std::size_t product = 1;
  for (const auto factor : factors) {
    if (factor != 0 && product > std::numeric_limits<std::size_t>::max() / factor) {
      Fail(std::string(label) + " size overflow");
    }
    product *= factor;
  }
  return product;
}

std::size_t PositiveJsonSize(const Json& value, std::string_view label) {
  if (!value.is_number_unsigned() && !value.is_number_integer()) {
    Fail(std::string(label) + " must be a positive integer");
  }
  const auto signed_value = value.get<std::int64_t>();
  if (signed_value <= 0) Fail(std::string(label) + " must be positive");
  return static_cast<std::size_t>(signed_value);
}

std::vector<std::string> JsonStringList(
    const Json& value, std::string_view label) {
  if (!value.is_array() || value.empty()) {
    Fail(std::string(label) + " must be a non-empty array");
  }
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& item : value) {
    if (!item.is_string()) Fail(std::string(label) + " contains a non-string");
    const auto name = item.get<std::string>();
    if (name.empty() || !seen.insert(name).second) {
      Fail(std::string(label) + " contains an empty or duplicate identifier");
    }
    result.push_back(name);
  }
  return result;
}

void RequireEqual(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected, std::string_view label) {
  if (actual != expected) Fail(std::string(label) + " differs from host contract");
}

std::size_t ValidateHostContract(const ActionResidualHostContract& contract) {
  if (contract.condition_width == 0) Fail("host condition width must be positive");
  if (contract.context_layout.empty()) Fail("host context layout must not be empty");
  std::size_t next_offset = 0;
  std::set<std::string> context_names;
  for (const auto& field : contract.context_layout) {
    if (field.name.empty() || field.width == 0 || field.offset != next_offset ||
        !context_names.insert(field.name).second) {
      Fail("host context layout must be unique, positive, contiguous, and ordered");
    }
    if (next_offset > std::numeric_limits<std::size_t>::max() - field.width) {
      Fail("host context width overflow");
    }
    next_offset += field.width;
  }
  if (contract.site_layout.empty() || contract.action_layout.empty()) {
    Fail("host site/action layouts must not be empty");
  }
  const auto unique_layout = [](const std::vector<std::string>& names) {
    return std::all_of(names.begin(), names.end(), [](const std::string& name) {
             return !name.empty();
           }) && std::set<std::string>(names.begin(), names.end()).size() == names.size();
  };
  if (!unique_layout(contract.site_layout) || !unique_layout(contract.action_layout)) {
    Fail("host site/action layouts must contain unique non-empty identifiers");
  }
  if (contract.release_artifacts.empty()) {
    Fail("host release artifact pins must not be empty");
  }
  std::set<std::string> artifact_names;
  for (const auto& artifact : contract.release_artifacts) {
    if (artifact.name.empty() || !artifact_names.insert(artifact.name).second) {
      Fail("host release artifact names must be unique and non-empty");
    }
    RequireSha256(artifact.sha256, "host release artifact digest");
  }
  return next_offset;
}

Json ParseStrictJson(const std::filesystem::path& path) {
  const auto text = ReadTextFile(path, "artifact metadata");
  try {
    return Json::parse(text, nullptr, true, true);
  } catch (const Json::exception& error) {
    Fail(std::string("artifact metadata JSON is invalid: ") + error.what());
  }
}

std::string MetadataDigest(const Json& metadata) {
  Json payload = metadata;
  payload.erase("metadata_sha256");
  return Sha256Bytes(payload.dump(-1, ' ', false, Json::error_handler_t::strict) + "\n");
}

void RequireJsonObjectKeys(
    const Json& object, const std::set<std::string>& expected,
    std::string_view label) {
  if (!object.is_object()) Fail(std::string(label) + " must be an object");
  std::set<std::string> actual;
  for (const auto& [key, unused] : object.items()) {
    (void)unused;
    actual.insert(key);
  }
  if (actual != expected) Fail(std::string(label) + " keys differ");
}

void ValidateOrtTensor(
    Ort::Session& session, bool input, std::size_t index,
    std::string_view expected_name, std::size_t expected_width) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  if (name.get() == nullptr || expected_name != name.get()) {
    Fail(std::string(input ? "input" : "output") + " tensor name differs");
  }
  const auto info = input ? session.GetInputTypeInfo(index)
                          : session.GetOutputTypeInfo(index);
  const auto tensor = info.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Fail(std::string(expected_name) + " must use float32");
  }
  const auto shape = tensor.GetShape();
  if (shape.size() != 3 || shape[2] != static_cast<std::int64_t>(expected_width) ||
      (shape[0] != -1 && shape[0] != 0) || (shape[1] != -1 && shape[1] != 0)) {
    Fail(std::string(expected_name) + " shape must be dynamic [B,S,width]");
  }
}

void SetError(std::string* target, std::string_view message) noexcept {
  if (target == nullptr) return;
  try {
    target->assign(message);
  } catch (...) {
  }
}

}  // namespace

struct ActionResidualOverlay::Impl {
  bool enabled = false;
  std::size_t release_context_width = 0;
  std::size_t condition_width = 0;
  std::size_t action_width = 0;
  float max_abs_delta = 0.0F;
  std::vector<float> default_enabled_condition;
  std::unique_ptr<Ort::Env> environment;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::atomic<std::uint64_t> inference_calls{0};
};

ActionResidualOverlay::ActionResidualOverlay(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ActionResidualOverlay::~ActionResidualOverlay() = default;
ActionResidualOverlay::ActionResidualOverlay(ActionResidualOverlay&&) noexcept = default;
ActionResidualOverlay& ActionResidualOverlay::operator=(
    ActionResidualOverlay&&) noexcept = default;

std::unique_ptr<ActionResidualOverlay> ActionResidualOverlay::LoadFromConfig(
    const std::filesystem::path& config_path,
    const ActionResidualHostContract& host_contract) {
  const auto context_width = ValidateHostContract(host_contract);
  const auto config_file = RequireRegularFile(config_path, "residual overlay config");
  YAML::Node document;
  try {
    document = YAML::LoadFile(config_file.string());
  } catch (const YAML::Exception& error) {
    Fail(std::string("residual overlay YAML is invalid: ") + error.what());
  }
  RequireYamlKeys(document, {std::string(kRootKey)}, "overlay root");
  const auto config = document[std::string(kRootKey)];
  RequireYamlMapping(config, "motion compliance overlay");
  const bool enabled = RequiredYaml<bool>(config, "enabled", "overlay");

  auto impl = std::make_unique<Impl>();
  impl->enabled = enabled;
  impl->release_context_width = context_width;
  impl->condition_width = host_contract.condition_width;
  impl->action_width = host_contract.action_layout.size();
  if (!enabled) {
    return std::unique_ptr<ActionResidualOverlay>(
        new ActionResidualOverlay(std::move(impl)));
  }

  RequireYamlKeys(
      config,
      {"enabled", "artifact_directory", "metadata_sha256", "checkpoint_sha256",
       "checkpoint_global_step", "schema", "max_abs_delta", "context_layout",
       "condition_width", "default_enabled_condition", "site_layout",
       "action_layout", "release_artifacts"},
      "enabled overlay");
  const auto artifact_directory = RequireDirectory(
      RequiredYaml<std::string>(config, "artifact_directory", "overlay"),
      "artifact directory");
  const auto expected_metadata_sha =
      RequiredYaml<std::string>(config, "metadata_sha256", "overlay");
  const auto expected_checkpoint_sha =
      RequiredYaml<std::string>(config, "checkpoint_sha256", "overlay");
  const auto expected_step =
      RequiredYaml<std::int64_t>(config, "checkpoint_global_step", "overlay");
  const auto expected_schema = RequiredYaml<std::string>(config, "schema", "overlay");
  const auto max_abs_delta = RequiredYaml<double>(config, "max_abs_delta", "overlay");
  RequireSha256(expected_metadata_sha, "overlay metadata_sha256");
  RequireSha256(expected_checkpoint_sha, "overlay checkpoint_sha256");
  if (expected_step <= 0) Fail("overlay checkpoint_global_step must be positive");
  if (expected_schema != kArtifactSchema) Fail("overlay schema is unsupported");
  RequireFinitePositive(max_abs_delta, "overlay max_abs_delta");

  const auto config_context =
      YamlContextLayout(config["context_layout"], "context_layout");
  if (config_context.size() != host_contract.context_layout.size()) {
    Fail("overlay context layout length differs from host contract");
  }
  for (std::size_t index = 0; index < config_context.size(); ++index) {
    const auto& actual = config_context[index];
    const auto& expected = host_contract.context_layout[index];
    if (actual.name != expected.name || actual.offset != expected.offset ||
        actual.width != expected.width) {
      Fail("overlay context layout differs from host contract");
    }
  }
  const auto condition_width =
      RequiredYaml<std::size_t>(config, "condition_width", "overlay");
  if (condition_width != host_contract.condition_width) {
    Fail("overlay condition width differs from host contract");
  }
  impl->default_enabled_condition = YamlFiniteFloatList(
      config["default_enabled_condition"], "default_enabled_condition");
  if (impl->default_enabled_condition.size() != condition_width ||
      impl->default_enabled_condition.front() != 1.0F ||
      std::any_of(
          impl->default_enabled_condition.begin() + 1,
          impl->default_enabled_condition.end(),
          [](float value) { return value <= 0.0F; })) {
    Fail(
        "default enabled condition must encode [1, positive threshold, "
        "positive Kp] at the declared width");
  }

  const auto config_sites = YamlStringList(config["site_layout"], "site_layout");
  const auto config_actions =
      YamlStringList(config["action_layout"], "action_layout");
  RequireEqual(config_sites, host_contract.site_layout, "site layout");
  RequireEqual(config_actions, host_contract.action_layout, "action layout");

  const auto config_release =
      YamlReleaseArtifacts(config["release_artifacts"], "release_artifacts");
  if (config_release.size() != host_contract.release_artifacts.size()) {
    Fail("overlay release artifact count differs from host-owned pins");
  }
  for (std::size_t index = 0; index < config_release.size(); ++index) {
    const auto& declared = config_release[index];
    const auto& host = host_contract.release_artifacts[index];
    if (declared.name != host.name || declared.sha256 != host.sha256) {
      Fail("overlay release artifact identity differs from host-owned pins");
    }
    if (Sha256File(host.path, "host release artifact " + host.name) != host.sha256) {
      Fail("runtime release artifact differs from the pinned host base");
    }
  }

  const auto metadata_path = artifact_directory / "action_residual.metadata.json";
  const auto metadata = ParseStrictJson(metadata_path);
  RequireJsonObjectKeys(
      metadata,
      {"schema", "model_kind", "source_checkpoint", "model", "inputs", "output",
       "network", "context_layout", "site_layout", "action_layout",
       "framework_versions", "metadata_sha256"},
      "artifact metadata");
  if (!metadata["metadata_sha256"].is_string() ||
      metadata["metadata_sha256"].get<std::string>() != expected_metadata_sha ||
      MetadataDigest(metadata) != expected_metadata_sha) {
    Fail("artifact metadata digest differs from the externally pinned digest");
  }
  if (metadata["schema"] != expected_schema ||
      metadata["model_kind"] != "bounded_action_residual") {
    Fail("artifact schema/model kind differs");
  }
  const auto& source = metadata["source_checkpoint"];
  if (!source.is_object() || source.value("sha256", "") != expected_checkpoint_sha ||
      source.value("global_step", std::int64_t{0}) != expected_step) {
    Fail("artifact source checkpoint provenance differs");
  }

  const auto& inputs = metadata["inputs"];
  if (!inputs.is_object() || !inputs.contains(kReleaseInput) ||
      !inputs.contains(kConditionInput) ||
      PositiveJsonSize(inputs[std::string(kReleaseInput)]["width"],
                       "release context width") != context_width ||
      PositiveJsonSize(inputs[std::string(kConditionInput)]["width"],
                       "condition width") != condition_width) {
    Fail("artifact input widths differ from host contract");
  }
  const auto& output = metadata["output"];
  if (!output.is_object() || output.value("name", "") != kActionOutput ||
      PositiveJsonSize(output["width"], "action width") != config_actions.size()) {
    Fail("artifact output contract differs");
  }
  const auto& network = metadata["network"];
  if (!network.is_object() ||
      PositiveJsonSize(network["residual_context_width"], "residual context width") !=
          context_width + condition_width ||
      network.value("activation", "") != "silu" ||
      network.value("output_activation", "") != "tanh" ||
      network.value("max_abs_delta", 0.0) != max_abs_delta) {
    Fail("artifact network contract differs");
  }
  RequireEqual(JsonStringList(metadata["site_layout"], "metadata site_layout"),
               config_sites, "metadata site layout");
  RequireEqual(JsonStringList(metadata["action_layout"], "metadata action_layout"),
               config_actions, "metadata action layout");
  const auto& metadata_context = metadata["context_layout"];
  if (!metadata_context.is_array() ||
      metadata_context.size() != host_contract.context_layout.size()) {
    Fail("metadata context layout length differs");
  }
  for (std::size_t index = 0; index < host_contract.context_layout.size(); ++index) {
    const auto& field = host_contract.context_layout[index];
    const auto& actual = metadata_context[index];
    if (!actual.is_object() || actual.value("name", "") != field.name ||
        actual.value("offset", std::size_t{~0U}) != field.offset ||
        actual.value("width", std::size_t{0}) != field.width) {
      Fail("metadata context layout differs from host contract");
    }
  }

  const auto& model = metadata["model"];
  if (!model.is_object() || model.value("opset", 0) != 17) {
    Fail("artifact model opset must be exactly 17");
  }
  const auto model_file = model.value("file", std::string{});
  if (model_file.empty() || model_file.find('/') != std::string::npos ||
      model_file.find('\\') != std::string::npos || model_file == "." ||
      model_file == "..") {
    Fail("artifact model filename must be local");
  }
  const auto model_path = RequireRegularFile(artifact_directory / model_file, "residual model");
  const auto declared_model_sha = model.value("sha256", std::string{});
  RequireSha256(declared_model_sha, "artifact model SHA-256");
  if (Sha256File(model_path, "residual model") != declared_model_sha) {
    Fail("artifact residual model digest mismatch");
  }

  impl->max_abs_delta = static_cast<float>(max_abs_delta);
  impl->environment = std::make_unique<Ort::Env>(
      ORT_LOGGING_LEVEL_WARNING, "motion_compliance_action_residual");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  impl->session = std::make_unique<Ort::Session>(
      *impl->environment, model_path.c_str(), options);
  if (impl->session->GetInputCount() != 2 ||
      impl->session->GetOutputCount() != 1) {
    Fail("residual ONNX must expose exactly two inputs and one output");
  }
  ValidateOrtTensor(*impl->session, true, 0, kReleaseInput, context_width);
  ValidateOrtTensor(*impl->session, true, 1, kConditionInput, condition_width);
  ValidateOrtTensor(*impl->session, false, 0, kActionOutput, config_actions.size());
  return std::unique_ptr<ActionResidualOverlay>(
      new ActionResidualOverlay(std::move(impl)));
}

bool ActionResidualOverlay::enabled() const noexcept {
  return impl_ != nullptr && impl_->enabled;
}

std::size_t ActionResidualOverlay::release_context_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->release_context_width;
}

std::size_t ActionResidualOverlay::condition_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->condition_width;
}

std::size_t ActionResidualOverlay::action_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->action_width;
}

float ActionResidualOverlay::max_abs_delta() const noexcept {
  return impl_ == nullptr ? 0.0F : impl_->max_abs_delta;
}

std::span<const float> ActionResidualOverlay::default_enabled_condition() const noexcept {
  if (impl_ == nullptr) return {};
  return impl_->default_enabled_condition;
}

std::uint64_t ActionResidualOverlay::inference_calls() const noexcept {
  return impl_ == nullptr ? 0 : impl_->inference_calls.load(std::memory_order_relaxed);
}

bool ActionResidualOverlay::Compose(
    std::size_t batch, std::size_t sequence,
    std::span<const float> release_action_context,
    std::span<const float> condition, std::span<const float> release_action,
    std::span<const std::uint8_t> enabled_gate,
    std::span<float> composed_action, std::string* error) noexcept {
  try {
    if (error != nullptr) error->clear();
    if (impl_ == nullptr) Fail("residual overlay implementation is missing");
    if (batch == 0 || sequence == 0) {
      Fail("residual batch and sequence dimensions must be positive");
    }
    const auto rows = CheckedProduct({batch, sequence}, "residual rows");
    const auto action_values = CheckedProduct({rows, impl_->action_width}, "action");
    if (release_action.size() != action_values ||
        composed_action.size() != action_values) {
      Fail("release/composed action size differs from [B,S,A]");
    }
    std::memmove(
        composed_action.data(), release_action.data(),
        action_values * sizeof(float));
    if (!impl_->enabled) return true;
    if (enabled_gate.size() != rows) Fail("enabled gate size differs from [B,S]");
    bool any_enabled = false;
    for (const auto gate : enabled_gate) {
      if (gate > 1U) Fail("enabled gate must contain only 0 or 1");
      any_enabled = any_enabled || gate == 1U;
    }
    if (!any_enabled) return true;
    const auto context_values =
        CheckedProduct({rows, impl_->release_context_width}, "release context");
    const auto condition_values =
        CheckedProduct({rows, impl_->condition_width}, "condition");
    if (release_action_context.size() != context_values ||
        condition.size() != condition_values) {
      Fail("residual input size differs from [B,S,width]");
    }

    std::vector<float> safe_context(context_values, 0.0F);
    std::vector<float> safe_condition(condition_values, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      if (enabled_gate[row] == 0U) continue;
      const auto action_offset = row * impl_->action_width;
      const auto context_offset = row * impl_->release_context_width;
      const auto condition_offset = row * impl_->condition_width;
      for (std::size_t index = 0; index < impl_->action_width; ++index) {
        if (!std::isfinite(release_action[action_offset + index])) {
          Fail("enabled release action contains NaN or Inf");
        }
      }
      for (std::size_t index = 0; index < impl_->release_context_width; ++index) {
        const auto value = release_action_context[context_offset + index];
        if (!std::isfinite(value)) Fail("enabled release context contains NaN or Inf");
        safe_context[context_offset + index] = value;
      }
      for (std::size_t index = 0; index < impl_->condition_width; ++index) {
        const auto value = condition[condition_offset + index];
        if (!std::isfinite(value)) Fail("enabled condition contains NaN or Inf");
        safe_condition[condition_offset + index] = value;
      }
      if (safe_condition[condition_offset] != 1.0F ||
          std::any_of(
              safe_condition.begin() + static_cast<std::ptrdiff_t>(condition_offset + 1),
              safe_condition.begin() + static_cast<std::ptrdiff_t>(
                  condition_offset + impl_->condition_width),
              [](float value) { return value <= 0.0F; })) {
        Fail("enabled condition must encode [1, positive threshold, positive Kp]");
      }
    }

    std::array<std::int64_t, 3> context_shape{
        static_cast<std::int64_t>(batch), static_cast<std::int64_t>(sequence),
        static_cast<std::int64_t>(impl_->release_context_width)};
    std::array<std::int64_t, 3> condition_shape{
        static_cast<std::int64_t>(batch), static_cast<std::int64_t>(sequence),
        static_cast<std::int64_t>(impl_->condition_width)};
    auto context_tensor = Ort::Value::CreateTensor<float>(
        impl_->memory_info, safe_context.data(), safe_context.size(),
        context_shape.data(), context_shape.size());
    auto condition_tensor = Ort::Value::CreateTensor<float>(
        impl_->memory_info, safe_condition.data(), safe_condition.size(),
        condition_shape.data(), condition_shape.size());
    std::array<Ort::Value, 2> inputs{
        std::move(context_tensor), std::move(condition_tensor)};
    constexpr std::array<const char*, 2> input_names{
        "release_action_context", "motion_compliance_condition"};
    constexpr std::array<const char*, 1> output_names{"action_delta"};
    impl_->inference_calls.fetch_add(1, std::memory_order_relaxed);
    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      Fail("residual inference returned an invalid output count/type");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape != std::vector<std::int64_t>{
                            static_cast<std::int64_t>(batch),
                            static_cast<std::int64_t>(sequence),
                            static_cast<std::int64_t>(impl_->action_width)} ||
        output_info.GetElementCount() != action_values) {
      Fail("residual inference output shape/dtype differs from [B,S,A]");
    }
    const auto* delta = outputs[0].GetTensorData<float>();
    const auto tolerance = std::max(1.0e-7F, impl_->max_abs_delta * 1.0e-6F);
    for (std::size_t row = 0; row < rows; ++row) {
      if (enabled_gate[row] == 0U) continue;
      const auto offset = row * impl_->action_width;
      for (std::size_t index = 0; index < impl_->action_width; ++index) {
        const auto residual = delta[offset + index];
        if (!std::isfinite(residual) ||
            std::abs(residual) > impl_->max_abs_delta + tolerance) {
          Fail("residual inference exceeded its finite action-delta bound");
        }
        const auto bounded = std::clamp(
            residual, -impl_->max_abs_delta, impl_->max_abs_delta);
        const auto composed = release_action[offset + index] + bounded;
        if (!std::isfinite(composed)) Fail("composed action contains NaN or Inf");
        composed_action[offset + index] = composed;
      }
    }
    return true;
  } catch (const std::exception& exception) {
    if (release_action.size() == composed_action.size() && !release_action.empty()) {
      std::memmove(
          composed_action.data(), release_action.data(),
          release_action.size() * sizeof(float));
    }
    SetError(error, exception.what());
    return false;
  } catch (...) {
    if (release_action.size() == composed_action.size() && !release_action.empty()) {
      std::memmove(
          composed_action.data(), release_action.data(),
          release_action.size() * sizeof(float));
    }
    SetError(error, "unknown residual composition failure");
    return false;
  }
}

bool ActionResidualOverlay::EvaluateAnyPositiveGate(
    std::span<const double> values, std::size_t gate_value_count,
    double minimum, double maximum, bool* active, std::string* error) noexcept {
  try {
    if (error != nullptr) error->clear();
    if (active == nullptr) Fail("gate output pointer must not be null");
    *active = false;
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
      Fail("gate validation range is invalid");
    }
    if (gate_value_count > values.size()) {
      Fail("gate value count exceeds supplied controls");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
      const auto value = values[index];
      if (!std::isfinite(value) || value < minimum || value > maximum) {
        Fail("compliance control contains a non-finite or out-of-range value");
      }
      if (index < gate_value_count && value > 0.0) *active = true;
    }
    return true;
  } catch (const std::exception& exception) {
    if (active != nullptr) *active = false;
    SetError(error, exception.what());
    return false;
  } catch (...) {
    if (active != nullptr) *active = false;
    SetError(error, "unknown compliance gate failure");
    return false;
  }
}

}  // namespace universal_tracker::compliance
