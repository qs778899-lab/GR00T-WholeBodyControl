find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(LIBMD_LIBRARY NAMES md REQUIRED)

add_library(motion_compliance_action_residual STATIC
  src/action_residual_overlay.cpp)

target_include_directories(motion_compliance_action_residual
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(motion_compliance_action_residual
  PUBLIC onnxruntime::onnxruntime
  PRIVATE yaml-cpp nlohmann_json::nlohmann_json ${LIBMD_LIBRARY})

set_target_properties(motion_compliance_action_residual PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

target_compile_options(motion_compliance_action_residual PRIVATE
  -Wall -Wextra -Wpedantic)

# The repository-wide deployment flags include -ffast-math, which permits the
# compiler to assume NaN/Inf do not exist and can invalidate this safety
# boundary's explicit finite-value checks. Keep the small optional CPU overlay
# strict without changing the released policy target's arithmetic flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(motion_compliance_action_residual PRIVATE
    -fno-fast-math)
endif()
