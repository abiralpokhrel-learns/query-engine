#include "gpu/cuda_runtime.h"

namespace qe::gpu {

struct Workspace::Impl {};

DeviceInfo QueryDevice() {
  DeviceInfo info;
  info.build_enabled = false;
  info.available = false;
  info.error = "CPU-only build";
  return info;
}

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;

bool Workspace::available() const {
  return false;
}

std::string Workspace::error() const {
  return "CUDA support is not compiled into this build";
}

Reduction Workspace::SumDouble(const double*, const uint8_t*, int64_t, int64_t, Timings*) {
  Reduction result;
  result.error = error();
  return result;
}

Reduction Workspace::SumFloat(const float*, const uint8_t*, int64_t, int64_t, Timings*) {
  Reduction result;
  result.error = error();
  return result;
}

SelfTest RunSelfTest() {
  SelfTest test;
  test.message = "CUDA support is not compiled into this build";
  return test;
}

} // namespace qe::gpu
