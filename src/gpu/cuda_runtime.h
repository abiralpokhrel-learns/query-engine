#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace qe::gpu {

struct DeviceInfo {
  bool build_enabled = false;
  bool available = false;
  std::string name;
  int major = 0;
  int minor = 0;
  uint64_t memory_bytes = 0;
  int driver_version = 0;
  int runtime_version = 0;
  std::string error;
};

DeviceInfo QueryDevice();

struct Timings {
  double transfer_ms = 0.0;
  double kernel_ms = 0.0;
};

struct Reduction {
  bool ok = false;
  int64_t valid_count = 0;
  double sum = 0.0;
  std::string error;
};

// A reusable device workspace. Allocating device buffers once per aggregate
// operator keeps per-batch transfers and kernel launches from being dominated
// by cudaMalloc and event creation. The buffers grow only when a batch needs
// more capacity.
class Workspace {
public:
  Workspace();
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  bool available() const;
  std::string error() const;

  // Sums the non-null values in a contiguous Arrow primitive array. The
  // values pointer and validity bitmap are the Arrow buffers; offset is the
  // array's logical offset. Only the requested slice is copied to the GPU.
  Reduction SumDouble(const double* values, const uint8_t* validity, int64_t offset, int64_t length,
                      Timings* timings);
  Reduction SumFloat(const float* values, const uint8_t* validity, int64_t offset, int64_t length,
                     Timings* timings);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct SelfTest {
  bool ok = false;
  std::string message;
};

// Runs a small reduction on the GPU and verifies the result on the host. This
// is a real kernel execution, not just a device query.
SelfTest RunSelfTest();

} // namespace qe::gpu
