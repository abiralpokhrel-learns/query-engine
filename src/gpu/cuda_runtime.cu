#include "gpu/cuda_runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace qe::gpu {
namespace {

struct ReduceOutput {
  double sum;
  long long count;
};

std::string ErrorText(cudaError_t status) {
  const char* name = cudaGetErrorName(status);
  const char* message = cudaGetErrorString(status);
  if (name == nullptr)
    return message == nullptr ? "unknown CUDA error" : message;
  if (message == nullptr)
    return name;
  return std::string(name) + ": " + message;
}

template <typename T>
__global__ void ReduceSumKernel(const T* values, const uint8_t* validity, int bit_offset,
                                int64_t length, ReduceOutput* output) {
  __shared__ double partial_sum[256];
  __shared__ long long partial_count[256];

  const int thread = static_cast<int>(threadIdx.x);
  double local_sum = 0.0;
  long long local_count = 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + thread; i < length;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (validity != nullptr) {
      const int64_t bit = static_cast<int64_t>(bit_offset) + i;
      if ((validity[bit >> 3] & static_cast<uint8_t>(1U << (bit & 7))) == 0)
        continue;
    }
    local_sum += static_cast<double>(values[i]);
    ++local_count;
  }

  partial_sum[thread] = local_sum;
  partial_count[thread] = local_count;
  __syncthreads();

  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      partial_sum[thread] += partial_sum[thread + stride];
      partial_count[thread] += partial_count[thread + stride];
    }
    __syncthreads();
  }

  if (thread == 0) {
    atomicAdd(&output->sum, partial_sum[0]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&output->count),
              static_cast<unsigned long long>(partial_count[0]));
  }
}

} // namespace

struct Workspace::Impl {
  void* device_values = nullptr;
  size_t values_capacity = 0;
  uint8_t* device_validity = nullptr;
  size_t validity_capacity = 0;
  ReduceOutput* device_output = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t host_to_device_done = nullptr;
  cudaEvent_t kernel_done = nullptr;
  cudaEvent_t device_to_host_done = nullptr;
  std::string error;

  Impl() {
    Initialize();
  }

  ~Impl() {
    Cleanup();
  }

  void Cleanup() {
    if (device_values != nullptr)
      cudaFree(device_values);
    if (device_validity != nullptr)
      cudaFree(device_validity);
    if (device_output != nullptr)
      cudaFree(device_output);
    if (start != nullptr)
      cudaEventDestroy(start);
    if (host_to_device_done != nullptr)
      cudaEventDestroy(host_to_device_done);
    if (kernel_done != nullptr)
      cudaEventDestroy(kernel_done);
    if (device_to_host_done != nullptr)
      cudaEventDestroy(device_to_host_done);
  }

  void Initialize() {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
    status = cudaEventCreate(&start);
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
    status = cudaEventCreate(&host_to_device_done);
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
    status = cudaEventCreate(&kernel_done);
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
    status = cudaEventCreate(&device_to_host_done);
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
    status = cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(ReduceOutput));
    if (status != cudaSuccess) {
      error = ErrorText(status);
      return;
    }
  }

  bool EnsureCapacity(size_t values_bytes, size_t validity_bytes) {
    if (values_bytes > values_capacity) {
      if (device_values != nullptr) {
        cudaFree(device_values);
        device_values = nullptr;
        values_capacity = 0;
      }
      cudaError_t status = cudaMalloc(&device_values, values_bytes);
      if (status != cudaSuccess) {
        error = ErrorText(status);
        return false;
      }
      values_capacity = values_bytes;
    }
    if (validity_bytes > validity_capacity) {
      if (device_validity != nullptr) {
        cudaFree(device_validity);
        device_validity = nullptr;
        validity_capacity = 0;
      }
      cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_validity), validity_bytes);
      if (status != cudaSuccess) {
        error = ErrorText(status);
        return false;
      }
      validity_capacity = validity_bytes;
    }
    return true;
  }

  template <typename T>
  Reduction ReduceSum(const T* values, const uint8_t* validity, int64_t offset, int64_t length,
                      Timings* timings) {
    Reduction result;
    if (!error.empty()) {
      result.error = error;
      return result;
    }
    if (length <= 0) {
      result.ok = true;
      return result;
    }
    if (values == nullptr) {
      result.error = "cannot reduce a null values buffer";
      return result;
    }

    const size_t values_bytes = static_cast<size_t>(length) * sizeof(T);
    const int64_t validity_start = offset / 8;
    const int64_t validity_end = (offset + length + 7) / 8;
    const size_t validity_bytes = static_cast<size_t>(validity_end - validity_start);
    if (!EnsureCapacity(values_bytes, validity_bytes)) {
      result.error = error;
      return result;
    }

    auto fail = [&](cudaError_t status) {
      error = ErrorText(status);
      result.error = error;
      return result;
    };

    cudaError_t status = cudaMemset(device_output, 0, sizeof(ReduceOutput));
    if (status != cudaSuccess)
      return fail(status);
    status = cudaEventRecord(start);
    if (status != cudaSuccess)
      return fail(status);
    status = cudaMemcpy(device_values, values + offset, values_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      return fail(status);
    if (validity != nullptr) {
      status = cudaMemcpy(device_validity, validity + validity_start, validity_bytes,
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess)
        return fail(status);
    }
    status = cudaEventRecord(host_to_device_done);
    if (status != cudaSuccess)
      return fail(status);

    const int bit_offset = static_cast<int>(offset % 8);
    const int blocks =
        static_cast<int>(std::min<int64_t>((length + 255) / 256, static_cast<int64_t>(256)));
    ReduceSumKernel<T><<<blocks, 256>>>(static_cast<const T*>(device_values),
                                        validity == nullptr ? nullptr : device_validity, bit_offset,
                                        length, device_output);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return fail(status);
    status = cudaEventRecord(kernel_done);
    if (status != cudaSuccess)
      return fail(status);

    ReduceOutput output{};
    status = cudaMemcpy(&output, device_output, sizeof(ReduceOutput), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      return fail(status);
    status = cudaEventRecord(device_to_host_done);
    if (status != cudaSuccess)
      return fail(status);
    status = cudaEventSynchronize(device_to_host_done);
    if (status != cudaSuccess)
      return fail(status);

    if (timings != nullptr) {
      float host_to_device_ms = 0.0F;
      float kernel_ms = 0.0F;
      float device_to_host_ms = 0.0F;
      cudaEventElapsedTime(&host_to_device_ms, start, host_to_device_done);
      cudaEventElapsedTime(&kernel_ms, host_to_device_done, kernel_done);
      cudaEventElapsedTime(&device_to_host_ms, kernel_done, device_to_host_done);
      timings->transfer_ms = static_cast<double>(host_to_device_ms + device_to_host_ms);
      timings->kernel_ms = static_cast<double>(kernel_ms);
    }

    result.ok = true;
    result.valid_count = static_cast<int64_t>(output.count);
    result.sum = output.sum;
    return result;
  }
};

DeviceInfo QueryDevice() {
  DeviceInfo info;
  info.build_enabled = true;

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    info.error = ErrorText(status);
    return info;
  }
  if (device_count == 0) {
    info.error = "no CUDA devices found";
    return info;
  }

  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    info.error = ErrorText(status);
    return info;
  }

  info.available = true;
  info.name = properties.name;
  info.major = properties.major;
  info.minor = properties.minor;
  info.memory_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
  cudaDriverGetVersion(&info.driver_version);
  cudaRuntimeGetVersion(&info.runtime_version);
  return info;
}

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;

bool Workspace::available() const {
  return impl_ != nullptr && impl_->error.empty();
}

std::string Workspace::error() const {
  return impl_ == nullptr ? "CUDA workspace is not initialized" : impl_->error;
}

Reduction Workspace::SumDouble(const double* values, const uint8_t* validity, int64_t offset,
                               int64_t length, Timings* timings) {
  return impl_->ReduceSum<double>(values, validity, offset, length, timings);
}

Reduction Workspace::SumFloat(const float* values, const uint8_t* validity, int64_t offset,
                              int64_t length, Timings* timings) {
  return impl_->ReduceSum<float>(values, validity, offset, length, timings);
}

SelfTest RunSelfTest() {
  SelfTest test;
  DeviceInfo info = QueryDevice();
  if (!info.available) {
    test.message = info.error;
    return test;
  }

  Workspace workspace;
  if (!workspace.available()) {
    test.message = workspace.error();
    return test;
  }

  constexpr int64_t n = 2048;
  std::vector<double> values(static_cast<size_t>(n));
  std::vector<uint8_t> validity(static_cast<size_t>((n + 7) / 8), 0xFF);
  double expected_sum = 0.0;
  int64_t expected_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    values[static_cast<size_t>(i)] = static_cast<double>(i % 97) * 0.5;
    if (i % 11 == 0) {
      validity[static_cast<size_t>(i / 8)] &= static_cast<uint8_t>(~(1U << (i % 8)));
    } else {
      expected_sum += values[static_cast<size_t>(i)];
      ++expected_count;
    }
  }

  Timings timings;
  Reduction reduction = workspace.SumDouble(values.data(), validity.data(), 0, n, &timings);
  if (!reduction.ok) {
    test.message = reduction.error;
    return test;
  }
  if (reduction.valid_count != expected_count) {
    test.message = "device reduction returned the wrong non-null count";
    return test;
  }
  const double tolerance = 1e-9 * std::max(1.0, std::fabs(expected_sum));
  if (std::fabs(reduction.sum - expected_sum) > tolerance) {
    test.message = "device reduction returned the wrong sum";
    return test;
  }

  test.ok = true;
  test.message = "device reduction verified: " + std::to_string(reduction.valid_count) +
                 " non-null values, transfer_ms=" + std::to_string(timings.transfer_ms) +
                 " kernel_ms=" + std::to_string(timings.kernel_ms);
  return test;
}

} // namespace qe::gpu
