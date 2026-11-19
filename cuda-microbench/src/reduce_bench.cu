// Standalone CUDA reduction microbenchmark for qe.
//
// Compares four ways to sum an array of doubles on the GPU:
//   1. one strided accumulation per thread plus a global atomicAdd
//   2. a shared-memory tree reduction per block, then a small final reduction
//   3. warp-shuffle reduction with vectorized double2 loads, then a small
//      final reduction
//   4. CUB's DeviceReduce::Sum as a library baseline
//
// It also times three host/device transfer paths for the end-to-end query:
// pageable cudaMemcpy, pinned memory with one stream, and pinned memory split
// across four streams so copy and compute can overlap. Every result is checked
// against a long-double host reference before it is reported.

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;
constexpr int kItemsPerThread = 8;
constexpr int kMaxBlocks = 8192;
constexpr int kMaxStreamBlocks = 2048;
constexpr int kStreams = 4;

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = (call);                                                                  \
    if (status_ != cudaSuccess) {                                                                  \
      std::ostringstream message_;                                                                 \
      message_ << #call << " failed: " << cudaGetErrorString(status_);                             \
      throw std::runtime_error(message_.str());                                                    \
    }                                                                                              \
  } while (0)

struct Stats {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct KernelResult {
  Stats stats;
  double sum = 0.0;
  bool ok = false;
};

struct TransferResult {
  bool ran = false;
  Stats stats;
  double gbps = 0.0;
  double sum = 0.0;
  bool ok = false;
  std::string note;
};

struct FilterResult {
  Stats stats;
  double sum = 0.0;
  unsigned long long count = 0;
  bool ok = false;
};

struct SizeResult {
  size_t elements = 0;
  double bytes = 0.0;
  double reference = 0.0;
  KernelResult atomic;
  KernelResult block;
  KernelResult warp;
  KernelResult cub;
  FilterResult filter;
  TransferResult pageable;
  TransferResult pinned;
  TransferResult streams;
};

struct DeviceInfo {
  std::string name;
  int major = 0;
  int minor = 0;
  int sms = 0;
  size_t memory_bytes = 0;
  int memory_clock_khz = 0;
  int bus_width_bits = 0;
  double theoretical_gbps = 0.0;
};

Stats summarize(std::vector<double> samples) {
  if (samples.empty())
    return {};
  std::sort(samples.begin(), samples.end());
  Stats stats;
  stats.min_ms = samples.front();
  stats.max_ms = samples.back();
  const size_t middle = samples.size() / 2;
  stats.median_ms =
      (samples.size() % 2 == 0) ? 0.5 * (samples[middle - 1] + samples[middle]) : samples[middle];
  return stats;
}

template <typename Fn> Stats measure_cuda_events(int iterations, Fn&& fn) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    samples.push_back(static_cast<double>(elapsed));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  return summarize(std::move(samples));
}

template <typename Fn> Stats measure_wall_clock(int iterations, Fn&& fn) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return summarize(std::move(samples));
}

int grid_for(size_t work_items, int cap) {
  const size_t per_block = static_cast<size_t>(kBlock) * kItemsPerThread;
  const size_t blocks = (work_items + per_block - 1) / per_block;
  return static_cast<int>(std::min<size_t>(std::max<size_t>(blocks, 1), cap));
}

__device__ __forceinline__ double warp_reduce_sum(double value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void reduce_atomic_kernel(const double* __restrict__ input, double* __restrict__ output,
                                     size_t count) {
  double sum = 0.0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    sum += input[i];
  }
  atomicAdd(output, sum);
}

template <int Block>
__global__ void reduce_block_kernel(const double* __restrict__ input, double* __restrict__ partials,
                                    size_t count) {
  double sum = 0.0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * Block + threadIdx.x; i < count;
       i += static_cast<size_t>(Block) * gridDim.x) {
    sum += input[i];
  }

  __shared__ double shared[Block];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = Block / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void reduce_partials_kernel(const double* __restrict__ partials,
                                       double* __restrict__ output, int count) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    sum += partials[i];
  }
  sum = warp_reduce_sum(sum);

  // Launched with exactly kBlock threads, so this always has kBlock / 32 slots.
  __shared__ double warp_sums[kBlock / 32];
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = (threadIdx.x < kBlock / 32) ? warp_sums[threadIdx.x] : 0.0;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      *output = sum;
    }
  }
}

template <int Block>
__global__ void reduce_warp_kernel(const double* __restrict__ input, double* __restrict__ partials,
                                   size_t count) {
  constexpr int kWarps = Block / 32;
  double sum = 0.0;

  // double2 loads move 16 bytes per instruction and halve the index work.
  const size_t pairs = count / 2;
  const double2* vector_input = reinterpret_cast<const double2*>(input);
  for (size_t i = static_cast<size_t>(blockIdx.x) * Block + threadIdx.x; i < pairs;
       i += static_cast<size_t>(Block) * gridDim.x) {
    const double2 value = vector_input[i];
    sum += value.x + value.y;
  }
  if ((count & 1u) != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    sum += input[count - 1];
  }

  sum = warp_reduce_sum(sum);

  __shared__ double warp_sums[kWarps];
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = (threadIdx.x < kWarps) ? warp_sums[threadIdx.x] : 0.0;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      partials[blockIdx.x] = sum;
    }
  }
}

template <int Block>
__global__ void filter_aggregate_kernel(const double* __restrict__ input, size_t count,
                                        double threshold, double* __restrict__ output_sum,
                                        unsigned long long* __restrict__ output_count) {
  constexpr int kWarps = Block / 32;
  double sum = 0.0;
  unsigned long long hits = 0;

  const size_t pairs = count / 2;
  const double2* vector_input = reinterpret_cast<const double2*>(input);
  for (size_t i = static_cast<size_t>(blockIdx.x) * Block + threadIdx.x; i < pairs;
       i += static_cast<size_t>(Block) * gridDim.x) {
    const double2 value = vector_input[i];
    if (value.x > threshold) {
      sum += value.x;
      ++hits;
    }
    if (value.y > threshold) {
      sum += value.y;
      ++hits;
    }
  }
  if ((count & 1u) != 0 && blockIdx.x == 0 && threadIdx.x == 0 && input[count - 1] > threshold) {
    sum += input[count - 1];
    ++hits;
  }

  sum = warp_reduce_sum(sum);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    hits += __shfl_down_sync(0xffffffffu, hits, offset);
  }

  __shared__ double warp_sums[kWarps];
  __shared__ unsigned long long warp_counts[kWarps];
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
    warp_counts[threadIdx.x >> 5] = hits;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = (threadIdx.x < kWarps) ? warp_sums[threadIdx.x] : 0.0;
    hits = (threadIdx.x < kWarps) ? warp_counts[threadIdx.x] : 0;
    sum = warp_reduce_sum(sum);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      hits += __shfl_down_sync(0xffffffffu, hits, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(output_sum, sum);
      atomicAdd(output_count, hits);
    }
  }
}

void launch_warp_reduce(const double* input, size_t count, double* partials, double* output,
                        int blocks, cudaStream_t stream) {
  reduce_warp_kernel<kBlock><<<blocks, kBlock, 0, stream>>>(input, partials, count);
  reduce_partials_kernel<<<1, kBlock, 0, stream>>>(partials, output, blocks);
}

double reference_sum(const std::vector<double>& values) {
  long double sum = 0.0L;
  for (const double value : values) {
    sum += static_cast<long double>(value);
  }
  return static_cast<double>(sum);
}

double fetch_result(double* device_output) {
  double value = 0.0;
  CUDA_CHECK(cudaMemcpy(&value, device_output, sizeof(double), cudaMemcpyDeviceToHost));
  return value;
}

bool close_enough(double got, double reference) {
  const double scale = std::max(1.0, std::abs(reference));
  return std::abs(got - reference) <= 1e-6 * scale;
}

DeviceInfo query_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  DeviceInfo info;
  info.name = properties.name;
  info.major = properties.major;
  info.minor = properties.minor;
  info.sms = properties.multiProcessorCount;
  info.memory_bytes = properties.totalGlobalMem;
  info.memory_clock_khz = properties.memoryClockRate;
  info.bus_width_bits = properties.memoryBusWidth;
  // DDR memory transfers twice per clock; the bus is measured in bits.
  info.theoretical_gbps = 2.0 * static_cast<double>(info.memory_clock_khz) * 1e3 *
                          (static_cast<double>(info.bus_width_bits) / 8.0) / 1e9;
  return info;
}

std::string human_bytes(double bytes) {
  std::ostringstream out;
  if (bytes >= 1024.0 * 1024.0 * 1024.0) {
    out << std::fixed << std::setprecision(2) << bytes / (1024.0 * 1024.0 * 1024.0) << " GiB";
  } else if (bytes >= 1024.0 * 1024.0) {
    out << std::fixed << std::setprecision(1) << bytes / (1024.0 * 1024.0) << " MiB";
  } else {
    out << std::fixed << std::setprecision(1) << bytes / 1024.0 << " KiB";
  }
  return out.str();
}

std::string json_escape(const std::string& text) {
  std::string escaped;
  for (const char c : text) {
    if (c == '"' || c == '\\')
      escaped.push_back('\\');
    escaped.push_back(c);
  }
  return escaped;
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm parts{};
  gmtime_r(&now, &parts);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
  return buffer;
}

SizeResult run_size(const DeviceInfo& device, size_t elements, int iterations,
                    int transfer_iterations, bool do_transfer, bool do_cub) {
  SizeResult result;
  result.elements = elements;
  result.bytes = static_cast<double>(elements) * sizeof(double);

  std::vector<double> host_data(elements);
  std::mt19937_64 generator(20261118u);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (double& value : host_data) {
    value = distribution(generator);
  }
  result.reference = reference_sum(host_data);

  constexpr double kFilterThreshold = 0.0;
  double reference_filter_sum = 0.0;
  unsigned long long reference_filter_count = 0;
  for (const double value : host_data) {
    if (value > kFilterThreshold) {
      reference_filter_sum += value;
      ++reference_filter_count;
    }
  }

  double* device_input = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, static_cast<size_t>(result.bytes)));
  double* device_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&device_partials, kMaxBlocks * sizeof(double)));
  double* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(double)));
  unsigned long long* device_filter_count = nullptr;
  CUDA_CHECK(cudaMalloc(&device_filter_count, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpy(device_input, host_data.data(), static_cast<size_t>(result.bytes),
                        cudaMemcpyHostToDevice));

  const int scalar_blocks = grid_for(elements, kMaxBlocks);
  const int warp_blocks = grid_for(elements / 2, kMaxBlocks);
  // The naive path is deliberately shaped like the first version most people
  // write: up to one element per thread and one atomicAdd per thread. The grid
  // is capped, so larger inputs loop but still issue one atomic per thread.
  const int atomic_blocks =
      static_cast<int>(std::min<size_t>((elements + kBlock - 1) / kBlock, 65535));

  // Warm every path once so the first timed run is not dominated by page faults
  // or context setup.
  CUDA_CHECK(cudaMemset(device_output, 0, sizeof(double)));
  reduce_atomic_kernel<<<atomic_blocks, kBlock>>>(device_input, device_output, elements);
  reduce_block_kernel<kBlock><<<scalar_blocks, kBlock>>>(device_input, device_partials, elements);
  reduce_partials_kernel<<<1, kBlock>>>(device_partials, device_output, scalar_blocks);
  launch_warp_reduce(device_input, elements, device_partials, device_output, warp_blocks, 0);
  filter_aggregate_kernel<kBlock><<<warp_blocks, kBlock>>>(device_input, elements, kFilterThreshold,
                                                           device_output, device_filter_count);
  CUDA_CHECK(cudaDeviceSynchronize());

  result.atomic.stats = measure_cuda_events(iterations, [&] {
    CUDA_CHECK(cudaMemset(device_output, 0, sizeof(double)));
    reduce_atomic_kernel<<<atomic_blocks, kBlock>>>(device_input, device_output, elements);
  });
  result.atomic.sum = fetch_result(device_output);
  result.atomic.ok = close_enough(result.atomic.sum, result.reference);

  result.block.stats = measure_cuda_events(iterations, [&] {
    reduce_block_kernel<kBlock><<<scalar_blocks, kBlock>>>(device_input, device_partials, elements);
    reduce_partials_kernel<<<1, kBlock>>>(device_partials, device_output, scalar_blocks);
  });
  result.block.sum = fetch_result(device_output);
  result.block.ok = close_enough(result.block.sum, result.reference);

  result.warp.stats = measure_cuda_events(iterations, [&] {
    launch_warp_reduce(device_input, elements, device_partials, device_output, warp_blocks, 0);
  });
  result.warp.sum = fetch_result(device_output);
  result.warp.ok = close_enough(result.warp.sum, result.reference);

  result.filter.stats = measure_cuda_events(iterations, [&] {
    CUDA_CHECK(cudaMemset(device_output, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(device_filter_count, 0, sizeof(unsigned long long)));
    filter_aggregate_kernel<kBlock><<<warp_blocks, kBlock>>>(
        device_input, elements, kFilterThreshold, device_output, device_filter_count);
  });
  result.filter.sum = fetch_result(device_output);
  CUDA_CHECK(cudaMemcpy(&result.filter.count, device_filter_count, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  result.filter.ok = result.filter.count == reference_filter_count &&
                     close_enough(result.filter.sum, reference_filter_sum);

  if (do_cub) {
    size_t temp_bytes = 0;
    const cudaError_t query = cub::DeviceReduce::Sum(nullptr, temp_bytes, device_input,
                                                     device_output, static_cast<int>(elements));
    if (query == cudaSuccess && temp_bytes > 0) {
      void* temp_storage = nullptr;
      CUDA_CHECK(cudaMalloc(&temp_storage, temp_bytes));
      result.cub.stats = measure_cuda_events(iterations, [&] {
        const cudaError_t status = cub::DeviceReduce::Sum(
            temp_storage, temp_bytes, device_input, device_output, static_cast<int>(elements), 0);
        if (status != cudaSuccess) {
          throw std::runtime_error(std::string("CUB reduce failed: ") + cudaGetErrorString(status));
        }
      });
      result.cub.sum = fetch_result(device_output);
      result.cub.ok = close_enough(result.cub.sum, result.reference);
      CUDA_CHECK(cudaFree(temp_storage));
    } else {
      result.cub.ok = false;
    }
  }

  if (do_transfer) {
    double* pinned_input = nullptr;
    const cudaError_t pinned_status =
        cudaHostAlloc(reinterpret_cast<void**>(&pinned_input), static_cast<size_t>(result.bytes),
                      cudaHostAllocDefault);
    if (pinned_status != cudaSuccess) {
      result.pageable.ran = true;
      result.pageable.stats = measure_wall_clock(transfer_iterations, [&] {
        CUDA_CHECK(cudaMemcpy(device_input, host_data.data(), static_cast<size_t>(result.bytes),
                              cudaMemcpyHostToDevice));
        launch_warp_reduce(device_input, elements, device_partials, device_output, warp_blocks, 0);
        (void)fetch_result(device_output);
      });
      result.pageable.sum = fetch_result(device_output);
      result.pageable.ok = close_enough(result.pageable.sum, result.reference);
      result.pageable.gbps = 2.0 * result.bytes / (result.pageable.stats.median_ms * 1e6);
      result.pinned.note = "pinned allocation failed";
      result.streams.note = "pinned allocation failed";
      cudaGetLastError();
    } else {
      std::memcpy(pinned_input, host_data.data(), static_cast<size_t>(result.bytes));
      double* pinned_result = nullptr;
      CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_result), sizeof(double),
                               cudaHostAllocDefault));

      result.pageable.ran = true;
      result.pageable.stats = measure_wall_clock(transfer_iterations, [&] {
        CUDA_CHECK(cudaMemcpy(device_input, host_data.data(), static_cast<size_t>(result.bytes),
                              cudaMemcpyHostToDevice));
        launch_warp_reduce(device_input, elements, device_partials, device_output, warp_blocks, 0);
        CUDA_CHECK(
            cudaMemcpy(pinned_result, device_output, sizeof(double), cudaMemcpyDeviceToHost));
      });
      result.pageable.sum = *pinned_result;
      result.pageable.ok = close_enough(result.pageable.sum, result.reference);
      result.pageable.gbps = 2.0 * result.bytes / (result.pageable.stats.median_ms * 1e6);

      cudaStream_t stream = nullptr;
      CUDA_CHECK(cudaStreamCreate(&stream));
      result.pinned.ran = true;
      result.pinned.stats = measure_wall_clock(transfer_iterations, [&] {
        CUDA_CHECK(cudaMemcpyAsync(device_input, pinned_input, static_cast<size_t>(result.bytes),
                                   cudaMemcpyHostToDevice, stream));
        launch_warp_reduce(device_input, elements, device_partials, device_output, warp_blocks,
                           stream);
        CUDA_CHECK(cudaMemcpyAsync(pinned_result, device_output, sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
      });
      result.pinned.sum = *pinned_result;
      result.pinned.ok = close_enough(result.pinned.sum, result.reference);
      result.pinned.gbps = 2.0 * result.bytes / (result.pinned.stats.median_ms * 1e6);
      CUDA_CHECK(cudaStreamDestroy(stream));

      if (elements >= static_cast<size_t>(kStreams) * 2) {
        const size_t chunk = (elements / kStreams) & ~static_cast<size_t>(1);
        double* pinned_chunk_results = nullptr;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_chunk_results),
                                 kStreams * sizeof(double), cudaHostAllocDefault));
        cudaStream_t streams[kStreams]{};
        double* chunk_outputs[kStreams]{};
        double* chunk_partials[kStreams]{};
        for (int index = 0; index < kStreams; ++index) {
          CUDA_CHECK(cudaStreamCreate(&streams[index]));
          CUDA_CHECK(cudaMalloc(&chunk_outputs[index], sizeof(double)));
          CUDA_CHECK(cudaMalloc(&chunk_partials[index], kMaxStreamBlocks * sizeof(double)));
        }

        const auto launch_streams = [&] {
          for (int index = 0; index < kStreams; ++index) {
            const size_t offset = static_cast<size_t>(index) * chunk;
            const size_t count = (index == kStreams - 1) ? elements - offset : chunk;
            const int blocks = grid_for(count / 2, kMaxStreamBlocks);
            CUDA_CHECK(cudaMemcpyAsync(device_input + offset, pinned_input + offset,
                                       count * sizeof(double), cudaMemcpyHostToDevice,
                                       streams[index]));
            launch_warp_reduce(device_input + offset, count, chunk_partials[index],
                               chunk_outputs[index], blocks, streams[index]);
            CUDA_CHECK(cudaMemcpyAsync(pinned_chunk_results + index, chunk_outputs[index],
                                       sizeof(double), cudaMemcpyDeviceToHost, streams[index]));
          }
        };

        launch_streams();
        CUDA_CHECK(cudaDeviceSynchronize());
        result.streams.ran = true;
        result.streams.stats = measure_wall_clock(transfer_iterations, launch_streams);
        result.streams.sum =
            std::accumulate(pinned_chunk_results, pinned_chunk_results + kStreams, 0.0);
        result.streams.ok = close_enough(result.streams.sum, result.reference);
        result.streams.gbps = 2.0 * result.bytes / (result.streams.stats.median_ms * 1e6);

        for (int index = 0; index < kStreams; ++index) {
          CUDA_CHECK(cudaFree(chunk_outputs[index]));
          CUDA_CHECK(cudaFree(chunk_partials[index]));
          CUDA_CHECK(cudaStreamDestroy(streams[index]));
        }
        CUDA_CHECK(cudaFreeHost(pinned_chunk_results));
      } else {
        result.streams.note = "too few elements";
      }

      CUDA_CHECK(cudaFreeHost(pinned_result));
      CUDA_CHECK(cudaFreeHost(pinned_input));
    }
  }

  CUDA_CHECK(cudaFree(device_filter_count));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_partials));
  CUDA_CHECK(cudaFree(device_input));
  if (!result.cub.ok && do_cub) {
    // Nothing to do: cub.ok stays false when CUB could not be set up.
  }
  return result;
}

void write_json(const std::string& path, const DeviceInfo& device,
                const std::vector<SizeResult>& results, int iterations, int transfer_iterations) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("cannot open " + path);
  out << std::setprecision(9);
  out << "{\n";
  out << "  \"generated_at_utc\": \"" << utc_timestamp() << "\",\n";
  out << "  \"device\": {\n";
  out << "    \"name\": \"" << json_escape(device.name) << "\",\n";
  out << "    \"compute_capability\": \"" << device.major << "." << device.minor << "\",\n";
  out << "    \"sms\": " << device.sms << ",\n";
  out << "    \"memory_bytes\": " << device.memory_bytes << ",\n";
  out << "    \"memory_clock_khz\": " << device.memory_clock_khz << ",\n";
  out << "    \"bus_width_bits\": " << device.bus_width_bits << ",\n";
  out << "    \"theoretical_gbps\": " << device.theoretical_gbps << "\n";
  out << "  },\n";
  out << "  \"config\": {\"kernel_iterations\": " << iterations
      << ", \"transfer_iterations\": " << transfer_iterations << "},\n";
  out << "  \"results\": [\n";
  for (size_t index = 0; index < results.size(); ++index) {
    const SizeResult& size = results[index];
    out << "    {\n";
    out << "      \"elements\": " << size.elements << ",\n";
    out << "      \"bytes\": " << size.bytes << ",\n";
    out << "      \"reference_sum\": " << size.reference << ",\n";
    out << "      \"kernels\": {\n";
    const auto write_kernel = [&](const char* name, const KernelResult& kernel, bool last) {
      out << "        \"" << name << "\": {\"median_ms\": " << kernel.stats.median_ms
          << ", \"min_ms\": " << kernel.stats.min_ms << ", \"max_ms\": " << kernel.stats.max_ms
          << ", \"gb_per_s\": " << size.bytes / (kernel.stats.median_ms * 1e6)
          << ", \"sum\": " << kernel.sum << ", \"ok\": " << (kernel.ok ? "true" : "false") << "}"
          << (last ? "\n" : ",\n");
    };
    write_kernel("atomic", size.atomic, false);
    write_kernel("block", size.block, false);
    write_kernel("warp", size.warp, false);
    write_kernel("cub", size.cub, false);
    out << "        \"filter\": {\"median_ms\": " << size.filter.stats.median_ms
        << ", \"min_ms\": " << size.filter.stats.min_ms
        << ", \"max_ms\": " << size.filter.stats.max_ms
        << ", \"gb_per_s\": " << size.bytes / (size.filter.stats.median_ms * 1e6)
        << ", \"count\": " << size.filter.count << ", \"sum\": " << size.filter.sum
        << ", \"ok\": " << (size.filter.ok ? "true" : "false") << "}\n";
    out << "      },\n";
    out << "      \"transfer\": {\n";
    const auto write_transfer = [&](const char* name, const TransferResult& transfer, bool last) {
      out << "        \"" << name << "\": {\"ran\": " << (transfer.ran ? "true" : "false")
          << ", \"median_ms\": " << transfer.stats.median_ms
          << ", \"min_ms\": " << transfer.stats.min_ms << ", \"max_ms\": " << transfer.stats.max_ms
          << ", \"gb_per_s\": " << transfer.gbps << ", \"sum\": " << transfer.sum
          << ", \"ok\": " << (transfer.ok ? "true" : "false");
      if (!transfer.note.empty()) {
        out << ", \"note\": \"" << json_escape(transfer.note) << "\"";
      }
      out << "}" << (last ? "\n" : ",\n");
    };
    write_transfer("pageable", size.pageable, false);
    write_transfer("pinned", size.pinned, false);
    write_transfer("four_streams", size.streams, true);
    out << "      }\n";
    out << "    }" << (index + 1 == results.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

void write_markdown(const std::string& path, const DeviceInfo& device,
                    const std::vector<SizeResult>& results, int iterations,
                    int transfer_iterations) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("cannot open " + path);
  out << "# CUDA reduction microbenchmark\n\n";
  out << "Generated " << utc_timestamp() << " on **" << device.name << "** (compute "
      << device.major << "." << device.minor << ", " << device.sms << " SMs, "
      << human_bytes(static_cast<double>(device.memory_bytes))
      << "). Theoretical DRAM bandwidth: " << std::fixed << std::setprecision(1)
      << device.theoretical_gbps << " GB/s. " << iterations << " kernel iterations and "
      << transfer_iterations << " transfer iterations after warm-up; median values are shown.\n\n";
  out << "| Elements | Size | Atomic (ms) | Block (ms) | Warp (ms) | CUB (ms) | Filter (ms) | Warp "
         "GB/s | Filter GB/s | Result |\n";
  out << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
  for (const SizeResult& size : results) {
    const bool ok = size.atomic.ok && size.block.ok && size.warp.ok && size.filter.ok &&
                    (!size.cub.stats.median_ms || size.cub.ok);
    out << "| " << size.elements << " | " << human_bytes(size.bytes) << " | " << std::fixed
        << std::setprecision(2) << size.atomic.stats.median_ms << " | "
        << size.block.stats.median_ms << " | " << size.warp.stats.median_ms << " | ";
    if (size.cub.stats.median_ms > 0.0) {
      out << size.cub.stats.median_ms;
    } else {
      out << "skipped";
    }
    out << " | " << size.filter.stats.median_ms << " | " << std::setprecision(1)
        << size.bytes / (size.warp.stats.median_ms * 1e6) << " | "
        << size.bytes / (size.filter.stats.median_ms * 1e6) << " | " << (ok ? "verified" : "FAILED")
        << " |\n";
  }
  out << "\nThe `Filter` column runs the same vectorized pass with a `value > 0` "
         "predicate and accumulates `COUNT` plus `SUM` for matching values. It shows "
         "whether predicate evaluation costs anything beyond the memory traffic.\n";
  out << "\n## End-to-end transfer\n\n";
  out << "The transfer rows move the input to the GPU, run the warp-shuffle reduction, "
         "and copy one double back. `GB/s` counts both directions.\n\n";
  out << "| Elements | Pageable (ms) | Pinned (ms) | Four streams (ms) | Pageable GB/s | Pinned "
         "GB/s | Streams GB/s |\n";
  out << "|---:|---:|---:|---:|---:|---:|---:|\n";
  for (const SizeResult& size : results) {
    const auto cell = [](const TransferResult& transfer) {
      if (!transfer.ran)
        return std::string("skipped");
      std::ostringstream value;
      value << std::fixed << std::setprecision(2) << transfer.stats.median_ms;
      return value.str();
    };
    const auto bandwidth = [](const TransferResult& transfer) {
      if (!transfer.ran)
        return std::string("-");
      std::ostringstream value;
      value << std::fixed << std::setprecision(1) << transfer.gbps;
      return value.str();
    };
    out << "| " << size.elements << " | " << cell(size.pageable) << " | " << cell(size.pinned)
        << " | " << cell(size.streams) << " | " << bandwidth(size.pageable) << " | "
        << bandwidth(size.pinned) << " | " << bandwidth(size.streams) << " |\n";
  }
  out << "\nAll reductions and the filter aggregate were checked against a "
         "long-double host reference. The pageable path includes the driver's "
         "staging copy; the pinned and four-stream paths use `cudaMemcpyAsync`.\n";
}

} // namespace

int main(int argc, char** argv) {
  std::vector<size_t> sizes = {1u << 24, 1u << 26, 1u << 27}; // 16M, 64M, 128M doubles
  int iterations = 20;
  int transfer_iterations = 5;
  bool do_transfer = true;
  bool do_cub = true;
  std::string json_path;
  std::string markdown_path;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    const auto next = [&](const char* name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (arg == "--sizes") {
      sizes.clear();
      const std::string list = next("--sizes");
      std::stringstream stream(list);
      std::string item;
      while (std::getline(stream, item, ',')) {
        if (!item.empty())
          sizes.push_back(static_cast<size_t>(std::stoull(item)));
      }
      if (sizes.empty())
        throw std::runtime_error("--sizes produced no values");
    } else if (arg == "--iters") {
      iterations = std::stoi(next("--iters"));
    } else if (arg == "--transfer-iters") {
      transfer_iterations = std::stoi(next("--transfer-iters"));
    } else if (arg == "--no-transfer") {
      do_transfer = false;
    } else if (arg == "--no-cub") {
      do_cub = false;
    } else if (arg == "--json") {
      json_path = next("--json");
    } else if (arg == "--markdown") {
      markdown_path = next("--markdown");
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: reduce_bench [--sizes N,N,...] [--iters N] "
                   "[--transfer-iters N] [--no-transfer] [--no-cub] "
                   "[--json PATH] [--markdown PATH]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (iterations < 1 || transfer_iterations < 1) {
    throw std::runtime_error("iteration counts must be positive");
  }

  const DeviceInfo device = query_device();
  std::cout << "device: " << device.name << " (sm_" << device.major << device.minor << ", "
            << device.sms << " SMs, " << human_bytes(static_cast<double>(device.memory_bytes))
            << ")\n";
  std::cout << "theoretical DRAM bandwidth: " << std::fixed << std::setprecision(1)
            << device.theoretical_gbps << " GB/s\n";
  std::cout << std::left << std::setw(12) << "elements" << std::right << std::setw(12) << "size"
            << std::setw(12) << "atomic" << std::setw(12) << "block" << std::setw(12) << "warp"
            << std::setw(12) << "cub" << std::setw(12) << "filter" << std::setw(14) << "warp GB/s"
            << std::setw(12) << "filter GB/s" << "  result\n";

  std::vector<SizeResult> results;
  results.reserve(sizes.size());
  for (const size_t elements : sizes) {
    std::cerr << "running " << elements << " doubles...\n";
    SizeResult result =
        run_size(device, elements, iterations, transfer_iterations, do_transfer, do_cub);
    results.push_back(result);

    const bool ok = result.atomic.ok && result.block.ok && result.warp.ok && result.filter.ok &&
                    (!do_cub || result.cub.ok);
    std::cout << std::left << std::setw(12) << result.elements << std::right << std::setw(12)
              << human_bytes(result.bytes) << std::setw(12) << std::fixed << std::setprecision(2)
              << result.atomic.stats.median_ms << std::setw(12) << result.block.stats.median_ms
              << std::setw(12) << result.warp.stats.median_ms << std::setw(12);
    if (result.cub.stats.median_ms > 0.0) {
      std::cout << result.cub.stats.median_ms;
    } else {
      std::cout << "-";
    }
    std::cout << std::setw(12) << result.filter.stats.median_ms << std::setw(14)
              << std::setprecision(1) << result.bytes / (result.warp.stats.median_ms * 1e6)
              << std::setw(12) << result.bytes / (result.filter.stats.median_ms * 1e6) << "  "
              << (ok ? "verified" : "FAILED") << "\n";

    if (do_transfer) {
      const auto print_transfer = [](const char* label, const TransferResult& transfer) {
        std::cout << "  " << std::left << std::setw(18) << label << std::right;
        if (!transfer.ran) {
          std::cout << "skipped";
          if (!transfer.note.empty())
            std::cout << " (" << transfer.note << ")";
          std::cout << "\n";
          return;
        }
        std::cout << std::fixed << std::setprecision(2) << std::setw(9) << transfer.stats.median_ms
                  << " ms  " << std::setprecision(1) << std::setw(7) << transfer.gbps << " GB/s  "
                  << (transfer.ok ? "verified" : "FAILED") << "\n";
      };
      print_transfer("pageable", result.pageable);
      print_transfer("pinned / 1 stream", result.pinned);
      print_transfer("pinned / 4 streams", result.streams);
    }
  }

  if (!json_path.empty()) {
    std::filesystem::path path(json_path);
    if (path.has_parent_path())
      std::filesystem::create_directories(path.parent_path());
    write_json(json_path, device, results, iterations, transfer_iterations);
    std::cout << "wrote " << json_path << "\n";
  }
  if (!markdown_path.empty()) {
    std::filesystem::path path(markdown_path);
    if (path.has_parent_path())
      std::filesystem::create_directories(path.parent_path());
    write_markdown(markdown_path, device, results, iterations, transfer_iterations);
    std::cout << "wrote " << markdown_path << "\n";
  }
  return 0;
}
