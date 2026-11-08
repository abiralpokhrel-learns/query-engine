#include <iostream>

#include "gpu/cuda_runtime.h"

int main() {
  qe::gpu::DeviceInfo device = qe::gpu::QueryDevice();
  if (!device.build_enabled) {
    std::cerr << "CUDA support was not compiled into this build\n";
    return 1;
  }
  if (!device.available) {
    std::cerr << "no usable CUDA device: " << device.error << "\n";
    return 77;
  }

  std::cout << "device: " << device.name << "\n";
  std::cout << "compute: " << device.major << "." << device.minor << "\n";
  std::cout << "memory: " << (device.memory_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";

  qe::gpu::SelfTest test = qe::gpu::RunSelfTest();
  std::cout << test.message << "\n";
  return test.ok ? 0 : 1;
}
