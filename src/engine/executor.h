#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <memory>

#include "engine/operators.h"
#include "engine/plan.h"

namespace qe {

struct ExecutionOptions {
  int64_t batch_size = 8192;
  // The CPU path is the default. --gpu opts into the optional CUDA aggregate
  // operators; unsupported queries still fall back to the CPU implementation.
  bool use_gpu = false;
  // Parallel row-group decoding for scans. 1 keeps the single-threaded scan;
  // N > 1 decodes row groups concurrently while emitting batches in exactly
  // the serial order, so results do not depend on the thread count.
  int threads = 1;
};

struct ExecutionResult {
  std::shared_ptr<arrow::Table> table;
  ExecutionStats stats;
  double execute_ms = 0.0;
};

arrow::Result<ExecutionResult> ExecutePlan(const std::shared_ptr<LogicalPlan>& plan,
                                           const ExecutionOptions& options);

arrow::Result<std::unique_ptr<Operator>>
BuildOperator(const std::shared_ptr<LogicalPlan>& plan, const ExecutionOptions& options,
              const std::shared_ptr<ExecutionStats>& stats);

} // namespace qe
