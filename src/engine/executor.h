#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
  // Wrap every physical operator in a timing decorator. Disabled for normal
  // execution so the hot path stays allocation- and clock-free.
  bool profile = false;
};

// One node of the actual execution tree produced by EXPLAIN ANALYZE.
// `inclusive_ms` covers the operator and all of its children; `self_ms` in the
// printed output subtracts direct child time. `output_rows` counts rows emitted
// by the operator, so a scan reports rows after pushdown and an aggregate
// reports groups, not input rows.
struct OperatorProfile {
  std::string name;
  std::string detail;
  double inclusive_ms = 0.0;
  int64_t output_rows = 0;
  int64_t output_batches = 0;
  std::vector<std::shared_ptr<OperatorProfile>> children;
};

struct ExecutionResult {
  std::shared_ptr<arrow::Table> table;
  ExecutionStats stats;
  double execute_ms = 0.0;
  std::shared_ptr<OperatorProfile> profile;
};

arrow::Result<ExecutionResult> ExecutePlan(const std::shared_ptr<LogicalPlan>& plan,
                                           const ExecutionOptions& options);

arrow::Result<std::shared_ptr<Operator>>
BuildOperator(const std::shared_ptr<LogicalPlan>& plan, const ExecutionOptions& options,
              const std::shared_ptr<ExecutionStats>& stats);

// Writes an EXPLAIN ANALYZE tree. The total is supplied separately because the
// profile covers operator time while execute_ms also includes setup teardown.
void PrintProfile(const OperatorProfile& profile, double execute_ms, std::ostream& out);

} // namespace qe
