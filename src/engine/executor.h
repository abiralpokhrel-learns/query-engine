#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <memory>

#include "engine/operators.h"
#include "engine/plan.h"

namespace qe {

struct ExecutionOptions {
  int64_t batch_size = 8192;
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
