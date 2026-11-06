#include "engine/executor.h"

#include <chrono>
#include <vector>

namespace qe {

arrow::Result<std::unique_ptr<Operator>> BuildOperator(
    const std::shared_ptr<LogicalPlan>& plan, const ExecutionOptions& options,
    const std::shared_ptr<ExecutionStats>& stats) {
  switch (plan->kind) {
    case LogicalPlan::Kind::Scan: {
      std::shared_ptr<ParquetSource> source = plan->table_source;
      if (!source) {
        ARROW_ASSIGN_OR_RAISE(source, ParquetSource::Open(plan->table_path));
      }
      std::vector<std::string> output_names;
      output_names.reserve(plan->scan_columns.size());
      for (const auto& column : plan->scan_columns) {
        output_names.push_back(plan->table_alias + "." + column);
      }
      return std::make_unique<ScanOperator>(std::move(source), plan->scan_columns,
                                            std::move(output_names), plan->pushed_filters,
                                            options.batch_size, stats);
    }
    case LogicalPlan::Kind::Filter: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<FilterOperator>(std::move(child), plan->predicate);
    }
    case LogicalPlan::Kind::Join: {
      ARROW_ASSIGN_OR_RAISE(auto left, BuildOperator(plan->children[0], options, stats));
      ARROW_ASSIGN_OR_RAISE(auto right, BuildOperator(plan->children[1], options, stats));
      const bool build_left = plan->children[0]->estimated_rows < plan->children[1]->estimated_rows;
      return std::make_unique<JoinOperator>(std::move(left), std::move(right),
                                            plan->join_conditions, build_left,
                                            plan->left_outer);
    }
    case LogicalPlan::Kind::Aggregate: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<AggregateOperator>(std::move(child), plan->group_by,
                                                 plan->aggregates, plan->schema);
    }
    case LogicalPlan::Kind::Project: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<ProjectOperator>(std::move(child), plan->projections,
                                               plan->output_names, plan->schema);
    }
    case LogicalPlan::Kind::Distinct: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<DistinctOperator>(std::move(child));
    }
    case LogicalPlan::Kind::Sort: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<SortOperator>(std::move(child), plan->sort_keys,
                                            plan->limit_hint);
    }
    case LogicalPlan::Kind::Limit: {
      ARROW_ASSIGN_OR_RAISE(auto child, BuildOperator(plan->children[0], options, stats));
      return std::make_unique<LimitOperator>(std::move(child), plan->limit, plan->offset);
    }
  }
  return arrow::Status::Invalid("unknown plan node");
}

arrow::Result<ExecutionResult> ExecutePlan(const std::shared_ptr<LogicalPlan>& plan,
                                           const ExecutionOptions& options) {
  if (!plan) return arrow::Status::Invalid("cannot execute an empty plan");
  if (options.batch_size <= 0) {
    return arrow::Status::Invalid("batch size must be positive");
  }

  auto stats = std::make_shared<ExecutionStats>();
  ExecutionResult result;
  auto start = std::chrono::steady_clock::now();

  ARROW_ASSIGN_OR_RAISE(auto root, BuildOperator(plan, options, stats));

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, root->Next());
    if (!batch) break;
    if (batch->num_rows() > 0) batches.push_back(std::move(batch));
  }

  if (batches.empty()) batches.push_back(qe::MakeEmptyBatch(plan->schema));
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(plan->schema, batches));

  auto end = std::chrono::steady_clock::now();
  result.table = std::move(table);
  result.stats = std::move(*stats);
  result.execute_ms = std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

}  // namespace qe
