#include "engine/executor.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace qe {
namespace {

using Clock = std::chrono::steady_clock;

// A decorator that measures the wall time of each Next() call. Time spent in
// children is included here and subtracted when self time is printed, which is
// why the parent must receive the same profile node tree that its children
// populate.
class ProfilingOperator final : public Operator {
public:
  ProfilingOperator(std::shared_ptr<Operator> child, std::shared_ptr<OperatorProfile> profile)
      : child_(std::move(child)), profile_(std::move(profile)) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return child_->schema();
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override {
    const auto start = Clock::now();
    auto result = child_->Next();
    const auto end = Clock::now();
    profile_->inclusive_ms += std::chrono::duration<double, std::milli>(end - start).count();
    if (result.ok() && *result) {
      ++profile_->output_batches;
      profile_->output_rows += (*result)->num_rows();
    }
    return result;
  }

private:
  std::shared_ptr<Operator> child_;
  std::shared_ptr<OperatorProfile> profile_;
};

std::string ProfileName(const LogicalPlan& plan) {
  switch (plan.kind) {
  case LogicalPlan::Kind::Scan:
    return "Scan " + plan.table_alias;
  case LogicalPlan::Kind::Filter:
    return "Filter";
  case LogicalPlan::Kind::Join:
    return plan.left_outer ? "LeftHashJoin" : "HashJoin";
  case LogicalPlan::Kind::Aggregate:
    return plan.group_by.empty() ? "Aggregate" : "HashAggregate";
  case LogicalPlan::Kind::Project:
    return "Project";
  case LogicalPlan::Kind::Distinct:
    return "Distinct";
  case LogicalPlan::Kind::Sort:
    return "Sort";
  case LogicalPlan::Kind::Limit:
    return "Limit";
  }
  return "Operator";
}

std::string ProfileDetail(const LogicalPlan& plan) {
  std::ostringstream out;
  switch (plan.kind) {
  case LogicalPlan::Kind::Scan: {
    out << "columns=[";
    for (size_t i = 0; i < plan.scan_columns.size(); ++i) {
      if (i)
        out << ", ";
      out << plan.scan_columns[i];
    }
    out << "]";
    if (!plan.pushed_filters.empty()) {
      out << " pushdown=[";
      for (size_t i = 0; i < plan.pushed_filters.size(); ++i) {
        if (i)
          out << ", ";
        out << sql::ExprToString(plan.pushed_filters[i]);
      }
      out << "]";
    }
    break;
  }
  case LogicalPlan::Kind::Filter:
    out << sql::ExprToString(plan.predicate);
    break;
  case LogicalPlan::Kind::Join:
    out << "on [";
    for (size_t i = 0; i < plan.join_conditions.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(plan.join_conditions[i].left) << " = "
          << sql::ExprToString(plan.join_conditions[i].right);
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Aggregate:
    out << "group=[";
    for (size_t i = 0; i < plan.group_by.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(plan.group_by[i]);
    }
    out << "] aggregates=[";
    for (size_t i = 0; i < plan.aggregates.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(plan.aggregates[i]);
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Project:
    out << "[";
    for (size_t i = 0; i < plan.output_names.size(); ++i) {
      if (i)
        out << ", ";
      out << plan.output_names[i];
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Distinct:
    break;
  case LogicalPlan::Kind::Sort:
    out << "[";
    for (size_t i = 0; i < plan.sort_keys.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(plan.sort_keys[i].expression)
          << (plan.sort_keys[i].descending ? " DESC" : " ASC");
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Limit:
    out << plan.limit;
    if (plan.offset > 0)
      out << " offset=" << plan.offset;
    break;
  }
  return out.str();
}

struct BuiltOperator {
  std::shared_ptr<Operator> op;
  std::shared_ptr<OperatorProfile> profile;
};

arrow::Result<BuiltOperator>
BuildOperatorNode(const std::shared_ptr<LogicalPlan>& plan, const ExecutionOptions& options,
                  const std::shared_ptr<ExecutionStats>& stats,
                  const std::shared_ptr<OperatorProfile>& parent_profile) {
  if (!plan)
    return arrow::Status::Invalid("cannot build an empty plan node");

  std::shared_ptr<OperatorProfile> profile;
  if (options.profile) {
    profile = std::make_shared<OperatorProfile>();
    profile->name = ProfileName(*plan);
    profile->detail = ProfileDetail(*plan);
    if (parent_profile)
      parent_profile->children.push_back(profile);
  }

  auto build_child = [&](size_t index) -> arrow::Result<BuiltOperator> {
    return BuildOperatorNode(plan->children[index], options, stats, profile);
  };

  std::shared_ptr<Operator> op;
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
    op = std::make_shared<ScanOperator>(std::move(source), plan->scan_columns,
                                        std::move(output_names), plan->pushed_filters,
                                        options.batch_size, stats, options.threads);
    break;
  }
  case LogicalPlan::Kind::Filter: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<FilterOperator>(std::move(child.op), plan->predicate);
    break;
  }
  case LogicalPlan::Kind::Join: {
    ARROW_ASSIGN_OR_RAISE(auto left, build_child(0));
    ARROW_ASSIGN_OR_RAISE(auto right, build_child(1));
    const bool build_left = plan->children[0]->estimated_rows < plan->children[1]->estimated_rows;
    op = std::make_shared<JoinOperator>(std::move(left.op), std::move(right.op),
                                        plan->join_conditions, build_left, plan->left_outer);
    break;
  }
  case LogicalPlan::Kind::Aggregate: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<AggregateOperator>(std::move(child.op), plan->group_by, plan->aggregates,
                                             plan->schema, stats, options.use_gpu);
    break;
  }
  case LogicalPlan::Kind::Project: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<ProjectOperator>(std::move(child.op), plan->projections,
                                           plan->output_names, plan->schema);
    break;
  }
  case LogicalPlan::Kind::Distinct: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<DistinctOperator>(std::move(child.op));
    break;
  }
  case LogicalPlan::Kind::Sort: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<SortOperator>(std::move(child.op), plan->sort_keys, plan->limit_hint);
    break;
  }
  case LogicalPlan::Kind::Limit: {
    ARROW_ASSIGN_OR_RAISE(auto child, build_child(0));
    op = std::make_shared<LimitOperator>(std::move(child.op), plan->limit, plan->offset);
    break;
  }
  }

  if (!op)
    return arrow::Status::Invalid("unknown plan node");
  if (profile)
    op = std::make_shared<ProfilingOperator>(std::move(op), profile);
  return BuiltOperator{std::move(op), std::move(profile)};
}

double SelfMs(const OperatorProfile& profile) {
  double child_ms = 0.0;
  for (const auto& child : profile.children)
    child_ms += child->inclusive_ms;
  return std::max(0.0, profile.inclusive_ms - child_ms);
}

void PrintProfileNode(const OperatorProfile& profile, int depth, std::ostream& out) {
  out << std::string(static_cast<size_t>(depth) * 2, ' ') << profile.name;
  if (!profile.detail.empty())
    out << ' ' << profile.detail;
  out << " (actual time=" << profile.inclusive_ms << " ms, self=" << SelfMs(profile)
      << " ms, rows=" << profile.output_rows << ", batches=" << profile.output_batches << ")\n";
  for (const auto& child : profile.children)
    PrintProfileNode(*child, depth + 1, out);
}

} // namespace

arrow::Result<std::shared_ptr<Operator>>
BuildOperator(const std::shared_ptr<LogicalPlan>& plan, const ExecutionOptions& options,
              const std::shared_ptr<ExecutionStats>& stats) {
  // This entry point does not return the profile tree, so avoid paying for
  // decorators when the caller only wants the executable pipeline.
  ExecutionOptions unprofiled = options;
  unprofiled.profile = false;
  ARROW_ASSIGN_OR_RAISE(auto built, BuildOperatorNode(plan, unprofiled, stats, nullptr));
  return std::move(built.op);
}

arrow::Result<ExecutionResult> ExecutePlan(const std::shared_ptr<LogicalPlan>& plan,
                                           const ExecutionOptions& options) {
  if (!plan)
    return arrow::Status::Invalid("cannot execute an empty plan");
  if (options.batch_size <= 0) {
    return arrow::Status::Invalid("batch size must be positive");
  }

  auto stats = std::make_shared<ExecutionStats>();
  ExecutionResult result;
  auto start = std::chrono::steady_clock::now();

  ARROW_ASSIGN_OR_RAISE(auto built, BuildOperatorNode(plan, options, stats, nullptr));
  auto root = std::move(built.op);

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, root->Next());
    if (!batch)
      break;
    if (batch->num_rows() > 0)
      batches.push_back(std::move(batch));
  }

  if (batches.empty())
    batches.push_back(qe::MakeEmptyBatch(plan->schema));
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(plan->schema, batches));

  auto end = std::chrono::steady_clock::now();
  result.table = std::move(table);
  result.stats = std::move(*stats);
  result.execute_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.profile = std::move(built.profile);
  return result;
}

void PrintProfile(const OperatorProfile& profile, double execute_ms, std::ostream& out) {
  const std::ios::fmtflags flags = out.flags();
  const std::streamsize precision = out.precision();
  out << std::fixed << std::setprecision(3);
  out << "EXPLAIN ANALYZE\n";
  PrintProfileNode(profile, 0, out);
  out << "Execution time: " << execute_ms << " ms\n";
  out.flags(flags);
  out.precision(precision);
}

} // namespace qe
