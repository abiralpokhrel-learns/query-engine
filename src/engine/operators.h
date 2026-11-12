#pragma once

#include <arrow/api.h>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine/plan.h"
#include "gpu/cuda_runtime.h"
#include "sql/sql.h"
#include "storage/parquet.h"

namespace qe {

struct ExecutionStats {
  int64_t row_groups_total = 0;
  int64_t row_groups_read = 0;
  int64_t row_groups_skipped = 0;
  int64_t rows_scanned = 0;
  int64_t rows_after_pushdown = 0;
  int64_t batches_scanned = 0;
  // Rows counted from Parquet metadata instead of being decoded. Used by
  // COUNT(*) queries with no predicate.
  int64_t rows_from_metadata = 0;
  // Set only when a query actually ran a GPU kernel, not merely because the
  // binary was compiled with CUDA support.
  bool gpu_used = false;
  bool gpu_attempted = false;
  int64_t gpu_batches = 0;
  double gpu_transfer_ms = 0.0;
  double gpu_kernel_ms = 0.0;
};

class Operator {
public:
  virtual ~Operator() = default;
  virtual const std::shared_ptr<arrow::Schema>& schema() const = 0;
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() = 0;
};

class ScanOperator final : public Operator {
public:
  ScanOperator(std::shared_ptr<ParquetSource> source, std::vector<std::string> file_columns,
               std::vector<std::string> output_names, std::vector<sql::ExprPtr> filters,
               int64_t batch_size, std::shared_ptr<ExecutionStats> stats, int threads = 1);
  ~ScanOperator() override;

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return schema_;
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

  int row_groups_read() const {
    return static_cast<int>(row_groups_.size());
  }

private:
  arrow::Status Initialize();
  // Parallel path: row groups are decoded by worker threads with their own
  // Parquet readers, then emitted in the same order as the serial path.
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> NextParallel();
  arrow::Status StartWorkers();

  std::shared_ptr<ParquetSource> source_;
  std::vector<std::string> file_columns_;
  std::vector<std::string> output_names_;
  std::vector<sql::ExprPtr> filters_;
  int64_t batch_size_;
  int threads_ = 1;
  std::shared_ptr<ExecutionStats> stats_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<int> column_indices_;
  std::vector<int> row_groups_;
  std::vector<ScanFilter> scan_filters_;
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  struct Prefetch;
  std::unique_ptr<Prefetch> prefetch_;
  arrow::Status init_status_ = arrow::Status::OK();
  bool done_ = false;
  bool initialized_ = false;
  bool metadata_only_ = false;
  int64_t metadata_rows_ = 0;
};

class FilterOperator final : public Operator {
public:
  FilterOperator(std::shared_ptr<Operator> child, sql::ExprPtr predicate)
      : child_(std::move(child)), predicate_(std::move(predicate)) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return child_->schema();
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  std::shared_ptr<Operator> child_;
  sql::ExprPtr predicate_;
};

class ProjectOperator final : public Operator {
public:
  ProjectOperator(std::shared_ptr<Operator> child, std::vector<sql::ExprPtr> projections,
                  std::vector<std::string> output_names,
                  std::shared_ptr<arrow::Schema> output_schema)
      : child_(std::move(child)), projections_(std::move(projections)),
        output_names_(std::move(output_names)), output_schema_(std::move(output_schema)) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return output_schema_;
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  std::shared_ptr<Operator> child_;
  std::vector<sql::ExprPtr> projections_;
  std::vector<std::string> output_names_;
  std::shared_ptr<arrow::Schema> output_schema_;
};

class LimitOperator final : public Operator {
public:
  LimitOperator(std::shared_ptr<Operator> child, int64_t limit, int64_t offset = 0)
      : child_(std::move(child)), remaining_(limit), offset_remaining_(offset) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return child_->schema();
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  std::shared_ptr<Operator> child_;
  int64_t remaining_ = 0;
  int64_t offset_remaining_ = 0;
};

class DistinctOperator final : public Operator {
public:
  explicit DistinctOperator(std::shared_ptr<Operator> child) : child_(std::move(child)) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return child_->schema();
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  std::shared_ptr<Operator> child_;
  std::shared_ptr<arrow::RecordBatch> result_;
  bool done_ = false;
};

class SortOperator final : public Operator {
public:
  SortOperator(std::shared_ptr<Operator> child, std::vector<SortKey> sort_keys,
               int64_t limit_hint = -1)
      : child_(std::move(child)), sort_keys_(std::move(sort_keys)), limit_hint_(limit_hint) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return child_->schema();
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Materialize();

  std::shared_ptr<Operator> child_;
  std::vector<SortKey> sort_keys_;
  int64_t limit_hint_ = -1;
  std::shared_ptr<arrow::RecordBatch> result_;
  bool done_ = false;
};

class AggregateOperator final : public Operator {
public:
  AggregateOperator(std::shared_ptr<Operator> child, std::vector<sql::ExprPtr> group_by,
                    std::vector<sql::ExprPtr> aggregates,
                    std::shared_ptr<arrow::Schema> output_schema,
                    std::shared_ptr<ExecutionStats> stats, bool use_gpu = false)
      : child_(std::move(child)), group_by_(std::move(group_by)),
        aggregates_(std::move(aggregates)), output_schema_(std::move(output_schema)),
        stats_(std::move(stats)), use_gpu_(use_gpu) {}

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return output_schema_;
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  enum class ExtremeKind : uint8_t { kNone, kInt64, kDouble, kBool, kString };

  struct AggState {
    int64_t count = 0;
    __extension__ __int128 int_sum = 0;
    double double_sum = 0;
    bool has_value = false;
    // MIN/MAX keep the winning value in its native representation. The old
    // path allocated an Arrow scalar for every row that became the new
    // extreme; this keeps the accumulation loop allocation-free.
    ExtremeKind extreme_kind = ExtremeKind::kNone;
    int64_t int_extreme = 0;
    double double_extreme = 0;
    bool bool_extreme = false;
    // Allocated only for string MIN/MAX. Keeping a std::string inline would
    // make every count/SUM/AVG state much larger and slow the grouped loop.
    std::unique_ptr<std::string> string_extreme;
  };
  struct Group {
    std::vector<std::shared_ptr<arrow::Scalar>> keys;
  };

  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view value) const noexcept {
      return std::hash<std::string_view>{}(value);
    }
  };

  // Open-addressed index for a single integer GROUP BY key. The generic
  // string-keyed map remains the fallback for strings, multiple keys, and
  // non-integer expressions.
  struct FlatSlot {
    uint64_t key = 0;
    size_t group = 0;
    uint8_t used = 0;
    uint8_t is_null = 0;
  };

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> BuildResult();
  void AddGroup(Group group);
  AggState& State(size_t aggregate, size_t group) {
    return states_[group * aggregates_.size() + aggregate];
  }
  const AggState& State(size_t aggregate, size_t group) const {
    return states_[group * aggregates_.size() + aggregate];
  }
  arrow::Status Consume(const std::shared_ptr<arrow::RecordBatch>& batch);
  arrow::Status ConsumeGpu(const std::shared_ptr<arrow::RecordBatch>& batch,
                           const std::vector<std::shared_ptr<arrow::Array>>& argument_arrays);
  arrow::Status FlushGpuBuffer(size_t index);
  arrow::Status FinishGpu();
  arrow::Status BuildGroupIds(int64_t num_rows,
                              const std::vector<std::shared_ptr<arrow::Array>>& group_arrays,
                              std::vector<size_t>& group_ids);
  arrow::Status AccumulateBatch(const std::vector<std::shared_ptr<arrow::Array>>& argument_arrays,
                                const std::vector<size_t>& group_ids);
  size_t FlatFindOrInsert(uint64_t key, bool is_null, bool* inserted);
  void FlatGrow();
  bool ExpandDenseKey(int64_t key);
  void DisableDense();

  struct GpuBuffer {
    std::vector<double> values;
    double sum = 0.0;
    int64_t count = 0;
  };

  std::shared_ptr<Operator> child_;
  std::vector<sql::ExprPtr> group_by_;
  std::vector<sql::ExprPtr> aggregates_;
  std::shared_ptr<arrow::Schema> output_schema_;
  std::shared_ptr<ExecutionStats> stats_;
  bool use_gpu_ = false;
  int gpu_mode_ = 0; // 0 undecided, 1 active, -1 CPU fallback
  std::unique_ptr<gpu::Workspace> gpu_workspace_;
  std::vector<GpuBuffer> gpu_buffers_;
  std::vector<Group> groups_;
  // States for all aggregates of a group are adjacent, so the fused
  // count/SUM/AVG loop touches one group entry per row instead of one cache
  // line per aggregate pass.
  std::vector<AggState> states_;
  std::unordered_map<std::string, size_t, StringHash, std::equal_to<>> group_index_;
  std::vector<FlatSlot> flat_slots_;
  // For a single integer key with a compact range, a direct array is faster
  // than hashing on every row. It is only enabled when the first batch shows
  // a small enough span, and falls back to the open-addressed table if a later
  // batch widens the range too much.
  std::vector<int32_t> dense_group_ids_;
  int64_t dense_min_key_ = 0;
  int64_t dense_max_key_ = 0;
  int32_t dense_null_group_ = -1;
  bool dense_active_ = false;
  // Low-cardinality single-string grouping stays in a small linear list and
  // avoids constructing a key string and hashing it for every row. It migrates
  // to the generic hash map once the cardinality grows.
  struct SmallStringGroup {
    std::string key;
    size_t group = 0;
  };
  std::vector<SmallStringGroup> small_string_groups_;
  int32_t small_null_group_ = -1;
  bool string_map_active_ = false;
  std::vector<size_t> batch_group_ids_;
  size_t flat_size_ = 0;
  // 0 undecided, 1 compact integer, 2 single string, -1 generic serialized map
  int group_key_mode_ = 0;
  std::shared_ptr<arrow::RecordBatch> result_;
  bool done_ = false;
};

class JoinOperator final : public Operator {
public:
  JoinOperator(std::shared_ptr<Operator> left, std::shared_ptr<Operator> right,
               std::vector<JoinCondition> conditions, bool build_left, bool left_outer = false);

  const std::shared_ptr<arrow::Schema>& schema() const override {
    return output_schema_;
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() override;

private:
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Build();
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> MaterializeLeftOuter();
  arrow::Result<std::shared_ptr<arrow::RecordBatch>>
  Probe(const std::shared_ptr<arrow::RecordBatch>& probe_batch,
        const std::vector<std::shared_ptr<arrow::Array>>& probe_keys);

  // Single-key joins with compact integer or string keys use the tables below
  // instead of the serialized-key map. The build side is materialized once, so
  // its chains can be built in ascending row order in one pass; there is no
  // aggregate-style cross-batch migration to support.
  enum class JoinKeyMode : uint8_t { kGeneric, kInteger, kString };
  struct IntegerSlot {
    uint64_t key = 0;
    int64_t head = -1;
    int64_t last = -1;
    bool used = false;
  };
  struct StringSlot {
    std::string_view value;
    size_t hash = 0;
    int64_t head = -1;
    int64_t last = -1;
    bool used = false;
  };

  arrow::Status BuildKeyTable();
  // Fills the index builders in the same order as the serialized-key path:
  // probe rows in batch order, each with its matches in ascending build-row
  // order. When preserve_unmatched is set, every probe row without a match
  // appends a single (row, unmatched_build_row) pair.
  arrow::Status MatchTyped(const std::shared_ptr<arrow::Array>& probe_key, bool preserve_unmatched,
                           int64_t unmatched_build_row, arrow::Int64Builder& probe_rows,
                           arrow::Int64Builder& build_rows);

  std::shared_ptr<Operator> left_;
  std::shared_ptr<Operator> right_;
  std::vector<JoinCondition> conditions_;
  bool build_left_ = false;
  bool left_outer_ = false;
  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  std::shared_ptr<arrow::Schema> output_schema_;
  std::shared_ptr<arrow::RecordBatch> build_batch_;
  std::vector<std::shared_ptr<arrow::Array>> build_keys_;
  std::unordered_map<std::string, std::vector<int64_t>> hash_;
  JoinKeyMode key_mode_ = JoinKeyMode::kGeneric;
  std::vector<IntegerSlot> integer_slots_;
  std::vector<StringSlot> string_slots_;
  // For each build row, the next build row with the same key; -1 ends a chain.
  std::vector<int64_t> next_row_;
  bool built_ = false;
  bool done_ = false;
};

std::shared_ptr<arrow::RecordBatch> MakeEmptyBatch(const std::shared_ptr<arrow::Schema>& schema);

} // namespace qe
