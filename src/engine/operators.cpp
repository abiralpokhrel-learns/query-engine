#include "engine/operators.h"

#include <arrow/array/concatenate.h>
#include <arrow/array/util.h>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/checked_cast.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "engine/expression.h"
#include "engine/key.h"
#include "gpu/cuda_runtime.h"

namespace qe {
namespace {

using arrow::internal::checked_cast;

// The GPU aggregate buffers values on the host in bounded chunks. A flush
// performs one host-to-device copy, one reduction kernel, and one result copy,
// so per-batch synchronization does not dominate small scans.
constexpr size_t kGpuBufferValues = 1U << 20;

// Upper bound on the direct-index grouping table. At 4 bytes per slot this is
// 16 MiB, and it is only allocated when the observed key span is small.
constexpr uint64_t kMaxDenseRange = 1U << 22;
constexpr size_t kSmallStringGroups = 32;

std::shared_ptr<arrow::RecordBatch>
MakeEmptyBatchImpl(const std::shared_ptr<arrow::Schema>& schema) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto array = arrow::MakeArrayOfNull(schema->field(i)->type(), 0);
    columns.push_back(array.ValueOrDie());
  }
  return arrow::RecordBatch::Make(schema, 0, std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MaterializeOperator(Operator& op) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, op.Next());
    if (!batch)
      break;
    if (batch->num_rows() > 0)
      batches.push_back(batch);
  }
  if (batches.empty())
    return MakeEmptyBatchImpl(op.schema());
  if (batches.size() == 1)
    return batches[0];
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(op.schema(), batches));
  return table->CombineChunksToBatch();
}

void AppendScanFilters(const sql::ExprPtr& expr, const ParquetSource& source,
                       std::vector<ScanFilter>& out) {
  if (!expr)
    return;
  if (expr->kind == sql::ExprKind::Binary && expr->binary_op == sql::BinaryOp::And) {
    AppendScanFilters(expr->left, source, out);
    AppendScanFilters(expr->right, source, out);
    return;
  }
  if (expr->kind != sql::ExprKind::Binary)
    return;

  sql::BinaryOp op = expr->binary_op;
  switch (op) {
  case sql::BinaryOp::Eq:
  case sql::BinaryOp::NotEq:
  case sql::BinaryOp::Lt:
  case sql::BinaryOp::LtEq:
  case sql::BinaryOp::Gt:
  case sql::BinaryOp::GtEq:
    break;
  default:
    return;
  }

  sql::ExprPtr column;
  sql::ExprPtr literal;
  if (expr->left->kind == sql::ExprKind::Column && expr->right->kind == sql::ExprKind::Literal) {
    column = expr->left;
    literal = expr->right;
  } else if (expr->right->kind == sql::ExprKind::Column &&
             expr->left->kind == sql::ExprKind::Literal) {
    column = expr->right;
    literal = expr->left;
    switch (op) {
    case sql::BinaryOp::Lt:
      op = sql::BinaryOp::Gt;
      break;
    case sql::BinaryOp::LtEq:
      op = sql::BinaryOp::GtEq;
      break;
    case sql::BinaryOp::Gt:
      op = sql::BinaryOp::Lt;
      break;
    case sql::BinaryOp::GtEq:
      op = sql::BinaryOp::LtEq;
      break;
    default:
      break;
    }
  } else {
    return;
  }

  if (literal->literal.type == sql::LiteralType::Null)
    return;
  auto index = source.FieldIndex(column->column);
  if (!index.ok())
    return;
  ScanFilter filter;
  filter.column_index = index.ValueOrDie();
  filter.op = op;
  filter.literal = literal->literal;
  out.push_back(std::move(filter));
}

int CompareNumeric(const std::shared_ptr<arrow::Array>& left, int64_t i,
                   const std::shared_ptr<arrow::Array>& right, int64_t j,
                   const std::shared_ptr<arrow::DataType>& type) {
  switch (type->id()) {
  case arrow::Type::INT8: {
    auto& a = checked_cast<const arrow::Int8Array&>(*left);
    auto& b = checked_cast<const arrow::Int8Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::INT16: {
    auto& a = checked_cast<const arrow::Int16Array&>(*left);
    auto& b = checked_cast<const arrow::Int16Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::INT32: {
    auto& a = checked_cast<const arrow::Int32Array&>(*left);
    auto& b = checked_cast<const arrow::Int32Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::INT64: {
    auto& a = checked_cast<const arrow::Int64Array&>(*left);
    auto& b = checked_cast<const arrow::Int64Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::UINT8: {
    auto& a = checked_cast<const arrow::UInt8Array&>(*left);
    auto& b = checked_cast<const arrow::UInt8Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::UINT16: {
    auto& a = checked_cast<const arrow::UInt16Array&>(*left);
    auto& b = checked_cast<const arrow::UInt16Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::UINT32: {
    auto& a = checked_cast<const arrow::UInt32Array&>(*left);
    auto& b = checked_cast<const arrow::UInt32Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::UINT64: {
    auto& a = checked_cast<const arrow::UInt64Array&>(*left);
    auto& b = checked_cast<const arrow::UInt64Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  case arrow::Type::DATE32: {
    auto& a = checked_cast<const arrow::Date32Array&>(*left);
    auto& b = checked_cast<const arrow::Date32Array&>(*right);
    return a.Value(i) < b.Value(j) ? -1 : (a.Value(i) > b.Value(j) ? 1 : 0);
  }
  default:
    return 0;
  }
}

int CompareFloating(const std::shared_ptr<arrow::Array>& left, int64_t i,
                    const std::shared_ptr<arrow::Array>& right, int64_t j,
                    const std::shared_ptr<arrow::DataType>& type) {
  double a = 0;
  double b = 0;
  if (type->id() == arrow::Type::FLOAT) {
    a = checked_cast<const arrow::FloatArray&>(*left).Value(i);
    b = checked_cast<const arrow::FloatArray&>(*right).Value(j);
  } else {
    a = checked_cast<const arrow::DoubleArray&>(*left).Value(i);
    b = checked_cast<const arrow::DoubleArray&>(*right).Value(j);
  }
  const bool a_nan = std::isnan(a);
  const bool b_nan = std::isnan(b);
  if (a_nan || b_nan) {
    if (a_nan && b_nan)
      return 0;
    return a_nan ? 1 : -1;
  }
  return a < b ? -1 : (a > b ? 1 : 0);
}

int CompareArrayValues(const std::shared_ptr<arrow::Array>& left, int64_t i,
                       const std::shared_ptr<arrow::Array>& right, int64_t j,
                       const std::shared_ptr<arrow::DataType>& type) {
  switch (type->id()) {
  case arrow::Type::BOOL: {
    auto& a = checked_cast<const arrow::BooleanArray&>(*left);
    auto& b = checked_cast<const arrow::BooleanArray&>(*right);
    return a.Value(i) == b.Value(j) ? 0 : (a.Value(i) ? 1 : -1);
  }
  case arrow::Type::FLOAT:
  case arrow::Type::DOUBLE:
    return CompareFloating(left, i, right, j, type);
  case arrow::Type::STRING: {
    auto& a = checked_cast<const arrow::StringArray&>(*left);
    auto& b = checked_cast<const arrow::StringArray&>(*right);
    return a.GetView(i).compare(b.GetView(j));
  }
  case arrow::Type::LARGE_STRING: {
    auto& a = checked_cast<const arrow::LargeStringArray&>(*left);
    auto& b = checked_cast<const arrow::LargeStringArray&>(*right);
    return a.GetView(i).compare(b.GetView(j));
  }
  case arrow::Type::BINARY: {
    auto& a = checked_cast<const arrow::BinaryArray&>(*left);
    auto& b = checked_cast<const arrow::BinaryArray&>(*right);
    return a.GetView(i).compare(b.GetView(j));
  }
  default:
    return CompareNumeric(left, i, right, j, type);
  }
}

bool IsSortableType(const arrow::DataType& type) {
  switch (type.id()) {
  case arrow::Type::BOOL:
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
  case arrow::Type::FLOAT:
  case arrow::Type::DOUBLE:
  case arrow::Type::STRING:
  case arrow::Type::LARGE_STRING:
  case arrow::Type::BINARY:
  case arrow::Type::DATE32:
    return true;
  default:
    return false;
  }
}

std::string_view ArrayStringView(const arrow::Array& array, int64_t row) {
  const auto id = array.type_id();
  if (id == arrow::Type::STRING) {
    return checked_cast<const arrow::StringArray&>(array).GetView(row);
  }
  if (id == arrow::Type::LARGE_STRING) {
    return checked_cast<const arrow::LargeStringArray&>(array).GetView(row);
  }
  return checked_cast<const arrow::BinaryArray&>(array).GetView(row);
}

bool IsIntegerKeyType(const arrow::DataType& type) {
  switch (type.id()) {
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
  case arrow::Type::DATE32:
    return true;
  default:
    return false;
  }
}

bool IsStringKeyType(const arrow::DataType& type) {
  return type.id() == arrow::Type::STRING || type.id() == arrow::Type::LARGE_STRING ||
         type.id() == arrow::Type::BINARY;
}

uint64_t IntegerKeyAt(const arrow::Array& array, int64_t row) {
  switch (array.type_id()) {
  case arrow::Type::INT8:
    return static_cast<uint64_t>(
        static_cast<int64_t>(checked_cast<const arrow::Int8Array&>(array).Value(row)));
  case arrow::Type::INT16:
    return static_cast<uint64_t>(
        static_cast<int64_t>(checked_cast<const arrow::Int16Array&>(array).Value(row)));
  case arrow::Type::INT32:
    return static_cast<uint64_t>(
        static_cast<int64_t>(checked_cast<const arrow::Int32Array&>(array).Value(row)));
  case arrow::Type::INT64:
    return static_cast<uint64_t>(checked_cast<const arrow::Int64Array&>(array).Value(row));
  case arrow::Type::UINT8:
    return checked_cast<const arrow::UInt8Array&>(array).Value(row);
  case arrow::Type::UINT16:
    return checked_cast<const arrow::UInt16Array&>(array).Value(row);
  case arrow::Type::UINT32:
    return checked_cast<const arrow::UInt32Array&>(array).Value(row);
  case arrow::Type::UINT64:
    return checked_cast<const arrow::UInt64Array&>(array).Value(row);
  case arrow::Type::DATE32:
    return static_cast<uint64_t>(
        static_cast<int64_t>(checked_cast<const arrow::Date32Array&>(array).Value(row)));
  default:
    return 0;
  }
}

// Extracts a signed key for the dense grouping path. Unsigned values that do
// not fit in int64_t make the caller fall back to the general flat table.
bool SignedIntegerKeyAt(const arrow::Array& array, int64_t row, int64_t* value) {
  switch (array.type_id()) {
  case arrow::Type::INT8:
    *value = checked_cast<const arrow::Int8Array&>(array).Value(row);
    return true;
  case arrow::Type::INT16:
    *value = checked_cast<const arrow::Int16Array&>(array).Value(row);
    return true;
  case arrow::Type::INT32:
    *value = checked_cast<const arrow::Int32Array&>(array).Value(row);
    return true;
  case arrow::Type::INT64:
    *value = checked_cast<const arrow::Int64Array&>(array).Value(row);
    return true;
  case arrow::Type::UINT8:
    *value = checked_cast<const arrow::UInt8Array&>(array).Value(row);
    return true;
  case arrow::Type::UINT16:
    *value = checked_cast<const arrow::UInt16Array&>(array).Value(row);
    return true;
  case arrow::Type::UINT32:
    *value = checked_cast<const arrow::UInt32Array&>(array).Value(row);
    return true;
  case arrow::Type::UINT64: {
    const uint64_t value64 = checked_cast<const arrow::UInt64Array&>(array).Value(row);
    if (value64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    *value = static_cast<int64_t>(value64);
    return true;
  }
  case arrow::Type::DATE32:
    *value = checked_cast<const arrow::Date32Array&>(array).Value(row);
    return true;
  default:
    return false;
  }
}

uint64_t MixKey(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// Power-of-two slot capacity that keeps the open-addressed join table at or
// below a 0.5 load factor for the given row count.
size_t SlotCapacityFor(int64_t rows) {
  size_t capacity = 16;
  while (capacity < static_cast<size_t>(rows) * 2 && capacity < (size_t{1} << 62))
    capacity <<= 1;
  return capacity;
}

} // namespace

std::shared_ptr<arrow::RecordBatch> MakeEmptyBatch(const std::shared_ptr<arrow::Schema>& schema) {
  return MakeEmptyBatchImpl(schema);
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

ScanOperator::ScanOperator(std::shared_ptr<ParquetSource> source,
                           std::vector<std::string> file_columns,
                           std::vector<std::string> output_names, std::vector<sql::ExprPtr> filters,
                           int64_t batch_size, std::shared_ptr<ExecutionStats> stats)
    : source_(std::move(source)), file_columns_(std::move(file_columns)),
      output_names_(std::move(output_names)), filters_(std::move(filters)), batch_size_(batch_size),
      stats_(std::move(stats)) {
  if (file_columns_.size() != output_names_.size()) {
    init_status_ = arrow::Status::Invalid("scan column/name count mismatch");
    return;
  }
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (size_t i = 0; i < file_columns_.size(); ++i) {
    auto index = source_->FieldIndex(file_columns_[i]);
    if (!index.ok()) {
      init_status_ = index.status();
      return;
    }
    column_indices_.push_back(index.ValueOrDie());
    fields.push_back(
        arrow::field(output_names_[i], source_->schema()->field(index.ValueOrDie())->type()));
  }
  schema_ = arrow::schema(std::move(fields));
}

arrow::Status ScanOperator::Initialize() {
  initialized_ = true;
  scan_filters_.clear();
  for (const auto& filter : filters_) {
    AppendScanFilters(filter, *source_, scan_filters_);
  }

  const auto& metadata = source_->metadata();
  if (!metadata) {
    return arrow::Status::Invalid("Parquet metadata is unavailable for " + source_->path());
  }
  stats_->row_groups_total += source_->num_row_groups();
  for (int i = 0; i < source_->num_row_groups(); ++i) {
    auto row_group = metadata->RowGroup(i);
    if (RowGroupCanBeSkipped(*row_group, scan_filters_)) {
      ++stats_->row_groups_skipped;
    } else {
      row_groups_.push_back(i);
    }
  }
  stats_->row_groups_read += static_cast<int64_t>(row_groups_.size());
  if (row_groups_.empty()) {
    done_ = true;
    return arrow::Status::OK();
  }
  if (column_indices_.empty()) {
    // SELECT COUNT(*) with no predicate can be answered from Parquet metadata.
    metadata_only_ = true;
    for (int row_group : row_groups_) {
      metadata_rows_ += metadata->RowGroup(row_group)->num_rows();
    }
    stats_->rows_from_metadata += metadata_rows_;
    return arrow::Status::OK();
  }
  ARROW_ASSIGN_OR_RAISE(reader_, source_->OpenReader(row_groups_, column_indices_, batch_size_));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ScanOperator::Next() {
  if (!init_status_.ok())
    return init_status_;
  if (!initialized_) {
    ARROW_RETURN_NOT_OK(Initialize());
  }
  if (done_)
    return nullptr;
  if (metadata_only_) {
    done_ = true;
    if (metadata_rows_ <= 0)
      return nullptr;
    // A zero-column batch with a row count is valid Arrow and lets the
    // aggregate operator use its count-only fast path.
    return arrow::RecordBatch::Make(schema_, metadata_rows_,
                                    std::vector<std::shared_ptr<arrow::Array>>{});
  }

  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader_->Next());
    if (!batch) {
      done_ = true;
      return nullptr;
    }
    ++stats_->batches_scanned;
    stats_->rows_scanned += batch->num_rows();

    auto renamed = arrow::RecordBatch::Make(schema_, batch->num_rows(), batch->columns());
    for (const auto& filter : filters_) {
      ARROW_ASSIGN_OR_RAISE(renamed, ApplyFilter(*renamed, *filter));
      if (renamed->num_rows() == 0)
        break;
    }
    stats_->rows_after_pushdown += renamed->num_rows();
    if (renamed->num_rows() > 0)
      return renamed;
  }
}

// ---------------------------------------------------------------------------
// Filter / Project / Limit
// ---------------------------------------------------------------------------

arrow::Result<std::shared_ptr<arrow::RecordBatch>> FilterOperator::Next() {
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
    if (!batch)
      return nullptr;
    ARROW_ASSIGN_OR_RAISE(auto filtered, ApplyFilter(*batch, *predicate_));
    if (filtered->num_rows() > 0)
      return filtered;
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ProjectOperator::Next() {
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  if (!batch)
    return nullptr;
  return ApplyProjection(*batch, projections_, output_names_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> LimitOperator::Next() {
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
    if (!batch)
      return nullptr;
    if (offset_remaining_ > 0) {
      if (batch->num_rows() <= offset_remaining_) {
        offset_remaining_ -= batch->num_rows();
        continue;
      }
      batch = batch->Slice(offset_remaining_, batch->num_rows() - offset_remaining_);
      offset_remaining_ = 0;
    }
    if (remaining_ < 0)
      return batch;
    if (remaining_ == 0)
      return nullptr;
    if (batch->num_rows() <= remaining_) {
      remaining_ -= batch->num_rows();
      return batch;
    }
    auto sliced = batch->Slice(0, remaining_);
    remaining_ = 0;
    return sliced;
  }
}

// ---------------------------------------------------------------------------
// Distinct
// ---------------------------------------------------------------------------

arrow::Result<std::shared_ptr<arrow::RecordBatch>> DistinctOperator::Next() {
  if (done_)
    return nullptr;
  done_ = true;
  ARROW_ASSIGN_OR_RAISE(auto batch, MaterializeOperator(*child_));
  if (batch->num_rows() <= 1) {
    result_ = std::move(batch);
    return result_;
  }

  std::unordered_set<std::string> seen;
  seen.reserve(static_cast<size_t>(batch->num_rows()));
  std::vector<int64_t> indices;
  indices.reserve(static_cast<size_t>(batch->num_rows()));
  std::string key;
  key.reserve(static_cast<size_t>(batch->num_columns()) * 10);
  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    key.clear();
    for (int column = 0; column < batch->num_columns(); ++column) {
      ARROW_RETURN_NOT_OK(AppendArrayKey(*batch->column(column), row, key));
    }
    if (seen.insert(key).second)
      indices.push_back(row);
  }
  if (indices.size() == static_cast<size_t>(batch->num_rows())) {
    result_ = std::move(batch);
    return result_;
  }

  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(indices.size())));
  for (int64_t index : indices)
    builder.UnsafeAppend(index);
  ARROW_ASSIGN_OR_RAISE(auto index_array, builder.Finish());

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(batch->num_columns());
  for (int column = 0; column < batch->num_columns(); ++column) {
    ARROW_ASSIGN_OR_RAISE(auto taken, arrow::compute::Take(*batch->column(column), *index_array));
    columns.push_back(std::move(taken));
  }
  result_ = arrow::RecordBatch::Make(batch->schema(), static_cast<int64_t>(indices.size()),
                                     std::move(columns));
  return result_;
}

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

arrow::Result<std::shared_ptr<arrow::RecordBatch>> SortOperator::Materialize() {
  ARROW_ASSIGN_OR_RAISE(auto batch, MaterializeOperator(*child_));
  if (batch->num_rows() <= 1)
    return batch;

  std::vector<std::shared_ptr<arrow::Array>> keys;
  keys.reserve(sort_keys_.size());
  for (const auto& key : sort_keys_) {
    ARROW_ASSIGN_OR_RAISE(auto array, EvaluateExpressionArray(*key.expression, *batch));
    if (!IsSortableType(*array->type())) {
      return arrow::Status::NotImplemented("ORDER BY type " + array->type()->ToString() +
                                           " is not supported");
    }
    if (array->length() != batch->num_rows()) {
      return arrow::Status::Invalid("ORDER BY expression produced the wrong number of rows");
    }
    keys.push_back(std::move(array));
  }

  std::vector<int64_t> indices(static_cast<size_t>(batch->num_rows()));
  std::iota(indices.begin(), indices.end(), 0);
  auto before = [&](int64_t i, int64_t j) {
    for (size_t k = 0; k < keys.size(); ++k) {
      const bool left_null = !keys[k]->IsValid(i);
      const bool right_null = !keys[k]->IsValid(j);
      if (left_null || right_null) {
        if (left_null && right_null)
          continue;
        return left_null ? false : true; // NULLs last, like DuckDB
      }
      int cmp = CompareArrayValues(keys[k], i, keys[k], j, keys[k]->type());
      if (cmp != 0)
        return sort_keys_[k].descending ? cmp > 0 : cmp < 0;
    }
    // Stable order for ties. This also makes nth_element deterministic below.
    return i < j;
  };

  int64_t output_rows = batch->num_rows();
  if (limit_hint_ == 0) {
    output_rows = 0;
    indices.clear();
  } else if (limit_hint_ > 0 && limit_hint_ < batch->num_rows()) {
    // Top-K: partition first, then sort only the K survivors. This avoids an
    // O(n log n) full sort for the common ORDER BY ... LIMIT shape.
    output_rows = limit_hint_;
    std::nth_element(indices.begin(), indices.begin() + limit_hint_, indices.end(), before);
    indices.resize(static_cast<size_t>(limit_hint_));
  }
  if (output_rows > 1) {
    std::stable_sort(indices.begin(), indices.end(), before);
  }

  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(output_rows));
  for (int64_t index : indices)
    builder.UnsafeAppend(index);
  ARROW_ASSIGN_OR_RAISE(auto index_array, builder.Finish());

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(batch->num_columns());
  for (int i = 0; i < batch->num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto taken, arrow::compute::Take(*batch->column(i), *index_array));
    columns.push_back(std::move(taken));
  }
  return arrow::RecordBatch::Make(batch->schema(), output_rows, std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> SortOperator::Next() {
  if (done_)
    return nullptr;
  done_ = true;
  ARROW_ASSIGN_OR_RAISE(result_, Materialize());
  return result_;
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

void AggregateOperator::AddGroup(Group group) {
  groups_.push_back(std::move(group));
  const size_t needed = groups_.size() * aggregates_.size();
  if (needed > states_.size()) {
    // Grow in blocks so high-cardinality grouping does not call resize once
    // per group. The extra initialized states are harmless.
    const size_t grown_groups = std::max<size_t>(256, groups_.size() * 2);
    states_.resize(grown_groups * aggregates_.size());
  }
}

bool AggregateOperator::ExpandDenseKey(int64_t key) {
  if (key >= dense_min_key_ && key <= dense_max_key_)
    return true;
  const uint64_t span =
      static_cast<uint64_t>(dense_max_key_) - static_cast<uint64_t>(dense_min_key_) + 1;
  if (key < dense_min_key_) {
    const uint64_t delta = static_cast<uint64_t>(dense_min_key_) - static_cast<uint64_t>(key);
    if (delta > kMaxDenseRange - span)
      return false;
    dense_group_ids_.insert(dense_group_ids_.begin(), static_cast<size_t>(delta), int32_t{-1});
    dense_min_key_ = key;
  } else {
    const uint64_t delta = static_cast<uint64_t>(key) - static_cast<uint64_t>(dense_max_key_);
    if (delta > kMaxDenseRange - span)
      return false;
    dense_group_ids_.resize(dense_group_ids_.size() + static_cast<size_t>(delta), int32_t{-1});
    dense_max_key_ = key;
  }
  return true;
}

void AggregateOperator::DisableDense() {
  if (!dense_active_)
    return;

  size_t capacity = 16;
  while (capacity < (groups_.size() + 1) * 2)
    capacity <<= 1;
  flat_slots_.assign(capacity, FlatSlot{});
  flat_size_ = 0;
  for (size_t i = 0; i < dense_group_ids_.size(); ++i) {
    const int32_t group = dense_group_ids_[i];
    if (group < 0)
      continue;
    bool inserted = false;
    const uint64_t key = static_cast<uint64_t>(dense_min_key_) + i;
    const size_t slot = FlatFindOrInsert(key, false, &inserted);
    flat_slots_[slot].group = static_cast<size_t>(group);
  }
  if (dense_null_group_ >= 0) {
    bool inserted = false;
    const size_t slot = FlatFindOrInsert(0, true, &inserted);
    flat_slots_[slot].group = static_cast<size_t>(dense_null_group_);
  }
  dense_group_ids_.clear();
  dense_null_group_ = -1;
  dense_active_ = false;
}

arrow::Status
AggregateOperator::BuildGroupIds(int64_t num_rows,
                                 const std::vector<std::shared_ptr<arrow::Array>>& group_arrays,
                                 std::vector<size_t>& group_ids) {
  group_ids.resize(static_cast<size_t>(num_rows));
  if (group_by_.empty()) {
    if (groups_.empty())
      AddGroup(Group{});
    std::fill(group_ids.begin(), group_ids.end(), size_t{0});
    return arrow::Status::OK();
  }

  if (group_by_.size() == 1) {
    const auto& key_type = *group_arrays[0]->type();
    if (IsIntegerKeyType(key_type)) {
      group_key_mode_ = 1;
    } else if (IsStringKeyType(key_type)) {
      group_key_mode_ = 2;
    } else {
      group_key_mode_ = -1;
    }
  } else {
    group_key_mode_ = -1;
  }

  if (group_key_mode_ == 1) {
    const auto& key_array = *group_arrays[0];
    auto add_key_group = [&](int64_t row) -> arrow::Result<size_t> {
      const size_t group_index = groups_.size();
      Group group;
      group.keys.reserve(1);
      ARROW_ASSIGN_OR_RAISE(auto scalar, key_array.GetScalar(row));
      group.keys.push_back(std::move(scalar));
      AddGroup(std::move(group));
      return group_index;
    };

    if (!dense_active_ && groups_.empty() && flat_size_ == 0) {
      bool have_key = false;
      bool supported = true;
      int64_t min_key = 0;
      int64_t max_key = 0;
      for (int64_t row = 0; row < num_rows; ++row) {
        if (key_array.IsNull(row))
          continue;
        int64_t key = 0;
        if (!SignedIntegerKeyAt(key_array, row, &key)) {
          supported = false;
          break;
        }
        if (!have_key || key < min_key)
          min_key = key;
        if (!have_key || key > max_key)
          max_key = key;
        have_key = true;
      }
      if (supported && have_key) {
        const uint64_t span = static_cast<uint64_t>(max_key) - static_cast<uint64_t>(min_key);
        if (span < kMaxDenseRange) {
          dense_min_key_ = min_key;
          dense_max_key_ = max_key;
          dense_group_ids_.assign(static_cast<size_t>(span + 1), int32_t{-1});
          dense_null_group_ = -1;
          dense_active_ = true;
        }
      }
    }

    if (dense_active_) {
      bool dense_ok = true;
      for (int64_t row = 0; row < num_rows; ++row) {
        const bool is_null = key_array.IsNull(row);
        size_t group_index;
        if (is_null) {
          if (dense_null_group_ < 0) {
            if (groups_.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
              dense_ok = false;
              break;
            }
            ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
            dense_null_group_ = static_cast<int32_t>(group_index);
          } else {
            group_index = static_cast<size_t>(dense_null_group_);
          }
        } else {
          int64_t key = 0;
          if (!SignedIntegerKeyAt(key_array, row, &key) || !ExpandDenseKey(key)) {
            dense_ok = false;
            break;
          }
          const size_t index = static_cast<size_t>(static_cast<uint64_t>(key) -
                                                   static_cast<uint64_t>(dense_min_key_));
          const int32_t existing = dense_group_ids_[index];
          if (existing >= 0) {
            group_index = static_cast<size_t>(existing);
          } else {
            if (groups_.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
              dense_ok = false;
              break;
            }
            ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
            dense_group_ids_[index] = static_cast<int32_t>(group_index);
          }
        }
        group_ids[static_cast<size_t>(row)] = group_index;
      }
      if (dense_ok)
        return arrow::Status::OK();
      DisableDense();
      // Rebuild this batch through the flat table. Groups created during the
      // partial dense pass are retained.
    }

    for (int64_t row = 0; row < num_rows; ++row) {
      const bool is_null = key_array.IsNull(row);
      const uint64_t key = is_null ? 0 : IntegerKeyAt(key_array, row);
      bool inserted = false;
      const size_t slot = FlatFindOrInsert(key, is_null, &inserted);
      size_t group_index;
      if (inserted) {
        ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
        flat_slots_[slot].group = group_index;
      } else {
        group_index = flat_slots_[slot].group;
      }
      group_ids[static_cast<size_t>(row)] = group_index;
    }
    return arrow::Status::OK();
  }

  if (group_key_mode_ == 2) {
    const auto& key_array = *group_arrays[0];
    auto add_key_group = [&](int64_t row) -> arrow::Result<size_t> {
      const size_t group_index = groups_.size();
      Group group;
      group.keys.reserve(1);
      ARROW_ASSIGN_OR_RAISE(auto scalar, key_array.GetScalar(row));
      group.keys.push_back(std::move(scalar));
      AddGroup(std::move(group));
      return group_index;
    };

    for (int64_t row = 0; row < num_rows; ++row) {
      if (!string_map_active_ && small_string_groups_.size() >= kSmallStringGroups) {
        group_index_.reserve(kSmallStringGroups * 2);
        for (auto& entry : small_string_groups_)
          group_index_.emplace(std::move(entry.key), entry.group);
        small_string_groups_.clear();
        string_map_active_ = true;
      }

      if (key_array.IsNull(row)) {
        size_t group_index;
        if (small_null_group_ < 0) {
          ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
          small_null_group_ = static_cast<int32_t>(group_index);
        } else {
          group_index = static_cast<size_t>(small_null_group_);
        }
        group_ids[static_cast<size_t>(row)] = group_index;
        continue;
      }

      const std::string_view key = ArrayStringView(key_array, row);
      size_t group_index = 0;
      if (!string_map_active_) {
        bool found = false;
        for (const auto& entry : small_string_groups_) {
          if (entry.key == key) {
            group_index = entry.group;
            found = true;
            break;
          }
        }
        if (!found) {
          ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
          small_string_groups_.push_back({std::string(key), group_index});
        }
      } else {
        auto found = group_index_.find(key);
        if (found == group_index_.end()) {
          ARROW_ASSIGN_OR_RAISE(group_index, add_key_group(row));
          group_index_.emplace(std::string(key), group_index);
        } else {
          group_index = found->second;
        }
      }
      group_ids[static_cast<size_t>(row)] = group_index;
    }
    return arrow::Status::OK();
  }

  // Generic keys are serialized once per row, then the aggregate values are
  // accumulated in separate batch passes. This keeps the hot aggregate loops
  // free of key construction, map lookups, and per-aggregate branches.
  std::string key;
  key.reserve(group_arrays.size() * 10);
  for (int64_t row = 0; row < num_rows; ++row) {
    key.clear();
    for (const auto& array : group_arrays) {
      ARROW_RETURN_NOT_OK(AppendArrayKey(*array, row, key));
    }

    auto found = group_index_.find(key);
    size_t group_index;
    if (found == group_index_.end()) {
      group_index = groups_.size();
      group_index_.emplace(key, group_index);
      Group group;
      group.keys.reserve(group_arrays.size());
      for (const auto& array : group_arrays) {
        ARROW_ASSIGN_OR_RAISE(auto scalar, array->GetScalar(row));
        group.keys.push_back(std::move(scalar));
      }
      AddGroup(std::move(group));
    } else {
      group_index = found->second;
    }
    group_ids[static_cast<size_t>(row)] = group_index;
  }
  return arrow::Status::OK();
}

arrow::Status AggregateOperator::AccumulateBatch(
    const std::vector<std::shared_ptr<arrow::Array>>& argument_arrays,
    const std::vector<size_t>& group_ids) {
  const int64_t num_rows = static_cast<int64_t>(group_ids.size());
  const size_t num_aggregates = aggregates_.size();

  // The common analytical shape (COUNT/SUM/AVG with no MIN/MAX) is fused into
  // one pass. Each row updates all of its group's adjacent states, so a
  // high-cardinality query pays one random access per row instead of one per
  // aggregate. Values and validity bitmaps are resolved once per batch.
  bool simple_aggregates = true;
  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto function = aggregates_[i]->function;
    if (function == "min" || function == "max") {
      const auto type_id = argument_arrays[i]->type_id();
      if (type_id != arrow::Type::FLOAT && type_id != arrow::Type::DOUBLE) {
        simple_aggregates = false;
        break;
      }
      continue;
    }
    if (function != "count" && function != "sum" && function != "avg") {
      simple_aggregates = false;
      break;
    }
  }

  if (simple_aggregates) {
    enum class FastKind : uint8_t {
      kCountStar,
      kCount,
      kSumDouble,
      kSumInt,
      kAvg,
      kMinDouble,
      kMaxDouble
    };
    struct FastView {
      FastKind kind = FastKind::kCountStar;
      const double* doubles = nullptr;
      const int64_t* ints = nullptr;
      const uint8_t* bitmap = nullptr;
      int64_t offset = 0;
    };

    std::vector<FastView> views(num_aggregates);
    std::vector<std::shared_ptr<arrow::Array>> keepalive(num_aggregates);
    for (size_t i = 0; i < num_aggregates; ++i) {
      const auto& aggregate = *aggregates_[i];
      FastView& view = views[i];
      if (aggregate.star) {
        view.kind = FastKind::kCountStar;
        continue;
      }

      const auto& arguments = *argument_arrays[i];
      view.bitmap = arguments.null_bitmap_data();
      view.offset = arguments.offset();
      if (aggregate.function == "count") {
        view.kind = FastKind::kCount;
        continue;
      }

      if (aggregate.function == "min" || aggregate.function == "max") {
        std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
        if (numeric->type_id() != arrow::Type::DOUBLE) {
          ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::float64()));
          numeric = casted;
          keepalive[i] = numeric;
        }
        const auto& values = checked_cast<const arrow::DoubleArray&>(*numeric);
        view.kind = aggregate.function == "min" ? FastKind::kMinDouble : FastKind::kMaxDouble;
        view.doubles = values.raw_values();
        view.bitmap = values.null_bitmap_data();
        view.offset = values.offset();
        continue;
      }

      const bool double_output =
          output_schema_->field(static_cast<int>(group_by_.size() + i))->type()->id() ==
          arrow::Type::DOUBLE;
      if (aggregate.function == "avg" || double_output) {
        std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
        if (numeric->type_id() != arrow::Type::DOUBLE) {
          ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::float64()));
          numeric = casted;
          keepalive[i] = numeric;
        }
        const auto& values = checked_cast<const arrow::DoubleArray&>(*numeric);
        view.kind = aggregate.function == "avg" ? FastKind::kAvg : FastKind::kSumDouble;
        view.doubles = values.raw_values();
        view.bitmap = values.null_bitmap_data();
        view.offset = values.offset();
      } else {
        std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
        if (numeric->type_id() != arrow::Type::INT64) {
          ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::int64()));
          numeric = casted;
          keepalive[i] = numeric;
        }
        const auto& values = checked_cast<const arrow::Int64Array&>(*numeric);
        view.kind = FastKind::kSumInt;
        view.ints = values.raw_values();
        view.bitmap = values.null_bitmap_data();
        view.offset = values.offset();
      }
    }

    AggState* states = states_.data();
    for (int64_t row = 0; row < num_rows; ++row) {
      AggState* group_state = states + group_ids[static_cast<size_t>(row)] * num_aggregates;
      for (size_t i = 0; i < num_aggregates; ++i) {
        const FastView& view = views[i];
        AggState& state = group_state[i];
        if (view.kind == FastKind::kCountStar) {
          ++state.count;
          continue;
        }
        if (view.bitmap != nullptr && !arrow::bit_util::GetBit(view.bitmap, view.offset + row)) {
          continue;
        }
        switch (view.kind) {
        case FastKind::kCount:
          ++state.count;
          break;
        case FastKind::kSumDouble:
          state.double_sum += view.doubles[view.offset + row];
          state.has_value = true;
          break;
        case FastKind::kSumInt:
          state.int_sum += view.ints[view.offset + row];
          state.has_value = true;
          break;
        case FastKind::kAvg:
          state.double_sum += view.doubles[view.offset + row];
          ++state.count;
          state.has_value = true;
          break;
        case FastKind::kMinDouble:
        case FastKind::kMaxDouble: {
          const double value = view.doubles[view.offset + row];
          const bool better = !state.has_value ||
                              (view.kind == FastKind::kMinDouble ? value < state.double_extreme
                                                                 : value > state.double_extreme);
          if (better) {
            state.extreme_kind = ExtremeKind::kDouble;
            state.double_extreme = value;
            state.has_value = true;
          }
          break;
        }
        case FastKind::kCountStar:
          break;
        }
      }
    }
    return arrow::Status::OK();
  }

  const size_t state_stride = aggregates_.size();
  for (size_t i = 0; i < state_stride; ++i) {
    const auto& aggregate = *aggregates_[i];
    auto state_at = [&](int64_t row) -> AggState& {
      return states_[group_ids[static_cast<size_t>(row)] * state_stride + i];
    };
    if (aggregate.star) {
      for (int64_t row = 0; row < num_rows; ++row) {
        ++state_at(row).count;
      }
      continue;
    }

    const auto& arguments = *argument_arrays[i];
    if (aggregate.function == "count") {
      const uint8_t* bitmap = arguments.null_bitmap_data();
      const int64_t offset = arguments.offset();
      if (bitmap == nullptr) {
        for (int64_t row = 0; row < num_rows; ++row) {
          ++state_at(row).count;
        }
      } else {
        for (int64_t row = 0; row < num_rows; ++row) {
          if (arrow::bit_util::GetBit(bitmap, offset + row)) {
            ++state_at(row).count;
          }
        }
      }
      continue;
    }

    if (aggregate.function == "sum" || aggregate.function == "avg") {
      const bool double_output =
          output_schema_->field(static_cast<int>(group_by_.size() + i))->type()->id() ==
          arrow::Type::DOUBLE;
      if (aggregate.function == "avg" || double_output) {
        std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
        if (numeric->type_id() != arrow::Type::DOUBLE) {
          ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::float64()));
          numeric = casted;
        }
        const auto& values = checked_cast<const arrow::DoubleArray&>(*numeric);
        const double* data = values.raw_values();
        const uint8_t* bitmap = values.null_bitmap_data();
        const int64_t offset = values.offset();
        if (bitmap == nullptr) {
          for (int64_t row = 0; row < num_rows; ++row) {
            AggState& state = state_at(row);
            state.double_sum += data[offset + row];
            state.has_value = true;
            if (aggregate.function == "avg")
              ++state.count;
          }
        } else {
          for (int64_t row = 0; row < num_rows; ++row) {
            if (!arrow::bit_util::GetBit(bitmap, offset + row))
              continue;
            AggState& state = state_at(row);
            state.double_sum += data[offset + row];
            state.has_value = true;
            if (aggregate.function == "avg")
              ++state.count;
          }
        }
      } else {
        std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
        if (numeric->type_id() != arrow::Type::INT64) {
          ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::int64()));
          numeric = casted;
        }
        const auto& values = checked_cast<const arrow::Int64Array&>(*numeric);
        const int64_t* data = values.raw_values();
        const uint8_t* bitmap = values.null_bitmap_data();
        const int64_t offset = values.offset();
        if (bitmap == nullptr) {
          for (int64_t row = 0; row < num_rows; ++row) {
            AggState& state = state_at(row);
            state.int_sum += data[offset + row];
            state.has_value = true;
          }
        } else {
          for (int64_t row = 0; row < num_rows; ++row) {
            if (!arrow::bit_util::GetBit(bitmap, offset + row))
              continue;
            AggState& state = state_at(row);
            state.int_sum += data[offset + row];
            state.has_value = true;
          }
        }
      }
      continue;
    }

    if (aggregate.function != "min" && aggregate.function != "max") {
      return arrow::Status::Invalid("unsupported aggregate function " + aggregate.function);
    }

    const bool want_min = aggregate.function == "min";
    const auto type_id = arguments.type_id();
    if (arrow::is_integer(type_id) || type_id == arrow::Type::DATE32) {
      std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
      if (type_id != arrow::Type::INT64 && type_id != arrow::Type::DATE32) {
        ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::int64()));
        numeric = casted;
      }
      const int64_t* data = nullptr;
      const uint8_t* bitmap = numeric->null_bitmap_data();
      const int64_t offset = numeric->offset();
      const bool is_date = numeric->type_id() == arrow::Type::DATE32;
      std::vector<int64_t> dates;
      if (is_date) {
        const auto& date_values = checked_cast<const arrow::Date32Array&>(*numeric);
        dates.resize(static_cast<size_t>(num_rows));
        for (int64_t row = 0; row < num_rows; ++row)
          dates[static_cast<size_t>(row)] = date_values.Value(row);
        data = dates.data();
      } else {
        data = checked_cast<const arrow::Int64Array&>(*numeric).raw_values();
      }
      if (bitmap == nullptr) {
        for (int64_t row = 0; row < num_rows; ++row) {
          AggState& state = state_at(row);
          const int64_t value = is_date ? data[row] : data[offset + row];
          if (!state.has_value ||
              (want_min ? value < state.int_extreme : value > state.int_extreme)) {
            state.extreme_kind = ExtremeKind::kInt64;
            state.int_extreme = value;
            state.has_value = true;
          }
        }
      } else {
        for (int64_t row = 0; row < num_rows; ++row) {
          if (!arrow::bit_util::GetBit(bitmap, offset + row))
            continue;
          AggState& state = state_at(row);
          const int64_t value = is_date ? data[row] : data[offset + row];
          if (!state.has_value ||
              (want_min ? value < state.int_extreme : value > state.int_extreme)) {
            state.extreme_kind = ExtremeKind::kInt64;
            state.int_extreme = value;
            state.has_value = true;
          }
        }
      }
      continue;
    }

    if (type_id == arrow::Type::FLOAT || type_id == arrow::Type::DOUBLE) {
      std::shared_ptr<arrow::Array> numeric = argument_arrays[i];
      if (type_id != arrow::Type::DOUBLE) {
        ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(*numeric, arrow::float64()));
        numeric = casted;
      }
      const auto& values = checked_cast<const arrow::DoubleArray&>(*numeric);
      const double* data = values.raw_values();
      const uint8_t* bitmap = values.null_bitmap_data();
      const int64_t offset = values.offset();
      if (bitmap == nullptr) {
        for (int64_t row = 0; row < num_rows; ++row) {
          AggState& state = state_at(row);
          const double value = data[offset + row];
          if (!state.has_value ||
              (want_min ? value < state.double_extreme : value > state.double_extreme)) {
            state.extreme_kind = ExtremeKind::kDouble;
            state.double_extreme = value;
            state.has_value = true;
          }
        }
      } else {
        for (int64_t row = 0; row < num_rows; ++row) {
          if (!arrow::bit_util::GetBit(bitmap, offset + row))
            continue;
          AggState& state = state_at(row);
          const double value = data[offset + row];
          if (!state.has_value ||
              (want_min ? value < state.double_extreme : value > state.double_extreme)) {
            state.extreme_kind = ExtremeKind::kDouble;
            state.double_extreme = value;
            state.has_value = true;
          }
        }
      }
      continue;
    }

    if (type_id == arrow::Type::BOOL) {
      const auto& values = checked_cast<const arrow::BooleanArray&>(arguments);
      const uint8_t* bitmap = values.null_bitmap_data();
      for (int64_t row = 0; row < num_rows; ++row) {
        if (bitmap != nullptr && !arrow::bit_util::GetBit(bitmap, values.offset() + row))
          continue;
        AggState& state = state_at(row);
        const bool value = values.Value(row);
        if (!state.has_value ||
            (want_min ? value < state.bool_extreme : value > state.bool_extreme)) {
          state.extreme_kind = ExtremeKind::kBool;
          state.bool_extreme = value;
          state.has_value = true;
        }
      }
      continue;
    }

    if (type_id == arrow::Type::STRING || type_id == arrow::Type::LARGE_STRING ||
        type_id == arrow::Type::BINARY) {
      const uint8_t* bitmap = arguments.null_bitmap_data();
      const int64_t offset = arguments.offset();
      for (int64_t row = 0; row < num_rows; ++row) {
        if (bitmap != nullptr && !arrow::bit_util::GetBit(bitmap, offset + row))
          continue;
        AggState& state = state_at(row);
        const std::string_view value = ArrayStringView(arguments, row);
        if (!state.has_value) {
          state.extreme_kind = ExtremeKind::kString;
          state.string_extreme = std::make_unique<std::string>(value);
          state.has_value = true;
        } else {
          const int cmp = state.string_extreme->compare(value);
          if ((want_min && cmp > 0) || (!want_min && cmp < 0)) {
            state.string_extreme->assign(value);
          }
        }
      }
      continue;
    }

    return arrow::Status::Invalid("MIN/MAX does not support type " + arguments.type()->ToString());
  }
  return arrow::Status::OK();
}

void AggregateOperator::FlatGrow() {
  const size_t capacity = flat_slots_.empty() ? 16 : flat_slots_.size() * 2;
  std::vector<FlatSlot> old = std::move(flat_slots_);
  flat_slots_.assign(capacity, FlatSlot{});
  flat_size_ = 0;
  const size_t mask = capacity - 1;
  for (const FlatSlot& slot : old) {
    if (!slot.used)
      continue;
    size_t index = MixKey(slot.key ^ (slot.is_null ? 0x9e3779b97f4a7c15ULL : 0)) & mask;
    while (flat_slots_[index].used)
      index = (index + 1) & mask;
    flat_slots_[index] = slot;
    ++flat_size_;
  }
}

size_t AggregateOperator::FlatFindOrInsert(uint64_t key, bool is_null, bool* inserted) {
  if (flat_slots_.empty() || (flat_size_ + 1) * 10 >= flat_slots_.size() * 7) {
    FlatGrow();
  }
  const size_t mask = flat_slots_.size() - 1;
  size_t index = MixKey(key ^ (is_null ? 0x9e3779b97f4a7c15ULL : 0)) & mask;
  while (flat_slots_[index].used) {
    if (flat_slots_[index].key == key && flat_slots_[index].is_null == (is_null ? 1 : 0)) {
      *inserted = false;
      return index;
    }
    index = (index + 1) & mask;
  }
  flat_slots_[index].used = 1;
  flat_slots_[index].key = key;
  flat_slots_[index].is_null = is_null ? 1 : 0;
  flat_slots_[index].group = 0;
  ++flat_size_;
  *inserted = true;
  return index;
}

arrow::Status AggregateOperator::Consume(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (group_by_.empty() && !aggregates_.empty() &&
      std::all_of(aggregates_.begin(), aggregates_.end(),
                  [](const sql::ExprPtr& aggregate) { return aggregate->star; })) {
    if (groups_.empty())
      AddGroup(Group{});
    for (size_t i = 0; i < aggregates_.size(); ++i)
      State(i, 0).count += batch->num_rows();
    return arrow::Status::OK();
  }

  std::vector<std::shared_ptr<arrow::Array>> group_arrays;
  group_arrays.reserve(group_by_.size());
  for (const auto& expr : group_by_) {
    ARROW_ASSIGN_OR_RAISE(auto array, EvaluateExpressionArray(*expr, *batch));
    group_arrays.push_back(std::move(array));
  }

  std::vector<std::shared_ptr<arrow::Array>> argument_arrays(aggregates_.size());
  for (size_t i = 0; i < aggregates_.size(); ++i) {
    if (!aggregates_[i]->star) {
      ARROW_ASSIGN_OR_RAISE(argument_arrays[i],
                            EvaluateExpressionArray(*aggregates_[i]->argument, *batch));
    }
  }

  if (use_gpu_ && gpu_mode_ != -1 && group_by_.empty()) {
    auto gpu_status = ConsumeGpu(batch, argument_arrays);
    if (gpu_status.ok())
      return gpu_status;
    if (gpu_mode_ == 0 && gpu_status.IsNotImplemented()) {
      gpu_mode_ = -1;
    } else {
      return gpu_status;
    }
  }

  ARROW_RETURN_NOT_OK(BuildGroupIds(batch->num_rows(), group_arrays, batch_group_ids_));
  return AccumulateBatch(argument_arrays, batch_group_ids_);
}

arrow::Status
AggregateOperator::ConsumeGpu(const std::shared_ptr<arrow::RecordBatch>& batch,
                              const std::vector<std::shared_ptr<arrow::Array>>& argument_arrays) {
  if (gpu_mode_ == 0) {
    bool has_gpu_work = false;
    for (size_t i = 0; i < aggregates_.size(); ++i) {
      const auto& aggregate = *aggregates_[i];
      if (aggregate.star) {
        if (aggregate.function != "count") {
          return arrow::Status::NotImplemented("GPU aggregate does not support " +
                                               aggregate.function + "(*)");
        }
        continue;
      }
      if (aggregate.function == "count")
        continue;
      if (aggregate.function != "sum" && aggregate.function != "avg") {
        return arrow::Status::NotImplemented("GPU aggregate does not support " +
                                             aggregate.function);
      }
      const auto out_type = output_schema_->field(static_cast<int>(group_by_.size() + i))->type();
      if (out_type->id() != arrow::Type::DOUBLE) {
        return arrow::Status::NotImplemented(
            "GPU aggregate currently supports only double-precision SUM/AVG results");
      }
      const auto& arguments = argument_arrays[i];
      if (!arguments) {
        return arrow::Status::Invalid("GPU aggregate argument was not evaluated");
      }
      if (arguments->type()->id() != arrow::Type::DOUBLE &&
          arguments->type()->id() != arrow::Type::FLOAT) {
        return arrow::Status::NotImplemented(
            "GPU aggregate currently supports only FLOAT/DOUBLE SUM/AVG arguments");
      }
      if (arguments->length() > 0 &&
          (arguments->data()->buffers.size() < 2 || !arguments->data()->buffers[1])) {
        return arrow::Status::Invalid("GPU aggregate received a numeric array without values");
      }
      has_gpu_work = true;
    }
    if (!has_gpu_work) {
      return arrow::Status::NotImplemented("no GPU-supported aggregate in this query");
    }

    gpu_workspace_ = std::make_unique<gpu::Workspace>();
    if (!gpu_workspace_->available()) {
      return arrow::Status::NotImplemented("GPU workspace unavailable: " + gpu_workspace_->error());
    }
    gpu_buffers_.resize(aggregates_.size());
    for (size_t i = 0; i < aggregates_.size(); ++i) {
      const auto& aggregate = *aggregates_[i];
      if (aggregate.function == "sum" || aggregate.function == "avg") {
        gpu_buffers_[i].values.reserve(kGpuBufferValues);
      }
    }
    gpu_mode_ = 1;
    stats_->gpu_attempted = true;
  }

  if (groups_.empty())
    AddGroup(Group{});

  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto& aggregate = *aggregates_[i];
    if (aggregate.function == "count") {
      if (aggregate.star) {
        State(i, 0).count += batch->num_rows();
      } else {
        const auto& arguments = argument_arrays[i];
        State(i, 0).count += arguments->length() - arguments->null_count();
      }
      continue;
    }

    const auto& arguments = argument_arrays[i];
    const int64_t length = arguments->length();
    if (length == 0)
      continue;
    GpuBuffer& buffer = gpu_buffers_[i];
    if (arguments->type()->id() == arrow::Type::DOUBLE) {
      const double* values = reinterpret_cast<const double*>(arguments->data()->buffers[1]->data());
      if (arguments->null_count() == 0) {
        buffer.values.insert(buffer.values.end(), values + arguments->offset(),
                             values + arguments->offset() + length);
      } else {
        for (int64_t row = 0; row < length; ++row) {
          if (!arguments->IsNull(row))
            buffer.values.push_back(values[arguments->offset() + row]);
        }
      }
    } else {
      const float* values = reinterpret_cast<const float*>(arguments->data()->buffers[1]->data());
      if (arguments->null_count() == 0) {
        for (int64_t row = 0; row < length; ++row) {
          buffer.values.push_back(static_cast<double>(values[arguments->offset() + row]));
        }
      } else {
        for (int64_t row = 0; row < length; ++row) {
          if (!arguments->IsNull(row)) {
            buffer.values.push_back(static_cast<double>(values[arguments->offset() + row]));
          }
        }
      }
    }

    if (buffer.values.size() >= kGpuBufferValues) {
      ARROW_RETURN_NOT_OK(FlushGpuBuffer(i));
    }
  }
  return arrow::Status::OK();
}

arrow::Status AggregateOperator::FlushGpuBuffer(size_t index) {
  if (index >= gpu_buffers_.size())
    return arrow::Status::OK();
  GpuBuffer& buffer = gpu_buffers_[index];
  if (buffer.values.empty())
    return arrow::Status::OK();
  if (!gpu_workspace_ || !gpu_workspace_->available()) {
    return arrow::Status::Invalid("GPU workspace is not available");
  }

  gpu::Timings timings;
  gpu::Reduction reduction = gpu_workspace_->SumDouble(
      buffer.values.data(), nullptr, 0, static_cast<int64_t>(buffer.values.size()), &timings);
  if (!reduction.ok) {
    return arrow::Status::Invalid("GPU aggregate failed: " + reduction.error);
  }

  buffer.sum += reduction.sum;
  buffer.count += reduction.valid_count;
  buffer.values.clear();
  stats_->gpu_used = true;
  ++stats_->gpu_batches;
  stats_->gpu_transfer_ms += timings.transfer_ms;
  stats_->gpu_kernel_ms += timings.kernel_ms;
  return arrow::Status::OK();
}

arrow::Status AggregateOperator::FinishGpu() {
  if (gpu_mode_ != 1)
    return arrow::Status::OK();

  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto& aggregate = *aggregates_[i];
    if (aggregate.function == "sum" || aggregate.function == "avg") {
      ARROW_RETURN_NOT_OK(FlushGpuBuffer(i));
    }
  }

  if (groups_.empty())
    AddGroup(Group{});

  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto& aggregate = *aggregates_[i];
    if (aggregate.function != "sum" && aggregate.function != "avg")
      continue;
    const GpuBuffer& buffer = gpu_buffers_[i];
    AggState& state = State(i, 0);
    if (buffer.count == 0)
      continue;
    state.double_sum += buffer.sum;
    if (aggregate.function == "avg")
      state.count += buffer.count;
    state.has_value = true;
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> AggregateOperator::BuildResult() {
  if (groups_.empty() && group_by_.empty())
    AddGroup(Group{});
  if (groups_.empty())
    return MakeEmptyBatchImpl(output_schema_);

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(group_by_.size() + aggregates_.size());

  for (size_t i = 0; i < group_by_.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto name, ResolveColumnName(*child_->schema(), *group_by_[i]));
    int index = child_->schema()->GetFieldIndex(name);
    if (index < 0)
      return arrow::Status::Invalid("group column not found");
    ARROW_ASSIGN_OR_RAISE(auto builder, arrow::MakeBuilder(child_->schema()->field(index)->type()));
    for (const auto& group : groups_) {
      ARROW_RETURN_NOT_OK(builder->AppendScalar(*group.keys[i]));
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder->Finish());
    columns.push_back(std::move(array));
  }

  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto& aggregate = *aggregates_[i];
    auto out_type = output_schema_->field(group_by_.size() + i)->type();
    std::unique_ptr<arrow::ArrayBuilder> builder;
    if (aggregate.function == "count") {
      builder = std::make_unique<arrow::Int64Builder>();
    } else if (aggregate.function == "avg") {
      builder = std::make_unique<arrow::DoubleBuilder>();
    } else if (aggregate.function == "sum") {
      if (out_type->id() == arrow::Type::DOUBLE) {
        builder = std::make_unique<arrow::DoubleBuilder>();
      } else {
        builder = std::make_unique<arrow::Int64Builder>();
      }
    } else {
      ARROW_ASSIGN_OR_RAISE(builder, arrow::MakeBuilder(out_type));
    }

    for (size_t group_index = 0; group_index < groups_.size(); ++group_index) {
      const AggState& state = State(i, group_index);
      if (aggregate.function == "count") {
        ARROW_RETURN_NOT_OK(checked_cast<arrow::Int64Builder&>(*builder).Append(state.count));
      } else if (aggregate.function == "sum") {
        if (!state.has_value) {
          ARROW_RETURN_NOT_OK(builder->AppendNull());
        } else if (out_type->id() == arrow::Type::DOUBLE) {
          ARROW_RETURN_NOT_OK(
              checked_cast<arrow::DoubleBuilder&>(*builder).Append(state.double_sum));
        } else {
          if (state.int_sum < std::numeric_limits<int64_t>::min() ||
              state.int_sum > std::numeric_limits<int64_t>::max()) {
            return arrow::Status::Invalid("SUM overflowed the int64 range");
          }
          ARROW_RETURN_NOT_OK(checked_cast<arrow::Int64Builder&>(*builder).Append(
              static_cast<int64_t>(state.int_sum)));
        }
      } else if (aggregate.function == "avg") {
        if (state.count == 0) {
          ARROW_RETURN_NOT_OK(builder->AppendNull());
        } else {
          ARROW_RETURN_NOT_OK(checked_cast<arrow::DoubleBuilder&>(*builder).Append(
              state.double_sum / static_cast<double>(state.count)));
        }
      } else if (aggregate.function == "min" || aggregate.function == "max") {
        if (!state.has_value) {
          ARROW_RETURN_NOT_OK(builder->AppendNull());
        } else if (state.extreme_kind == ExtremeKind::kInt64) {
          const int64_t value = state.int_extreme;
          switch (out_type->id()) {
          case arrow::Type::INT8:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::Int8Builder&>(*builder).Append(static_cast<int8_t>(value)));
            break;
          case arrow::Type::INT16:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::Int16Builder&>(*builder).Append(static_cast<int16_t>(value)));
            break;
          case arrow::Type::INT32:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::Int32Builder&>(*builder).Append(static_cast<int32_t>(value)));
            break;
          case arrow::Type::INT64:
            ARROW_RETURN_NOT_OK(checked_cast<arrow::Int64Builder&>(*builder).Append(value));
            break;
          case arrow::Type::UINT8:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::UInt8Builder&>(*builder).Append(static_cast<uint8_t>(value)));
            break;
          case arrow::Type::UINT16:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::UInt16Builder&>(*builder).Append(static_cast<uint16_t>(value)));
            break;
          case arrow::Type::UINT32:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::UInt32Builder&>(*builder).Append(static_cast<uint32_t>(value)));
            break;
          case arrow::Type::UINT64:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::UInt64Builder&>(*builder).Append(static_cast<uint64_t>(value)));
            break;
          case arrow::Type::DATE32:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::Date32Builder&>(*builder).Append(static_cast<int32_t>(value)));
            break;
          default:
            return arrow::Status::Invalid("MIN/MAX cannot build output type " +
                                          out_type->ToString());
          }
        } else if (state.extreme_kind == ExtremeKind::kDouble) {
          if (out_type->id() == arrow::Type::FLOAT) {
            ARROW_RETURN_NOT_OK(checked_cast<arrow::FloatBuilder&>(*builder).Append(
                static_cast<float>(state.double_extreme)));
          } else if (out_type->id() == arrow::Type::DOUBLE) {
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::DoubleBuilder&>(*builder).Append(state.double_extreme));
          } else {
            return arrow::Status::Invalid("MIN/MAX cannot build output type " +
                                          out_type->ToString());
          }
        } else if (state.extreme_kind == ExtremeKind::kBool) {
          ARROW_RETURN_NOT_OK(
              checked_cast<arrow::BooleanBuilder&>(*builder).Append(state.bool_extreme));
        } else if (state.extreme_kind == ExtremeKind::kString) {
          switch (out_type->id()) {
          case arrow::Type::STRING:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::StringBuilder&>(*builder).Append(*state.string_extreme));
            break;
          case arrow::Type::LARGE_STRING:
            ARROW_RETURN_NOT_OK(
                checked_cast<arrow::LargeStringBuilder&>(*builder).Append(*state.string_extreme));
            break;
          case arrow::Type::BINARY:
            ARROW_RETURN_NOT_OK(checked_cast<arrow::BinaryBuilder&>(*builder).Append(
                reinterpret_cast<const uint8_t*>(state.string_extreme->data()),
                static_cast<int32_t>(state.string_extreme->size())));
            break;
          default:
            return arrow::Status::Invalid("MIN/MAX cannot build output type " +
                                          out_type->ToString());
          }
        } else {
          return arrow::Status::Invalid("MIN/MAX has no accumulated value");
        }
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder->Finish());
    columns.push_back(std::move(array));
  }

  return arrow::RecordBatch::Make(output_schema_, static_cast<int64_t>(groups_.size()),
                                  std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> AggregateOperator::Next() {
  if (done_)
    return nullptr;
  done_ = true;

  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
    if (!batch)
      break;
    ARROW_RETURN_NOT_OK(Consume(batch));
  }
  ARROW_RETURN_NOT_OK(FinishGpu());
  ARROW_ASSIGN_OR_RAISE(result_, BuildResult());
  return result_;
}

// ---------------------------------------------------------------------------
// Hash join
// ---------------------------------------------------------------------------

JoinOperator::JoinOperator(std::shared_ptr<Operator> left, std::shared_ptr<Operator> right,
                           std::vector<JoinCondition> conditions, bool build_left, bool left_outer)
    : left_(std::move(left)), right_(std::move(right)), conditions_(std::move(conditions)),
      build_left_(left_outer ? false : build_left), left_outer_(left_outer) {
  left_schema_ = left_->schema();
  right_schema_ = right_->schema();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int i = 0; i < left_schema_->num_fields(); ++i)
    fields.push_back(left_schema_->field(i));
  for (int i = 0; i < right_schema_->num_fields(); ++i)
    fields.push_back(right_schema_->field(i));
  output_schema_ = arrow::schema(std::move(fields));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> JoinOperator::Build() {
  built_ = true;
  Operator& build_operator = build_left_ ? *left_ : *right_;
  ARROW_ASSIGN_OR_RAISE(build_batch_, MaterializeOperator(build_operator));

  for (const auto& condition : conditions_) {
    const auto& expr = build_left_ ? condition.left : condition.right;
    ARROW_ASSIGN_OR_RAISE(auto key, EvaluateExpressionArray(*expr, *build_batch_));
    build_keys_.push_back(std::move(key));
  }

  // A single integer or string key uses the typed tables below. They avoid
  // building a serialized key string for every probe row; the build side is
  // materialized once, so the tables never need to grow or migrate.
  if (conditions_.size() == 1) {
    const auto& key_type = *build_keys_[0]->type();
    if (IsIntegerKeyType(key_type)) {
      key_mode_ = JoinKeyMode::kInteger;
    } else if (IsStringKeyType(key_type)) {
      key_mode_ = JoinKeyMode::kString;
    }
  }
  if (key_mode_ != JoinKeyMode::kGeneric) {
    ARROW_RETURN_NOT_OK(BuildKeyTable());
    return build_batch_;
  }

  for (int64_t row = 0; row < build_batch_->num_rows(); ++row) {
    std::string key;
    key.reserve(build_keys_.size() * 10);
    bool has_null = false;
    for (const auto& array : build_keys_) {
      if (array->IsNull(row)) {
        has_null = true;
        break;
      }
      ARROW_RETURN_NOT_OK(AppendArrayKey(*array, row, key));
    }
    // SQL equality never matches NULL, unlike grouping where NULL is a group.
    if (has_null)
      continue;
    hash_[key].push_back(row);
  }
  return build_batch_;
}

arrow::Status JoinOperator::BuildKeyTable() {
  next_row_.assign(static_cast<size_t>(build_batch_->num_rows()), -1);
  const size_t capacity = SlotCapacityFor(build_batch_->num_rows());
  const size_t mask = capacity - 1;
  const auto& array = *build_keys_[0];

  if (key_mode_ == JoinKeyMode::kInteger) {
    integer_slots_.assign(capacity, IntegerSlot{});
    for (int64_t row = 0; row < build_batch_->num_rows(); ++row) {
      // SQL equality never matches NULL; skip it like the serialized path.
      if (array.IsNull(row))
        continue;
      const uint64_t key = IntegerKeyAt(array, row);
      size_t index = MixKey(key) & mask;
      while (integer_slots_[index].used && integer_slots_[index].key != key)
        index = (index + 1) & mask;
      IntegerSlot& slot = integer_slots_[index];
      if (slot.used) {
        next_row_[static_cast<size_t>(slot.last)] = row;
        slot.last = row;
      } else {
        slot.used = true;
        slot.key = key;
        slot.head = row;
        slot.last = row;
      }
    }
    return arrow::Status::OK();
  }

  string_slots_.assign(capacity, StringSlot{});
  for (int64_t row = 0; row < build_batch_->num_rows(); ++row) {
    if (array.IsNull(row))
      continue;
    const std::string_view value = ArrayStringView(array, row);
    const size_t hash = std::hash<std::string_view>{}(value);
    size_t index = MixKey(static_cast<uint64_t>(hash)) & mask;
    while (string_slots_[index].used &&
           !(string_slots_[index].hash == hash && string_slots_[index].value == value))
      index = (index + 1) & mask;
    StringSlot& slot = string_slots_[index];
    if (slot.used) {
      next_row_[static_cast<size_t>(slot.last)] = row;
      slot.last = row;
    } else {
      slot.used = true;
      // The view points into the retained build-key array, which lives as
      // long as this operator.
      slot.value = value;
      slot.hash = hash;
      slot.head = row;
      slot.last = row;
    }
  }
  return arrow::Status::OK();
}

arrow::Status JoinOperator::MatchTyped(const std::shared_ptr<arrow::Array>& probe_key,
                                       bool preserve_unmatched, int64_t unmatched_build_row,
                                       arrow::Int64Builder& probe_rows,
                                       arrow::Int64Builder& build_rows) {
  const auto& array = *probe_key;

  if (key_mode_ == JoinKeyMode::kInteger) {
    const bool table_empty = integer_slots_.empty();
    const size_t mask = table_empty ? 0 : integer_slots_.size() - 1;
    for (int64_t row = 0; row < array.length(); ++row) {
      bool found = false;
      if (!table_empty && !array.IsNull(row)) {
        const uint64_t key = IntegerKeyAt(array, row);
        size_t index = MixKey(key) & mask;
        while (integer_slots_[index].used && integer_slots_[index].key != key)
          index = (index + 1) & mask;
        found = integer_slots_[index].used;
        if (found) {
          for (int64_t build_row = integer_slots_[index].head; build_row != -1;
               build_row = next_row_[static_cast<size_t>(build_row)]) {
            ARROW_RETURN_NOT_OK(probe_rows.Append(row));
            ARROW_RETURN_NOT_OK(build_rows.Append(build_row));
          }
        }
      }
      if (!found && preserve_unmatched) {
        ARROW_RETURN_NOT_OK(probe_rows.Append(row));
        ARROW_RETURN_NOT_OK(build_rows.Append(unmatched_build_row));
      }
    }
    return arrow::Status::OK();
  }

  const bool table_empty = string_slots_.empty();
  const size_t mask = table_empty ? 0 : string_slots_.size() - 1;
  for (int64_t row = 0; row < array.length(); ++row) {
    bool found = false;
    if (!table_empty && !array.IsNull(row)) {
      const std::string_view value = ArrayStringView(array, row);
      const size_t hash = std::hash<std::string_view>{}(value);
      size_t index = MixKey(static_cast<uint64_t>(hash)) & mask;
      while (string_slots_[index].used &&
             !(string_slots_[index].hash == hash && string_slots_[index].value == value))
        index = (index + 1) & mask;
      found = string_slots_[index].used;
      if (found) {
        for (int64_t build_row = string_slots_[index].head; build_row != -1;
             build_row = next_row_[static_cast<size_t>(build_row)]) {
          ARROW_RETURN_NOT_OK(probe_rows.Append(row));
          ARROW_RETURN_NOT_OK(build_rows.Append(build_row));
        }
      }
    }
    if (!found && preserve_unmatched) {
      ARROW_RETURN_NOT_OK(probe_rows.Append(row));
      ARROW_RETURN_NOT_OK(build_rows.Append(unmatched_build_row));
    }
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
JoinOperator::Probe(const std::shared_ptr<arrow::RecordBatch>& probe_batch,
                    const std::vector<std::shared_ptr<arrow::Array>>& probe_keys) {
  arrow::Int64Builder probe_indices;
  arrow::Int64Builder build_indices;
  if (key_mode_ != JoinKeyMode::kGeneric) {
    ARROW_RETURN_NOT_OK(MatchTyped(probe_keys[0], /*preserve_unmatched=*/false,
                                   /*unmatched_build_row=*/-1, probe_indices, build_indices));
  } else {
    for (int64_t row = 0; row < probe_batch->num_rows(); ++row) {
      std::string key;
      key.reserve(probe_keys.size() * 10);
      bool has_null = false;
      for (const auto& array : probe_keys) {
        if (array->IsNull(row)) {
          has_null = true;
          break;
        }
        ARROW_RETURN_NOT_OK(AppendArrayKey(*array, row, key));
      }
      if (has_null)
        continue;
      auto found = hash_.find(key);
      if (found == hash_.end())
        continue;
      for (int64_t build_row : found->second) {
        ARROW_RETURN_NOT_OK(probe_indices.Append(row));
        ARROW_RETURN_NOT_OK(build_indices.Append(build_row));
      }
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto probe_index_array, probe_indices.Finish());
  ARROW_ASSIGN_OR_RAISE(auto build_index_array, build_indices.Finish());
  const int64_t count = probe_index_array->length();
  if (count == 0)
    return nullptr;

  const auto& probe_batch_ref = probe_batch;
  const auto& build_batch_ref = build_batch_;
  const auto& left_batch = build_left_ ? build_batch_ref : probe_batch_ref;
  const auto& right_batch = build_left_ ? probe_batch_ref : build_batch_ref;
  const auto& left_indices = build_left_ ? build_index_array : probe_index_array;
  const auto& right_indices = build_left_ ? probe_index_array : build_index_array;

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(output_schema_->num_fields());
  for (int i = 0; i < left_schema_->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto taken, arrow::compute::Take(*left_batch->column(i), *left_indices));
    columns.push_back(std::move(taken));
  }
  for (int i = 0; i < right_schema_->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto taken,
                          arrow::compute::Take(*right_batch->column(i), *right_indices));
    columns.push_back(std::move(taken));
  }
  return arrow::RecordBatch::Make(output_schema_, count, std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> JoinOperator::MaterializeLeftOuter() {
  ARROW_ASSIGN_OR_RAISE(auto left_batch, MaterializeOperator(*left_));
  if (left_batch->num_rows() == 0)
    return MakeEmptyBatchImpl(output_schema_);

  std::vector<std::shared_ptr<arrow::Array>> probe_keys;
  probe_keys.reserve(conditions_.size());
  for (size_t i = 0; i < conditions_.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto key, EvaluateExpressionArray(*conditions_[i].left, *left_batch));
    if (!key->type()->Equals(build_keys_[i]->type())) {
      return arrow::Status::Invalid("join key types must match exactly; left is " +
                                    key->type()->ToString() + ", right is " +
                                    build_keys_[i]->type()->ToString());
    }
    probe_keys.push_back(std::move(key));
  }

  arrow::Int64Builder left_indices;
  arrow::Int64Builder right_indices;
  if (key_mode_ != JoinKeyMode::kGeneric) {
    ARROW_RETURN_NOT_OK(MatchTyped(probe_keys[0], /*preserve_unmatched=*/true,
                                   build_batch_->num_rows(), left_indices, right_indices));
  } else {
    for (int64_t row = 0; row < left_batch->num_rows(); ++row) {
      std::string key;
      key.reserve(probe_keys.size() * 10);
      bool has_null = false;
      for (const auto& array : probe_keys) {
        if (array->IsNull(row)) {
          has_null = true;
          break;
        }
        ARROW_RETURN_NOT_OK(AppendArrayKey(*array, row, key));
      }
      const auto found = has_null ? hash_.end() : hash_.find(key);
      if (found == hash_.end()) {
        // Unmatched preserved-side row: point at the appended null row.
        ARROW_RETURN_NOT_OK(left_indices.Append(row));
        ARROW_RETURN_NOT_OK(right_indices.Append(build_batch_->num_rows()));
      } else {
        for (int64_t build_row : found->second) {
          ARROW_RETURN_NOT_OK(left_indices.Append(row));
          ARROW_RETURN_NOT_OK(right_indices.Append(build_row));
        }
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto left_index_array, left_indices.Finish());
  ARROW_ASSIGN_OR_RAISE(auto right_index_array, right_indices.Finish());
  const int64_t count = left_index_array->length();

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(output_schema_->num_fields());
  for (int i = 0; i < left_schema_->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto taken,
                          arrow::compute::Take(*left_batch->column(i), *left_index_array));
    columns.push_back(std::move(taken));
  }
  for (int i = 0; i < right_schema_->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto null_row,
                          arrow::MakeArrayOfNull(right_schema_->field(i)->type(), 1));
    ARROW_ASSIGN_OR_RAISE(auto extended, arrow::Concatenate({build_batch_->column(i), null_row}));
    ARROW_ASSIGN_OR_RAISE(auto taken, arrow::compute::Take(*extended, *right_index_array));
    columns.push_back(std::move(taken));
  }
  return arrow::RecordBatch::Make(output_schema_, count, std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> JoinOperator::Next() {
  if (done_)
    return nullptr;
  if (!built_) {
    ARROW_RETURN_NOT_OK(Build());
  }
  if (left_outer_) {
    done_ = true;
    ARROW_ASSIGN_OR_RAISE(auto result, MaterializeLeftOuter());
    return result;
  }

  Operator& probe_operator = build_left_ ? *right_ : *left_;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto probe_batch, probe_operator.Next());
    if (!probe_batch) {
      done_ = true;
      return nullptr;
    }
    std::vector<std::shared_ptr<arrow::Array>> probe_keys;
    probe_keys.reserve(conditions_.size());
    for (size_t i = 0; i < conditions_.size(); ++i) {
      const auto& expr = build_left_ ? conditions_[i].right : conditions_[i].left;
      ARROW_ASSIGN_OR_RAISE(auto key, EvaluateExpressionArray(*expr, *probe_batch));
      if (!key->type()->Equals(build_keys_[i]->type())) {
        return arrow::Status::Invalid("join key types must match exactly; left is " +
                                      build_keys_[i]->type()->ToString() + ", right is " +
                                      key->type()->ToString());
      }
      probe_keys.push_back(std::move(key));
    }

    ARROW_ASSIGN_OR_RAISE(auto result, Probe(probe_batch, probe_keys));
    if (result && result->num_rows() > 0)
      return result;
  }
}

} // namespace qe
