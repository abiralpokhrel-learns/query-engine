#include "engine/operators.h"

#include <arrow/array/concatenate.h>
#include <arrow/array/util.h>
#include <arrow/compute/api.h>
#include <arrow/table.h>
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

namespace qe {
namespace {

using arrow::internal::checked_cast;

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

arrow::Result<int64_t> ScalarAsInt64(const std::shared_ptr<arrow::Scalar>& scalar) {
  switch (scalar->type->id()) {
  case arrow::Type::INT8:
    return checked_cast<const arrow::Int8Scalar&>(*scalar).value;
  case arrow::Type::INT16:
    return checked_cast<const arrow::Int16Scalar&>(*scalar).value;
  case arrow::Type::INT32:
    return checked_cast<const arrow::Int32Scalar&>(*scalar).value;
  case arrow::Type::INT64:
    return checked_cast<const arrow::Int64Scalar&>(*scalar).value;
  case arrow::Type::UINT8:
    return checked_cast<const arrow::UInt8Scalar&>(*scalar).value;
  case arrow::Type::UINT16:
    return checked_cast<const arrow::UInt16Scalar&>(*scalar).value;
  case arrow::Type::UINT32:
    return checked_cast<const arrow::UInt32Scalar&>(*scalar).value;
  case arrow::Type::UINT64:
    return static_cast<int64_t>(checked_cast<const arrow::UInt64Scalar&>(*scalar).value);
  default:
    return arrow::Status::Invalid("aggregate requires an integer column");
  }
}

arrow::Result<double> ScalarAsDouble(const std::shared_ptr<arrow::Scalar>& scalar) {
  switch (scalar->type->id()) {
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64: {
    ARROW_ASSIGN_OR_RAISE(auto value, ScalarAsInt64(scalar));
    return static_cast<double>(value);
  }
  case arrow::Type::FLOAT:
    return checked_cast<const arrow::FloatScalar&>(*scalar).value;
  case arrow::Type::DOUBLE:
    return checked_cast<const arrow::DoubleScalar&>(*scalar).value;
  default:
    return arrow::Status::Invalid("aggregate requires a numeric column");
  }
}

arrow::Result<int64_t> ArrayAsInt64(const arrow::Array& array, int64_t row) {
  switch (array.type_id()) {
  case arrow::Type::INT8:
    return checked_cast<const arrow::Int8Array&>(array).Value(row);
  case arrow::Type::INT16:
    return checked_cast<const arrow::Int16Array&>(array).Value(row);
  case arrow::Type::INT32:
    return checked_cast<const arrow::Int32Array&>(array).Value(row);
  case arrow::Type::INT64:
    return checked_cast<const arrow::Int64Array&>(array).Value(row);
  case arrow::Type::UINT8:
    return checked_cast<const arrow::UInt8Array&>(array).Value(row);
  case arrow::Type::UINT16:
    return checked_cast<const arrow::UInt16Array&>(array).Value(row);
  case arrow::Type::UINT32:
    return checked_cast<const arrow::UInt32Array&>(array).Value(row);
  case arrow::Type::UINT64:
    return static_cast<int64_t>(checked_cast<const arrow::UInt64Array&>(array).Value(row));
  default:
    return arrow::Status::Invalid("aggregate requires an integer column");
  }
}

arrow::Result<double> ArrayAsDouble(const arrow::Array& array, int64_t row) {
  switch (array.type_id()) {
  case arrow::Type::FLOAT:
    return checked_cast<const arrow::FloatArray&>(array).Value(row);
  case arrow::Type::DOUBLE:
    return checked_cast<const arrow::DoubleArray&>(array).Value(row);
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64: {
    ARROW_ASSIGN_OR_RAISE(auto value, ArrayAsInt64(array, row));
    return static_cast<double>(value);
  }
  default:
    return arrow::Status::Invalid("aggregate requires a numeric column");
  }
}

std::string_view ScalarStringView(const arrow::Scalar& scalar) {
  const auto id = scalar.type->id();
  if (id == arrow::Type::STRING) {
    const auto& buffer = checked_cast<const arrow::StringScalar&>(scalar).value;
    return std::string_view(reinterpret_cast<const char*>(buffer->data()), buffer->size());
  }
  if (id == arrow::Type::LARGE_STRING) {
    const auto& buffer = checked_cast<const arrow::LargeStringScalar&>(scalar).value;
    return std::string_view(reinterpret_cast<const char*>(buffer->data()), buffer->size());
  }
  const auto& buffer = checked_cast<const arrow::BinaryScalar&>(scalar).value;
  return std::string_view(reinterpret_cast<const char*>(buffer->data()), buffer->size());
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

// Compares one array value with a scalar without materializing an Arrow scalar
// per row. Used by MIN/MAX so the inner loop stays allocation-free.
arrow::Result<int> CompareArrayToScalar(const arrow::Array& array, int64_t row,
                                        const arrow::Scalar& scalar) {
  const auto type_id = array.type_id();
  if (arrow::is_integer(type_id)) {
    ARROW_ASSIGN_OR_RAISE(auto left, ArrayAsInt64(array, row));
    ARROW_ASSIGN_OR_RAISE(auto right, ScalarAsInt64(scalar.GetSharedPtr()));
    return left < right ? -1 : (left > right ? 1 : 0);
  }
  if (type_id == arrow::Type::FLOAT || type_id == arrow::Type::DOUBLE) {
    ARROW_ASSIGN_OR_RAISE(auto left, ArrayAsDouble(array, row));
    ARROW_ASSIGN_OR_RAISE(auto right, ScalarAsDouble(scalar.GetSharedPtr()));
    return left < right ? -1 : (left > right ? 1 : 0);
  }
  if (type_id == arrow::Type::DATE32) {
    const auto left = checked_cast<const arrow::Date32Array&>(array).Value(row);
    const auto right = checked_cast<const arrow::Date32Scalar&>(scalar).value;
    return left < right ? -1 : (left > right ? 1 : 0);
  }
  if (type_id == arrow::Type::BOOL) {
    const auto left = checked_cast<const arrow::BooleanArray&>(array).Value(row);
    const auto right = checked_cast<const arrow::BooleanScalar&>(scalar).value;
    return left == right ? 0 : (left ? 1 : -1);
  }
  if (type_id == arrow::Type::STRING || type_id == arrow::Type::LARGE_STRING ||
      type_id == arrow::Type::BINARY) {
    const auto left = ArrayStringView(array, row);
    const auto right = ScalarStringView(scalar);
    const int cmp = left.compare(right);
    return cmp < 0 ? -1 : (cmp > 0 ? 1 : 0);
  }
  return arrow::Status::Invalid("MIN/MAX does not support type " + array.type()->ToString());
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

uint64_t MixKey(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
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

arrow::Status
AggregateOperator::UpdateGroup(Group& group, int64_t row,
                               const std::vector<std::shared_ptr<arrow::Array>>& argument_arrays) {
  for (size_t i = 0; i < aggregates_.size(); ++i) {
    const auto& aggregate = *aggregates_[i];
    AggState& state = group.states[i];
    if (aggregate.star) {
      ++state.count;
      continue;
    }
    const auto& arguments = *argument_arrays[i];
    if (arguments.IsNull(row))
      continue;

    if (aggregate.function == "count") {
      ++state.count;
    } else if (aggregate.function == "sum") {
      if (output_schema_->field(group_by_.size() + i)->type()->id() == arrow::Type::DOUBLE) {
        ARROW_ASSIGN_OR_RAISE(auto value, ArrayAsDouble(arguments, row));
        state.double_sum += value;
      } else {
        ARROW_ASSIGN_OR_RAISE(auto value, ArrayAsInt64(arguments, row));
        state.int_sum += value;
      }
      state.has_value = true;
    } else if (aggregate.function == "avg") {
      ARROW_ASSIGN_OR_RAISE(auto value, ArrayAsDouble(arguments, row));
      state.double_sum += value;
      ++state.count;
      state.has_value = true;
    } else if (aggregate.function == "min" || aggregate.function == "max") {
      if (!state.has_value) {
        ARROW_ASSIGN_OR_RAISE(state.extreme, arguments.GetScalar(row));
        state.has_value = true;
      } else {
        ARROW_ASSIGN_OR_RAISE(int cmp, CompareArrayToScalar(arguments, row, *state.extreme));
        const bool better = aggregate.function == "min" ? cmp < 0 : cmp > 0;
        if (better) {
          ARROW_ASSIGN_OR_RAISE(state.extreme, arguments.GetScalar(row));
        }
      }
    }
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
    if (groups_.empty()) {
      Group group;
      group.states.resize(aggregates_.size());
      groups_.push_back(std::move(group));
    }
    for (auto& state : groups_[0].states)
      state.count += batch->num_rows();
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

  if (group_by_.size() == 1) {
    if (group_key_mode_ == 0) {
      group_key_mode_ = IsIntegerKeyType(*group_arrays[0]->type()) ? 1 : -1;
    }
  } else {
    group_key_mode_ = -1;
  }

  if (group_key_mode_ == 1) {
    const auto& key_array = *group_arrays[0];
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      const bool is_null = key_array.IsNull(row);
      const uint64_t key = is_null ? 0 : IntegerKeyAt(key_array, row);
      bool inserted = false;
      const size_t slot = FlatFindOrInsert(key, is_null, &inserted);
      size_t group_index;
      if (inserted) {
        group_index = groups_.size();
        Group group;
        group.keys.reserve(1);
        ARROW_ASSIGN_OR_RAISE(auto scalar, key_array.GetScalar(row));
        group.keys.push_back(std::move(scalar));
        group.states.resize(aggregates_.size());
        groups_.push_back(std::move(group));
        flat_slots_[slot].group = group_index;
      } else {
        group_index = flat_slots_[slot].group;
      }
      ARROW_RETURN_NOT_OK(UpdateGroup(groups_[group_index], row, argument_arrays));
    }
    return arrow::Status::OK();
  }

  std::string key;
  key.reserve(group_arrays.size() * 10);
  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    key.clear();
    for (const auto& array : group_arrays) {
      ARROW_RETURN_NOT_OK(AppendArrayKey(*array, row, key));
    }

    size_t group_index = 0;
    if (group_by_.empty()) {
      if (groups_.empty()) {
        Group group;
        group.states.resize(aggregates_.size());
        groups_.push_back(std::move(group));
        group_index_[key] = 0;
      }
    } else {
      auto found = group_index_.find(key);
      if (found == group_index_.end()) {
        group_index = groups_.size();
        group_index_.emplace(std::move(key), group_index);
        Group group;
        group.keys.reserve(group_arrays.size());
        // Materialize the key values only for the first row of each group.
        for (const auto& array : group_arrays) {
          ARROW_ASSIGN_OR_RAISE(auto scalar, array->GetScalar(row));
          group.keys.push_back(std::move(scalar));
        }
        group.states.resize(aggregates_.size());
        groups_.push_back(std::move(group));
      } else {
        group_index = found->second;
      }
    }

    ARROW_RETURN_NOT_OK(UpdateGroup(groups_[group_index], row, argument_arrays));
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> AggregateOperator::BuildResult() {
  if (groups_.empty() && group_by_.empty()) {
    Group group;
    group.states.resize(aggregates_.size());
    groups_.push_back(std::move(group));
  }
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

    for (const auto& group : groups_) {
      const AggState& state = group.states[i];
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
        } else {
          ARROW_RETURN_NOT_OK(builder->AppendScalar(*state.extreme));
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

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
JoinOperator::Probe(const std::shared_ptr<arrow::RecordBatch>& probe_batch,
                    const std::vector<std::shared_ptr<arrow::Array>>& probe_keys) {
  arrow::Int64Builder probe_indices;
  arrow::Int64Builder build_indices;
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
