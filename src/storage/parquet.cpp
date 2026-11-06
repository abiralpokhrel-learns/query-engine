#include "storage/parquet.h"

#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <parquet/arrow/reader.h>
#include <parquet/statistics.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string_view>

namespace qe {
namespace {

std::string ToLower(std::string_view text) {
  std::string result(text);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

bool IsSupportedType(const arrow::DataType& type) {
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
    case arrow::Type::DATE32:
      return true;
    default:
      return false;
  }
}

arrow::Status ValidateFlatSchema(const arrow::Schema& schema) {
  for (int i = 0; i < schema.num_fields(); ++i) {
    const auto& field = schema.field(i);
    if (!IsSupportedType(*field->type())) {
      return arrow::Status::NotImplemented(
          "column '" + field->name() + "' has unsupported type " + field->type()->ToString() +
          "; only flat bool, integer, floating point, string, and date32 columns are supported");
    }
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Scalar>> CastScalar(const std::shared_ptr<arrow::Scalar>& scalar,
                                                         const std::shared_ptr<arrow::DataType>& type) {
  if (scalar->type->Equals(type)) return scalar;
  return scalar->CastTo(type);
}

// Converts a SQL literal to an Arrow scalar. Dates may be written as ISO strings.
arrow::Result<std::shared_ptr<arrow::Scalar>> LiteralToScalar(const sql::Literal& literal,
                                                             const arrow::DataType& type) {
  switch (literal.type) {
    case sql::LiteralType::Null:
      return arrow::MakeNullScalar(type.GetSharedPtr());
    case sql::LiteralType::Integer: {
      if (type.id() == arrow::Type::DATE32) {
        return arrow::MakeScalar(arrow::date32(), literal.integer);
      }
      auto scalar = arrow::MakeScalar(literal.integer);
      return CastScalar(scalar, type.GetSharedPtr());
    }
    case sql::LiteralType::Double: {
      auto scalar = arrow::MakeScalar(literal.real);
      return CastScalar(scalar, type.GetSharedPtr());
    }
    case sql::LiteralType::String: {
      if (type.id() == arrow::Type::DATE32) {
        // Parse YYYY-MM-DD as days since the Unix epoch.
        if (literal.string.size() != 10 || literal.string[4] != '-' ||
            literal.string[7] != '-') {
          return arrow::Status::Invalid("date literal must be YYYY-MM-DD: '" + literal.string + "'");
        }
        int year = 0, month = 0, day = 0;
        try {
          year = std::stoi(literal.string.substr(0, 4));
          month = std::stoi(literal.string.substr(5, 2));
          day = std::stoi(literal.string.substr(8, 2));
        } catch (const std::exception&) {
          return arrow::Status::Invalid("invalid date literal: '" + literal.string + "'");
        }
        if (month < 1 || month > 12 || day < 1 || day > 31) {
          return arrow::Status::Invalid("invalid date literal: '" + literal.string + "'");
        }
        // Howard Hinnant's days_from_civil algorithm.
        year -= month <= 2;
        const int era = (year >= 0 ? year : year - 399) / 400;
        const unsigned yoe = static_cast<unsigned>(year - era * 400);
        const unsigned doy =
            (153 * static_cast<unsigned>(month > 2 ? month - 3 : month + 9) + 2) / 5 +
            static_cast<unsigned>(day) - 1;
        const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        const int64_t days = static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
        return arrow::MakeScalar(arrow::date32(), static_cast<int32_t>(days));
      }
      auto scalar = arrow::MakeScalar(literal.string);
      if (type.id() == arrow::Type::BINARY) {
        return CastScalar(scalar, type.GetSharedPtr());
      }
      return CastScalar(scalar, type.GetSharedPtr());
    }
    case sql::LiteralType::Boolean: {
      auto scalar = arrow::MakeScalar(literal.boolean);
      return CastScalar(scalar, type.GetSharedPtr());
    }
  }
  return arrow::Status::Invalid("unsupported literal");
}

// Produces min/max and literal scalars in one comparable type. Returns an error
// when the comparison cannot be proven safely; callers then simply keep the row
// group.
arrow::Result<std::pair<std::shared_ptr<arrow::Scalar>, std::shared_ptr<arrow::Scalar>>>
MakeComparable(const std::shared_ptr<arrow::Scalar>& value, const sql::Literal& literal) {
  std::shared_ptr<arrow::DataType> target;
  if (literal.type == sql::LiteralType::Double) {
    if (arrow::is_integer(value->type->id()) || value->type->id() == arrow::Type::FLOAT ||
        value->type->id() == arrow::Type::DOUBLE) {
      target = arrow::float64();
    }
  } else if (literal.type == sql::LiteralType::Integer) {
    if (value->type->id() == arrow::Type::FLOAT || value->type->id() == arrow::Type::DOUBLE) {
      target = arrow::float64();
    } else if (arrow::is_integer(value->type->id())) {
      target = arrow::int64();
    }
  } else if (literal.type == sql::LiteralType::String) {
    if (value->type->id() == arrow::Type::STRING || value->type->id() == arrow::Type::LARGE_STRING ||
        value->type->id() == arrow::Type::BINARY) {
      target = value->type;
    } else if (value->type->id() == arrow::Type::DATE32) {
      target = arrow::date32();
    }
  } else if (literal.type == sql::LiteralType::Boolean) {
    if (value->type->id() == arrow::Type::BOOL) target = arrow::boolean();
  }
  if (!target) {
    return arrow::Status::Invalid("cannot compare " + value->type->ToString() + " to this literal");
  }
  ARROW_ASSIGN_OR_RAISE(auto value_cast, CastScalar(value, target));
  ARROW_ASSIGN_OR_RAISE(auto literal_cast, LiteralToScalar(literal, *target));
  return std::make_pair(std::move(value_cast), std::move(literal_cast));
}

arrow::Result<bool> ScalarCompare(const std::string& function,
                                  const std::shared_ptr<arrow::Scalar>& left,
                                  const std::shared_ptr<arrow::Scalar>& right) {
  if (!left->is_valid || !right->is_valid) {
    return arrow::Status::Invalid("comparison with null");
  }
  ARROW_ASSIGN_OR_RAISE(auto datum, arrow::compute::CallFunction(function, {left, right}));
  if (!datum.is_scalar() || !datum.scalar()->is_valid) {
    return arrow::Status::Invalid("comparison did not produce a scalar");
  }
  return arrow::internal::checked_cast<const arrow::BooleanScalar&>(*datum.scalar()).value;
}

// true when <value> <op> <literal> is provably false for every value in
// [min, max].
arrow::Result<bool> RangeProvesNoMatch(const std::shared_ptr<arrow::Scalar>& min,
                                       const std::shared_ptr<arrow::Scalar>& max,
                                       const sql::Literal& literal, sql::BinaryOp op) {
  if (literal.type == sql::LiteralType::Null) {
    return arrow::Status::Invalid("null literal");
  }
  ARROW_ASSIGN_OR_RAISE(auto min_cmp, MakeComparable(min, literal));
  ARROW_ASSIGN_OR_RAISE(auto max_cmp, MakeComparable(max, literal));
  const auto& min_value = min_cmp.first;
  const auto& literal_min = min_cmp.second;
  const auto& max_value = max_cmp.first;
  const auto& literal_max = max_cmp.second;

  switch (op) {
    case sql::BinaryOp::Eq: {
      // v < min or v > max
      ARROW_ASSIGN_OR_RAISE(bool below, ScalarCompare("less", literal_min, min_value));
      ARROW_ASSIGN_OR_RAISE(bool above, ScalarCompare("greater", literal_max, max_value));
      return below || above;
    }
    case sql::BinaryOp::NotEq: {
      ARROW_ASSIGN_OR_RAISE(bool equal_min, ScalarCompare("equal", literal_min, min_value));
      ARROW_ASSIGN_OR_RAISE(bool equal_max, ScalarCompare("equal", literal_max, max_value));
      return equal_min && equal_max;
    }
    case sql::BinaryOp::Lt: {
      // no value < literal when min >= literal
      ARROW_ASSIGN_OR_RAISE(bool ge, ScalarCompare("greater_equal", min_value, literal_min));
      return ge;
    }
    case sql::BinaryOp::LtEq: {
      ARROW_ASSIGN_OR_RAISE(bool gt, ScalarCompare("greater", min_value, literal_min));
      return gt;
    }
    case sql::BinaryOp::Gt: {
      // no value > literal when max <= literal
      ARROW_ASSIGN_OR_RAISE(bool le, ScalarCompare("less_equal", max_value, literal_max));
      return le;
    }
    case sql::BinaryOp::GtEq: {
      ARROW_ASSIGN_OR_RAISE(bool lt, ScalarCompare("less", max_value, literal_max));
      return lt;
    }
    default:
      return arrow::Status::Invalid("operator is not a comparison");
  }
}

}  // namespace

ParquetSource::ParquetSource(std::string path, std::shared_ptr<arrow::Schema> schema,
                             std::unique_ptr<parquet::arrow::FileReader> reader)
    : path_(std::move(path)), schema_(std::move(schema)), reader_(std::move(reader)) {
  metadata_ = reader_->parquet_reader()->metadata();
  num_rows_ = metadata_ ? metadata_->num_rows() : 0;
  num_row_groups_ = reader_->num_row_groups();
}

arrow::Result<std::shared_ptr<ParquetSource>> ParquetSource::Open(const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto file, arrow::io::ReadableFile::Open(path));
  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(
      parquet::arrow::OpenFile(std::move(file), arrow::default_memory_pool(), &reader));
  reader->set_use_threads(false);

  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
  ARROW_RETURN_NOT_OK(ValidateFlatSchema(*schema));

  for (int i = 0; i < schema->num_fields(); ++i) {
    for (int j = i + 1; j < schema->num_fields(); ++j) {
      if (ToLower(schema->field(i)->name()) == ToLower(schema->field(j)->name())) {
        return arrow::Status::Invalid("Parquet file has duplicate column name '" +
                                      schema->field(i)->name() + "': " + path);
      }
    }
  }

  return std::shared_ptr<ParquetSource>(
      new ParquetSource(path, std::move(schema), std::move(reader)));
}

arrow::Result<int> ParquetSource::FieldIndex(const std::string& name) const {
  int found = -1;
  for (int i = 0; i < schema_->num_fields(); ++i) {
    if (ToLower(schema_->field(i)->name()) == ToLower(name)) {
      if (found >= 0) {
        return arrow::Status::Invalid("column name '" + name + "' is ambiguous in " + path_);
      }
      found = i;
    }
  }
  if (found < 0) {
    return arrow::Status::Invalid("column '" + name + "' does not exist in " + path_);
  }
  return found;
}

arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> ParquetSource::OpenReader(
    const std::vector<int>& row_groups, const std::vector<int>& columns, int64_t batch_size) {
  if (row_groups.empty()) {
    return arrow::Status::Invalid("cannot open a reader with no row groups");
  }
  reader_->set_batch_size(batch_size);
  std::unique_ptr<arrow::RecordBatchReader> reader;
  ARROW_RETURN_NOT_OK(reader_->GetRecordBatchReader(row_groups, columns, &reader));
  return reader;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetSource::ReadAll(
    const std::vector<int>& columns) const {
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader_->ReadTable(columns, &table));
  return table->CombineChunksToBatch();
}

bool RowGroupCanBeSkipped(const parquet::RowGroupMetaData& row_group,
                          const std::vector<ScanFilter>& filters) {
  if (filters.empty()) return false;
  for (const ScanFilter& filter : filters) {
    if (filter.column_index < 0 || filter.column_index >= row_group.num_columns()) continue;
    auto chunk = row_group.ColumnChunk(filter.column_index);
    if (!chunk || !chunk->is_stats_set()) continue;
    auto stats = chunk->statistics();
    if (!stats || !stats->HasMinMax() || stats->num_values() == 0) continue;

    std::shared_ptr<arrow::Scalar> min;
    std::shared_ptr<arrow::Scalar> max;
    auto status = parquet::arrow::StatisticsAsScalars(*stats, &min, &max);
    if (!status.ok() || !min || !max) continue;

    auto can_skip = RangeProvesNoMatch(min, max, filter.literal, filter.op);
    if (can_skip.ok() && can_skip.ValueOrDie()) {
      return true;
    }
  }
  return false;
}

std::string DescribeScanFilter(const ScanFilter& filter, const std::string& column_name) {
  const char* op = "";
  switch (filter.op) {
    case sql::BinaryOp::Eq: op = "="; break;
    case sql::BinaryOp::NotEq: op = "!="; break;
    case sql::BinaryOp::Lt: op = "<"; break;
    case sql::BinaryOp::LtEq: op = "<="; break;
    case sql::BinaryOp::Gt: op = ">"; break;
    case sql::BinaryOp::GtEq: op = ">="; break;
    default: op = "?"; break;
  }
  std::ostringstream out;
  out << column_name << " " << op << " ";
  switch (filter.literal.type) {
    case sql::LiteralType::Null: out << "NULL"; break;
    case sql::LiteralType::Integer: out << filter.literal.integer; break;
    case sql::LiteralType::Double: out << filter.literal.real; break;
    case sql::LiteralType::String: out << "'" << filter.literal.string << "'"; break;
    case sql::LiteralType::Boolean: out << (filter.literal.boolean ? "TRUE" : "FALSE"); break;
  }
  return out.str();
}

}  // namespace qe
