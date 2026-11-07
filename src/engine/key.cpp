#include "engine/key.h"

#include <arrow/type.h>
#include <arrow/util/checked_cast.h>

#include <cmath>
#include <cstring>

namespace qe {
namespace {

void AppendU8(std::string& out, uint8_t value) {
  out.push_back(static_cast<char>(value));
}

void AppendU32(std::string& out, uint32_t value) {
  out.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void AppendU64(std::string& out, uint64_t value) {
  out.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

} // namespace

bool IsSupportedKeyType(const arrow::DataType& type) {
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

bool AreKeyTypesCompatible(const arrow::DataType& left, const arrow::DataType& right) {
  if (left.Equals(right))
    return true;
  const auto left_id = left.id();
  const auto right_id = right.id();
  const bool left_numeric =
      arrow::is_integer(left_id) || left_id == arrow::Type::FLOAT || left_id == arrow::Type::DOUBLE;
  const bool right_numeric = arrow::is_integer(right_id) || right_id == arrow::Type::FLOAT ||
                             right_id == arrow::Type::DOUBLE;
  if (left_numeric && right_numeric)
    return true;
  const bool left_string = left_id == arrow::Type::STRING || left_id == arrow::Type::LARGE_STRING;
  const bool right_string =
      right_id == arrow::Type::STRING || right_id == arrow::Type::LARGE_STRING;
  return left_string && right_string;
}

arrow::Result<std::shared_ptr<arrow::DataType>> CommonKeyType(const arrow::DataType& left,
                                                              const arrow::DataType& right) {
  if (left.Equals(right))
    return left.GetSharedPtr();
  if (AreKeyTypesCompatible(left, right)) {
    if (left.id() == arrow::Type::STRING || left.id() == arrow::Type::LARGE_STRING) {
      return arrow::utf8();
    }
    if (left.id() == arrow::Type::FLOAT || left.id() == arrow::Type::DOUBLE ||
        right.id() == arrow::Type::FLOAT || right.id() == arrow::Type::DOUBLE) {
      return arrow::float64();
    }
    return arrow::int64();
  }
  return arrow::Status::Invalid("incompatible join key types: " + left.ToString() + " and " +
                                right.ToString());
}

arrow::Status AppendArrayKey(const arrow::Array& array, int64_t index, std::string& out) {
  using arrow::internal::checked_cast;
  if (array.IsNull(index)) {
    AppendU8(out, 0);
    return arrow::Status::OK();
  }
  switch (array.type_id()) {
  case arrow::Type::BOOL:
    AppendU8(out, 1);
    AppendU8(out, checked_cast<const arrow::BooleanArray&>(array).Value(index) ? 1 : 0);
    return arrow::Status::OK();
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
  case arrow::Type::DATE32: {
    AppendU8(out, 2);
    int64_t value = 0;
    switch (array.type_id()) {
    case arrow::Type::INT8:
      value = checked_cast<const arrow::Int8Array&>(array).Value(index);
      break;
    case arrow::Type::INT16:
      value = checked_cast<const arrow::Int16Array&>(array).Value(index);
      break;
    case arrow::Type::INT32:
      value = checked_cast<const arrow::Int32Array&>(array).Value(index);
      break;
    case arrow::Type::INT64:
      value = checked_cast<const arrow::Int64Array&>(array).Value(index);
      break;
    case arrow::Type::UINT8:
      value = static_cast<int64_t>(checked_cast<const arrow::UInt8Array&>(array).Value(index));
      break;
    case arrow::Type::UINT16:
      value = static_cast<int64_t>(checked_cast<const arrow::UInt16Array&>(array).Value(index));
      break;
    case arrow::Type::UINT32:
      value = static_cast<int64_t>(checked_cast<const arrow::UInt32Array&>(array).Value(index));
      break;
    case arrow::Type::UINT64:
      value = static_cast<int64_t>(checked_cast<const arrow::UInt64Array&>(array).Value(index));
      break;
    case arrow::Type::DATE32:
      value = checked_cast<const arrow::Date32Array&>(array).Value(index);
      break;
    default:
      break;
    }
    AppendU64(out, static_cast<uint64_t>(value));
    return arrow::Status::OK();
  }
  case arrow::Type::FLOAT: {
    AppendU8(out, 3);
    double value = checked_cast<const arrow::FloatArray&>(array).Value(index);
    if (value == 0.0)
      value = 0.0; // collapse -0.0
    if (std::isnan(value)) {
      const uint64_t nan_bits = 0x7ff8000000000000ULL;
      out.append(reinterpret_cast<const char*>(&nan_bits), sizeof(nan_bits));
    } else {
      out.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    return arrow::Status::OK();
  }
  case arrow::Type::DOUBLE: {
    AppendU8(out, 3);
    double value = checked_cast<const arrow::DoubleArray&>(array).Value(index);
    if (value == 0.0)
      value = 0.0;
    if (std::isnan(value)) {
      const uint64_t nan_bits = 0x7ff8000000000000ULL;
      out.append(reinterpret_cast<const char*>(&nan_bits), sizeof(nan_bits));
    } else {
      out.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    return arrow::Status::OK();
  }
  case arrow::Type::STRING:
  case arrow::Type::LARGE_STRING:
  case arrow::Type::BINARY: {
    AppendU8(out, 4);
    std::string_view value;
    if (array.type_id() == arrow::Type::STRING) {
      value = checked_cast<const arrow::StringArray&>(array).GetView(index);
    } else if (array.type_id() == arrow::Type::LARGE_STRING) {
      value = checked_cast<const arrow::LargeStringArray&>(array).GetView(index);
    } else {
      value = checked_cast<const arrow::BinaryArray&>(array).GetView(index);
    }
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.append(value.data(), value.size());
    return arrow::Status::OK();
  }
  default:
    return arrow::Status::Invalid("unsupported key type " + array.type()->ToString());
  }
}

} // namespace qe
