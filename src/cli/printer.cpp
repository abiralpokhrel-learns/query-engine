#include "cli/printer.h"

#include <arrow/array.h>
#include <arrow/type.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

namespace qe {
namespace {

constexpr size_t kMaxCellWidth = 42;

struct CivilDate {
  int year = 1970;
  int month = 1;
  int day = 1;
};

// Howard Hinnant's civil_from_days: days since 1970-01-01 -> calendar date.
CivilDate CivilFromDays(int64_t days) {
  int64_t z = days + 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int64_t y = static_cast<int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;

  CivilDate result;
  result.year = static_cast<int>(y + (m <= 2));
  result.month = static_cast<int>(m);
  result.day = static_cast<int>(d);
  return result;
}

std::string FormatDate32(int32_t value) {
  const CivilDate date = CivilFromDays(value);
  char buffer[32];
  if (date.year >= 0) {
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", date.year, date.month, date.day);
  } else {
    std::snprintf(buffer, sizeof(buffer), "-%04d-%02d-%02d", -date.year, date.month, date.day);
  }
  return buffer;
}

template <typename T> std::string FormatFloating(T value, int precision = 0) {
  if (std::isnan(value))
    return "nan";
  if (std::isinf(value))
    return value < 0 ? "-inf" : "inf";
  char buffer[64];
  auto result = precision > 0 ? std::to_chars(buffer, buffer + sizeof(buffer), value,
                                              std::chars_format::general, precision)
                              : std::to_chars(buffer, buffer + sizeof(buffer), value,
                                              std::chars_format::general);
  if (result.ec != std::errc()) {
    std::ostringstream fallback;
    fallback << std::setprecision(precision > 0 ? precision : 15) << value;
    return fallback.str();
  }
  return std::string(buffer, result.ptr);
}

// JSON keeps the shortest round-trippable representation; the human table
// trims binary floating-point noise so sums do not print as 875828.8700000027.
std::string FormatTableValue(const arrow::Array& array, int64_t index) {
  if (array.IsValid(index)) {
    if (array.type_id() == arrow::Type::FLOAT) {
      return FormatFloating(static_cast<const arrow::FloatArray&>(array).Value(index), 7);
    }
    if (array.type_id() == arrow::Type::DOUBLE) {
      return FormatFloating(static_cast<const arrow::DoubleArray&>(array).Value(index), 12);
    }
  }
  return FormatValue(array, index);
}

std::string EscapeForTable(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (char c : text) {
    switch (c) {
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      result.push_back(c);
      break;
    }
  }
  return result;
}

std::string Truncate(std::string text, size_t width) {
  if (text.size() <= width)
    return text;
  if (width <= 3)
    return text.substr(0, width);
  text.resize(width - 3);
  text += "...";
  return text;
}

struct ColumnView {
  std::vector<std::shared_ptr<arrow::Array>> chunks;
  int64_t num_rows = 0;
};

ColumnView MakeView(const std::shared_ptr<arrow::ChunkedArray>& column) {
  ColumnView view;
  for (const auto& chunk : column->chunks()) {
    if (!chunk || chunk->length() == 0)
      continue;
    view.chunks.push_back(chunk);
    view.num_rows += chunk->length();
  }
  return view;
}

std::string ValueAt(const ColumnView& view, int64_t row) {
  int64_t remaining = row;
  for (const auto& chunk : view.chunks) {
    if (remaining < chunk->length()) {
      return FormatTableValue(*chunk, remaining);
    }
    remaining -= chunk->length();
  }
  return "NULL";
}

void AppendJsonString(std::string& out, std::string_view text) {
  out.push_back('"');
  for (unsigned char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
        out += buffer;
      } else {
        out.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  out.push_back('"');
}

void AppendJsonValue(std::string& out, const arrow::Array& array, int64_t index) {
  if (!array.IsValid(index)) {
    out += "null";
    return;
  }

  switch (array.type_id()) {
  case arrow::Type::BOOL:
    out += static_cast<const arrow::BooleanArray&>(array).Value(index) ? "true" : "false";
    break;
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
    out += FormatValue(array, index);
    break;
  case arrow::Type::FLOAT:
  case arrow::Type::DOUBLE: {
    const std::string text = FormatValue(array, index);
    if (text == "nan" || text == "inf" || text == "-inf") {
      AppendJsonString(out, text);
    } else {
      out += text;
    }
    break;
  }
  case arrow::Type::STRING:
  case arrow::Type::LARGE_STRING:
  case arrow::Type::DATE32:
    AppendJsonString(out, FormatValue(array, index));
    break;
  case arrow::Type::NA:
    out += "null";
    break;
  default:
    AppendJsonString(out, FormatValue(array, index));
    break;
  }
}

} // namespace

std::string ColumnTypeName(const arrow::DataType& type) {
  switch (type.id()) {
  case arrow::Type::BOOL:
    return "bool";
  case arrow::Type::INT8:
    return "int8";
  case arrow::Type::INT16:
    return "int16";
  case arrow::Type::INT32:
    return "int32";
  case arrow::Type::INT64:
    return "int64";
  case arrow::Type::UINT8:
    return "uint8";
  case arrow::Type::UINT16:
    return "uint16";
  case arrow::Type::UINT32:
    return "uint32";
  case arrow::Type::UINT64:
    return "uint64";
  case arrow::Type::FLOAT:
    return "float";
  case arrow::Type::DOUBLE:
    return "double";
  case arrow::Type::STRING:
    return "string";
  case arrow::Type::LARGE_STRING:
    return "large_string";
  case arrow::Type::DATE32:
    return "date32";
  case arrow::Type::NA:
    return "null";
  default:
    return type.ToString();
  }
}

std::string FormatValue(const arrow::Array& array, int64_t index) {
  if (!array.IsValid(index))
    return "NULL";

  switch (array.type_id()) {
  case arrow::Type::BOOL:
    return static_cast<const arrow::BooleanArray&>(array).Value(index) ? "true" : "false";
  case arrow::Type::INT8:
    return std::to_string(static_cast<const arrow::Int8Array&>(array).Value(index));
  case arrow::Type::INT16:
    return std::to_string(static_cast<const arrow::Int16Array&>(array).Value(index));
  case arrow::Type::INT32:
    return std::to_string(static_cast<const arrow::Int32Array&>(array).Value(index));
  case arrow::Type::INT64:
    return std::to_string(static_cast<const arrow::Int64Array&>(array).Value(index));
  case arrow::Type::UINT8:
    return std::to_string(static_cast<const arrow::UInt8Array&>(array).Value(index));
  case arrow::Type::UINT16:
    return std::to_string(static_cast<const arrow::UInt16Array&>(array).Value(index));
  case arrow::Type::UINT32:
    return std::to_string(static_cast<const arrow::UInt32Array&>(array).Value(index));
  case arrow::Type::UINT64:
    return std::to_string(static_cast<const arrow::UInt64Array&>(array).Value(index));
  case arrow::Type::FLOAT:
    return FormatFloating(static_cast<const arrow::FloatArray&>(array).Value(index));
  case arrow::Type::DOUBLE:
    return FormatFloating(static_cast<const arrow::DoubleArray&>(array).Value(index));
  case arrow::Type::STRING:
    return static_cast<const arrow::StringArray&>(array).GetString(index);
  case arrow::Type::LARGE_STRING:
    return static_cast<const arrow::LargeStringArray&>(array).GetString(index);
  case arrow::Type::DATE32:
    return FormatDate32(static_cast<const arrow::Date32Array&>(array).Value(index));
  case arrow::Type::NA:
    return "NULL";
  default:
    break;
  }

  auto scalar = array.GetScalar(index);
  if (scalar.ok() && scalar.ValueOrDie())
    return scalar.ValueOrDie()->ToString();
  return "NULL";
}

void PrintTable(const arrow::Table& table, int64_t max_rows, std::ostream& out) {
  const int64_t total_rows = table.num_rows();
  int64_t visible_rows = total_rows;
  if (max_rows >= 0 && max_rows < visible_rows)
    visible_rows = max_rows;
  if (visible_rows < 0)
    visible_rows = 0;

  const int num_columns = table.num_columns();
  if (num_columns == 0) {
    out << "(query returned no columns)\n";
    return;
  }

  std::vector<std::string> headers;
  std::vector<size_t> widths;
  headers.reserve(num_columns);
  widths.reserve(num_columns);
  for (int c = 0; c < num_columns; ++c) {
    headers.push_back(table.schema()->field(c)->name());
    widths.push_back(std::min(headers.back().size(), kMaxCellWidth));
  }

  std::vector<ColumnView> views;
  views.reserve(num_columns);
  for (int c = 0; c < num_columns; ++c)
    views.push_back(MakeView(table.column(c)));

  std::vector<std::vector<std::string>> rows;
  rows.reserve(static_cast<size_t>(visible_rows));
  for (int64_t r = 0; r < visible_rows; ++r) {
    std::vector<std::string> row;
    row.reserve(num_columns);
    for (int c = 0; c < num_columns; ++c) {
      std::string value = EscapeForTable(ValueAt(views[c], r));
      if (value.size() > kMaxCellWidth)
        value = Truncate(std::move(value), kMaxCellWidth);
      widths[c] = std::max(widths[c], value.size());
      row.push_back(std::move(value));
    }
    rows.push_back(std::move(row));
  }

  auto print_rule = [&]() {
    for (int c = 0; c < num_columns; ++c) {
      if (c)
        out << "  ";
      out << std::string(widths[c], '-');
    }
    out << '\n';
  };

  for (int c = 0; c < num_columns; ++c) {
    if (c)
      out << "  ";
    out << std::left << std::setw(static_cast<int>(widths[c])) << headers[c];
  }
  out << '\n';
  print_rule();

  for (const auto& row : rows) {
    for (int c = 0; c < num_columns; ++c) {
      if (c)
        out << "  ";
      out << std::left << std::setw(static_cast<int>(widths[c])) << row[c];
    }
    out << '\n';
  }

  if (visible_rows < total_rows) {
    out << '(' << total_rows << " row" << (total_rows == 1 ? "" : "s") << ", showing first "
        << visible_rows << ")\n";
  } else {
    out << '(' << total_rows << " row" << (total_rows == 1 ? "" : "s") << ")\n";
  }
}

std::string TableToJson(const arrow::Table& table, int64_t max_rows) {
  const int64_t total_rows = table.num_rows();
  int64_t visible_rows = total_rows;
  if (max_rows >= 0 && max_rows < visible_rows)
    visible_rows = max_rows;
  if (visible_rows < 0)
    visible_rows = 0;

  std::vector<ColumnView> views;
  views.reserve(static_cast<size_t>(table.num_columns()));
  for (int c = 0; c < table.num_columns(); ++c)
    views.push_back(MakeView(table.column(c)));

  std::string out;
  out.reserve(static_cast<size_t>(visible_rows * table.num_columns() * 8 + 128));
  out += "{\"columns\":[";
  for (int c = 0; c < table.num_columns(); ++c) {
    if (c)
      out += ',';
    out += "{\"name\":";
    AppendJsonString(out, table.schema()->field(c)->name());
    out += ",\"type\":";
    AppendJsonString(out, ColumnTypeName(*table.schema()->field(c)->type()));
    out += '}';
  }
  out += "],\"rows\":[";

  for (int64_t r = 0; r < visible_rows; ++r) {
    if (r)
      out += ',';
    out += '[';
    for (int c = 0; c < table.num_columns(); ++c) {
      if (c)
        out += ',';
      const auto& view = views[c];
      int64_t remaining = r;
      bool found = false;
      for (const auto& chunk : view.chunks) {
        if (remaining < chunk->length()) {
          AppendJsonValue(out, *chunk, remaining);
          found = true;
          break;
        }
        remaining -= chunk->length();
      }
      if (!found)
        out += "null";
    }
    out += ']';
  }
  out += "]}";
  return out;
}

} // namespace qe
