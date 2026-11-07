#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <ostream>
#include <string>

namespace qe {

std::string FormatValue(const arrow::Array& array, int64_t index);
std::string ColumnTypeName(const arrow::DataType& type);

void PrintTable(const arrow::Table& table, int64_t max_rows, std::ostream& out);

// Serializes the result as {"columns":[{"name":...,"type":...}],"rows":[...]}.
// max_rows < 0 means every row.
std::string TableToJson(const arrow::Table& table, int64_t max_rows = -1);

} // namespace qe
