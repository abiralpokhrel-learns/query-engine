#pragma once

#include <arrow/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sql/sql.h"

namespace qe {

class ParquetSource;

struct TableInfo {
  std::string name; // name used in SQL
  std::string path; // path on disk
  std::shared_ptr<arrow::Schema> schema;
  int64_t num_rows = 0;
  int num_row_groups = 0;
  std::shared_ptr<ParquetSource> source;
};

// Owns an open Parquet file and its metadata. A ParquetSource outlives every
// RecordBatchReader it creates (an Arrow requirement).
class ParquetSource {
public:
  static arrow::Result<std::shared_ptr<ParquetSource>> Open(const std::string& path);

  const std::string& path() const {
    return path_;
  }
  const std::shared_ptr<arrow::Schema>& schema() const {
    return schema_;
  }
  int64_t num_rows() const {
    return num_rows_;
  }
  int num_row_groups() const {
    return num_row_groups_;
  }
  const std::shared_ptr<parquet::FileMetaData>& metadata() const {
    return metadata_;
  }

  // Case-insensitive lookup of a file column.
  arrow::Result<int> FieldIndex(const std::string& name) const;

  arrow::Result<std::unique_ptr<arrow::RecordBatchReader>>
  OpenReader(const std::vector<int>& row_groups, const std::vector<int>& columns,
             int64_t batch_size);

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ReadAll(const std::vector<int>& columns) const;

private:
  ParquetSource(std::string path, std::shared_ptr<arrow::Schema> schema,
                std::unique_ptr<parquet::arrow::FileReader> reader);

  std::string path_;
  std::shared_ptr<arrow::Schema> schema_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<parquet::FileMetaData> metadata_;
  int64_t num_rows_ = 0;
  int num_row_groups_ = 0;
};

// A predicate of the form <column> <op> <literal>, used only for row-group
// skipping. Row-level filtering is always applied by a filter operator, so a
// wrong or missing statistic can only make the scan slower, never incorrect.
struct ScanFilter {
  int column_index = -1;
  sql::BinaryOp op = sql::BinaryOp::Eq;
  sql::Literal literal;
};

bool RowGroupCanBeSkipped(const parquet::RowGroupMetaData& row_group,
                          const std::vector<ScanFilter>& filters);

std::string DescribeScanFilter(const ScanFilter& filter, const std::string& column_name);

} // namespace qe
