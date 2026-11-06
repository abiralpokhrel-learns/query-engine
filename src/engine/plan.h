#pragma once

#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

#include "sql/sql.h"
#include "storage/parquet.h"

namespace qe {

// A catalog holds the Parquet tables registered for one query. Names are
// matched case-insensitively.
class Catalog {
 public:
  arrow::Status AddFile(const std::string& name, const std::string& path);
  void Add(TableInfo info);
  const TableInfo* Find(const std::string& name) const;
  const std::vector<TableInfo>& tables() const { return tables_; }

 private:
  std::vector<TableInfo> tables_;
};

struct JoinCondition {
  sql::ExprPtr left;   // references the left input
  sql::ExprPtr right;  // references the right input
};

struct SortKey {
  sql::ExprPtr expression;
  bool descending = false;
};

struct LogicalPlan {
  enum class Kind { Scan, Filter, Join, Aggregate, Project, Distinct, Sort, Limit };

  Kind kind = Kind::Scan;
  std::vector<std::shared_ptr<LogicalPlan>> children;
  std::shared_ptr<arrow::Schema> schema;

  // Scan
  std::string table_name;
  std::string table_path;
  std::string table_alias;
  std::shared_ptr<ParquetSource> table_source;
  std::shared_ptr<arrow::Schema> file_schema;
  std::vector<std::string> scan_columns;  // file column names, file order
  std::vector<sql::ExprPtr> pushed_filters;
  int64_t estimated_rows = 0;

  // Filter
  sql::ExprPtr predicate;

  // Join
  std::vector<JoinCondition> join_conditions;
  bool left_outer = false;

  // Aggregate
  std::vector<sql::ExprPtr> group_by;
  std::vector<sql::ExprPtr> aggregates;
  std::vector<std::string> aggregate_names;

  // Project
  std::vector<sql::ExprPtr> projections;
  std::vector<std::string> output_names;

  // Sort
  std::vector<SortKey> sort_keys;
  // If the query has ORDER BY ... LIMIT, the sort only needs to retain the
  // first N rows. A value of -1 means "no limit".
  int64_t limit_hint = -1;

  // Limit
  int64_t limit = -1;
  int64_t offset = 0;
};

struct QueryPlan {
  std::shared_ptr<LogicalPlan> root;
  std::shared_ptr<arrow::Schema> output_schema;
  std::string explain;
};

arrow::Result<QueryPlan> PlanQuery(const sql::SelectStmt& statement, const Catalog& catalog);

}  // namespace qe
