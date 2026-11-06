#pragma once

#include <arrow/api.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "sql/sql.h"

namespace qe {

// Evaluates an expression against a batch, returning either an Array (for
// column or row-wise expressions) or a Scalar (for literals and foldable
// expressions). Aggregates are rewritten to column references by the planner.
arrow::Result<arrow::Datum> EvaluateExpression(const sql::Expr& expr,
                                               const arrow::RecordBatch& batch);

arrow::Result<std::shared_ptr<arrow::Array>> EvaluateExpressionArray(const sql::Expr& expr,
                                                                     const arrow::RecordBatch& batch);

// Evaluates a predicate and returns only rows where the predicate is TRUE.
// NULL is treated as FALSE, as SQL WHERE requires.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> ApplyFilter(
    const arrow::RecordBatch& batch, const sql::Expr& predicate);

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ApplyProjection(
    const arrow::RecordBatch& batch, const std::vector<sql::ExprPtr>& expressions,
    const std::vector<std::string>& output_names);

// Resolves a column reference against a batch schema. Field names are
// internally qualified as "<table>.<column>".
arrow::Result<std::string> ResolveColumnName(const arrow::Schema& schema, const sql::Expr& expr);

bool ContainsAggregate(const sql::ExprPtr& expr);
void CollectColumnRefs(const sql::ExprPtr& expr, std::set<std::string>& out);
std::string CanonicalExpression(const sql::Expr& expr);
bool ExpressionsEqual(const sql::Expr& left, const sql::Expr& right);

arrow::Result<std::shared_ptr<arrow::Scalar>> LiteralToScalar(const sql::Literal& literal,
                                                             const arrow::DataType& type);

}  // namespace qe
