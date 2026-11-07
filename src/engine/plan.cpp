#include "engine/plan.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_set>

#include "engine/expression.h"

namespace qe {
namespace {

std::string ToLower(std::string_view text) {
  std::string result(text);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

bool IEquals(std::string_view left, std::string_view right) {
  return ToLower(left) == ToLower(right);
}

sql::ExprPtr MakeColumn(std::string name) {
  auto expr = std::make_shared<sql::Expr>();
  expr->kind = sql::ExprKind::Column;
  expr->column = std::move(name);
  return expr;
}

arrow::Result<std::shared_ptr<arrow::DataType>> NaturalLiteralType(const sql::Literal& literal) {
  switch (literal.type) {
  case sql::LiteralType::Null:
    return arrow::null();
  case sql::LiteralType::Integer:
    return arrow::int64();
  case sql::LiteralType::Double:
    return arrow::float64();
  case sql::LiteralType::String:
    return arrow::utf8();
  case sql::LiteralType::Boolean:
    return arrow::boolean();
  }
  return arrow::Status::Invalid("unsupported literal type");
}

arrow::Result<std::shared_ptr<arrow::DataType>>
InferType(const sql::Expr& expr, const std::shared_ptr<arrow::Schema>& schema) {
  switch (expr.kind) {
  case sql::ExprKind::Column: {
    ARROW_ASSIGN_OR_RAISE(auto name, ResolveColumnName(*schema, expr));
    int index = schema->GetFieldIndex(name);
    if (index < 0)
      return arrow::Status::Invalid("column '" + name + "' not found");
    return schema->field(index)->type();
  }
  case sql::ExprKind::Literal:
    return NaturalLiteralType(expr.literal);
  case sql::ExprKind::IsNull:
    return arrow::boolean();
  case sql::ExprKind::Unary:
    if (expr.unary_op == sql::UnaryOp::Not)
      return arrow::boolean();
    return InferType(*expr.left, schema);
  case sql::ExprKind::Binary: {
    switch (expr.binary_op) {
    case sql::BinaryOp::Eq:
    case sql::BinaryOp::NotEq:
    case sql::BinaryOp::Lt:
    case sql::BinaryOp::LtEq:
    case sql::BinaryOp::Gt:
    case sql::BinaryOp::GtEq:
    case sql::BinaryOp::And:
    case sql::BinaryOp::Or:
      return arrow::boolean();
    case sql::BinaryOp::Add:
    case sql::BinaryOp::Sub:
    case sql::BinaryOp::Mul:
    case sql::BinaryOp::Div: {
      if (expr.binary_op == sql::BinaryOp::Div) {
        // Like DuckDB (and SQL generally), integer division produces a
        // double. The expression evaluator coerces both sides to double.
        return arrow::float64();
      }
      ARROW_ASSIGN_OR_RAISE(auto left, InferType(*expr.left, schema));
      ARROW_ASSIGN_OR_RAISE(auto right, InferType(*expr.right, schema));
      const bool floating = left->id() == arrow::Type::FLOAT || left->id() == arrow::Type::DOUBLE ||
                            right->id() == arrow::Type::FLOAT || right->id() == arrow::Type::DOUBLE;
      return floating ? arrow::float64() : arrow::int64();
    }
    }
    return arrow::Status::Invalid("unsupported binary expression");
  }
  case sql::ExprKind::Aggregate:
    return arrow::Status::Invalid("aggregate expression was not rewritten");
  case sql::ExprKind::Star:
    return arrow::Status::Invalid("star expression was not expanded");
  }
  return arrow::Status::Invalid("unsupported expression");
}

std::shared_ptr<arrow::Schema>
MakeQualifiedSchema(const std::string& alias, const std::shared_ptr<arrow::Schema>& file_schema,
                    const std::vector<std::string>& columns) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (const std::string& column : columns) {
    int index = file_schema->GetFieldIndex(column);
    if (index < 0)
      continue;
    fields.push_back(arrow::field(alias + "." + file_schema->field(index)->name(),
                                  file_schema->field(index)->type()));
  }
  return arrow::schema(std::move(fields));
}

std::vector<sql::ExprPtr> SplitConjuncts(const sql::ExprPtr& expr) {
  std::vector<sql::ExprPtr> result;
  if (!expr)
    return result;
  if (expr->kind == sql::ExprKind::Binary && expr->binary_op == sql::BinaryOp::And) {
    auto left = SplitConjuncts(expr->left);
    auto right = SplitConjuncts(expr->right);
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
  } else {
    result.push_back(expr);
  }
  return result;
}

sql::ExprPtr CombineConjuncts(const std::vector<sql::ExprPtr>& conjuncts) {
  sql::ExprPtr result;
  for (const auto& conjunct : conjuncts) {
    if (!result) {
      result = conjunct;
    } else {
      auto expr = std::make_shared<sql::Expr>();
      expr->kind = sql::ExprKind::Binary;
      expr->binary_op = sql::BinaryOp::And;
      expr->left = result;
      expr->right = conjunct;
      result = expr;
    }
  }
  return result;
}

std::string TableOf(const sql::Expr& expr) {
  if (expr.kind != sql::ExprKind::Column)
    return "";
  return expr.table;
}

struct Binding {
  std::string alias;
  const TableInfo* info = nullptr;
  std::shared_ptr<LogicalPlan> scan;
};

arrow::Result<const arrow::Field*> FindField(const TableInfo& info, const std::string& column) {
  for (int i = 0; i < info.schema->num_fields(); ++i) {
    if (IEquals(info.schema->field(i)->name(), column))
      return info.schema->field(i).get();
  }
  return arrow::Status::Invalid("column '" + column + "' does not exist in table '" + info.name +
                                "'");
}

arrow::Result<int> FindBinding(const std::vector<Binding>& bindings, const std::string& alias) {
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (IEquals(bindings[i].alias, alias))
      return static_cast<int>(i);
  }
  return arrow::Status::Invalid("table alias '" + alias + "' is not defined");
}

arrow::Status ResolveExpression(const sql::ExprPtr& expr, const std::vector<Binding>& bindings,
                                bool allow_aggregates) {
  if (!expr)
    return arrow::Status::OK();

  switch (expr->kind) {
  case sql::ExprKind::Column: {
    int found_binding = -1;
    std::string found_column;
    if (!expr->table.empty()) {
      ARROW_ASSIGN_OR_RAISE(found_binding, FindBinding(bindings, expr->table));
      ARROW_ASSIGN_OR_RAISE(auto field, FindField(*bindings[found_binding].info, expr->column));
      found_column = field->name();
    } else {
      for (size_t i = 0; i < bindings.size(); ++i) {
        for (int j = 0; j < bindings[i].info->schema->num_fields(); ++j) {
          if (IEquals(bindings[i].info->schema->field(j)->name(), expr->column)) {
            if (found_binding >= 0) {
              return arrow::Status::Invalid(
                  "column '" + expr->column +
                  "' is ambiguous; qualify it with a table alias (candidates: " +
                  bindings[found_binding].alias + "." + found_column + ", " + bindings[i].alias +
                  "." + bindings[i].info->schema->field(j)->name() + ")");
            }
            found_binding = static_cast<int>(i);
            found_column = bindings[i].info->schema->field(j)->name();
          }
        }
      }
      if (found_binding < 0) {
        std::ostringstream message;
        message << "column '" << expr->column << "' does not exist; available columns: ";
        for (size_t i = 0; i < bindings.size(); ++i) {
          for (int j = 0; j < bindings[i].info->schema->num_fields(); ++j) {
            if (i || j)
              message << ", ";
            message << bindings[i].alias << "." << bindings[i].info->schema->field(j)->name();
          }
        }
        return arrow::Status::Invalid(message.str());
      }
    }
    expr->table = bindings[found_binding].alias;
    expr->column = found_column;
    return arrow::Status::OK();
  }
  case sql::ExprKind::Aggregate: {
    if (!allow_aggregates) {
      return arrow::Status::Invalid("aggregate function " + expr->function +
                                    " is not allowed in this clause");
    }
    if (expr->argument && ContainsAggregate(expr->argument)) {
      return arrow::Status::Invalid("nested aggregate functions are not supported");
    }
    return ResolveExpression(expr->argument, bindings, false);
  }
  case sql::ExprKind::Binary:
    ARROW_RETURN_NOT_OK(ResolveExpression(expr->left, bindings, allow_aggregates));
    ARROW_RETURN_NOT_OK(ResolveExpression(expr->right, bindings, allow_aggregates));
    return arrow::Status::OK();
  case sql::ExprKind::Unary:
  case sql::ExprKind::IsNull:
    return ResolveExpression(expr->left, bindings, allow_aggregates);
  case sql::ExprKind::Literal:
    return arrow::Status::OK();
  case sql::ExprKind::Star:
    return arrow::Status::Invalid("star expression is only valid in a SELECT list");
  }
  return arrow::Status::Invalid("unsupported expression");
}

void CollectAggregates(const sql::ExprPtr& expr, std::vector<sql::ExprPtr>& out) {
  if (!expr)
    return;
  if (expr->kind == sql::ExprKind::Aggregate) {
    std::string canonical = CanonicalExpression(*expr);
    for (const auto& existing : out) {
      if (CanonicalExpression(*existing) == canonical)
        return;
    }
    out.push_back(expr);
    return;
  }
  CollectAggregates(expr->left, out);
  CollectAggregates(expr->right, out);
}

arrow::Status ValidateAggregate(const sql::Expr& aggregate) {
  if (aggregate.star) {
    if (aggregate.function != "count") {
      return arrow::Status::Invalid("only COUNT(*) accepts *");
    }
    return arrow::Status::OK();
  }
  if (!aggregate.argument) {
    return arrow::Status::Invalid("aggregate function requires an argument");
  }
  if (aggregate.argument->kind != sql::ExprKind::Column) {
    return arrow::Status::Invalid(
        "aggregate arguments must be plain columns; expressions such as SUM(a * b) are not "
        "supported");
  }
  return arrow::Status::OK();
}

arrow::Status ValidateGroupedExpression(const sql::ExprPtr& expr,
                                        const std::set<std::string>& group_canonicals) {
  if (expr->kind == sql::ExprKind::Aggregate) {
    return ValidateAggregate(*expr);
  }
  if (expr->kind == sql::ExprKind::Literal)
    return arrow::Status::OK();
  if (expr->kind == sql::ExprKind::Column) {
    if (!group_canonicals.contains(CanonicalExpression(*expr))) {
      return arrow::Status::Invalid("column " + CanonicalExpression(*expr) +
                                    " must appear in GROUP BY or be used in an aggregate function");
    }
    return arrow::Status::OK();
  }
  if (ContainsAggregate(expr)) {
    return arrow::Status::Invalid(
        "expressions that combine aggregates with other operations are not supported");
  }
  return arrow::Status::Invalid(
      "GROUP BY queries support group columns, aggregate functions, and literals only");
}

sql::ExprPtr RewriteGroupedExpression(const sql::ExprPtr& expr,
                                      const std::set<std::string>& aggregate_names,
                                      const std::set<std::string>& group_names) {
  if (expr->kind == sql::ExprKind::Aggregate) {
    std::string name = CanonicalExpression(*expr);
    if (!aggregate_names.contains(name))
      return nullptr;
    return MakeColumn(name);
  }
  if (expr->kind == sql::ExprKind::Column) {
    std::string name = CanonicalExpression(*expr);
    if (!group_names.contains(name))
      return nullptr;
    return MakeColumn(name);
  }
  if (expr->kind == sql::ExprKind::Literal)
    return expr;
  return nullptr;
}

arrow::Status ValidateHavingExpression(const sql::ExprPtr& expr,
                                       const std::set<std::string>& group_names) {
  if (!expr)
    return arrow::Status::OK();
  switch (expr->kind) {
  case sql::ExprKind::Aggregate:
    return ValidateAggregate(*expr);
  case sql::ExprKind::Column:
    if (!group_names.contains(CanonicalExpression(*expr))) {
      return arrow::Status::Invalid("HAVING column " + CanonicalExpression(*expr) +
                                    " must appear in GROUP BY or be used in an aggregate function");
    }
    return arrow::Status::OK();
  case sql::ExprKind::Literal:
    return arrow::Status::OK();
  case sql::ExprKind::Binary:
    ARROW_RETURN_NOT_OK(ValidateHavingExpression(expr->left, group_names));
    return ValidateHavingExpression(expr->right, group_names);
  case sql::ExprKind::Unary:
  case sql::ExprKind::IsNull:
    return ValidateHavingExpression(expr->left, group_names);
  case sql::ExprKind::Star:
    return arrow::Status::Invalid("star expression is only valid in a SELECT list");
  }
  return arrow::Status::Invalid("unsupported HAVING expression");
}

sql::ExprPtr RewriteGroupedExpressionDeep(const sql::ExprPtr& expr,
                                          const std::set<std::string>& aggregate_names,
                                          const std::set<std::string>& group_names) {
  if (!expr)
    return nullptr;
  if (expr->kind == sql::ExprKind::Aggregate) {
    std::string name = CanonicalExpression(*expr);
    if (!aggregate_names.contains(name))
      return nullptr;
    return MakeColumn(name);
  }
  if (expr->kind == sql::ExprKind::Column) {
    std::string name = CanonicalExpression(*expr);
    if (!group_names.contains(name))
      return nullptr;
    return MakeColumn(name);
  }
  if (expr->kind == sql::ExprKind::Literal)
    return expr;
  if (expr->kind == sql::ExprKind::Binary) {
    auto left = RewriteGroupedExpressionDeep(expr->left, aggregate_names, group_names);
    auto right = RewriteGroupedExpressionDeep(expr->right, aggregate_names, group_names);
    if (!left || !right)
      return nullptr;
    auto result = std::make_shared<sql::Expr>();
    result->kind = sql::ExprKind::Binary;
    result->binary_op = expr->binary_op;
    result->left = std::move(left);
    result->right = std::move(right);
    return result;
  }
  if (expr->kind == sql::ExprKind::Unary) {
    auto left = RewriteGroupedExpressionDeep(expr->left, aggregate_names, group_names);
    if (!left)
      return nullptr;
    auto result = std::make_shared<sql::Expr>();
    result->kind = sql::ExprKind::Unary;
    result->unary_op = expr->unary_op;
    result->left = std::move(left);
    return result;
  }
  if (expr->kind == sql::ExprKind::IsNull) {
    auto left = RewriteGroupedExpressionDeep(expr->left, aggregate_names, group_names);
    if (!left)
      return nullptr;
    auto result = std::make_shared<sql::Expr>();
    result->kind = sql::ExprKind::IsNull;
    result->left = std::move(left);
    return result;
  }
  return nullptr;
}

arrow::Result<std::shared_ptr<arrow::DataType>> AggregateResultType(const sql::Expr& aggregate,
                                                                    const arrow::Schema& input) {
  if (aggregate.function == "count")
    return arrow::int64();
  if (!aggregate.argument) {
    return arrow::Status::Invalid("aggregate function requires an argument");
  }
  ARROW_ASSIGN_OR_RAISE(auto name, ResolveColumnName(input, *aggregate.argument));
  int index = input.GetFieldIndex(name);
  if (index < 0)
    return arrow::Status::Invalid("aggregate argument column not found");
  auto type = input.field(index)->type();
  const bool integer = type->id() == arrow::Type::INT8 || type->id() == arrow::Type::INT16 ||
                       type->id() == arrow::Type::INT32 || type->id() == arrow::Type::INT64 ||
                       type->id() == arrow::Type::UINT8 || type->id() == arrow::Type::UINT16 ||
                       type->id() == arrow::Type::UINT32 || type->id() == arrow::Type::UINT64;
  const bool floating = type->id() == arrow::Type::FLOAT || type->id() == arrow::Type::DOUBLE;

  if (aggregate.function == "sum") {
    if (integer)
      return arrow::int64();
    if (floating)
      return arrow::float64();
    return arrow::Status::Invalid("SUM requires a numeric column, got " + type->ToString());
  }
  if (aggregate.function == "avg") {
    if (integer || floating)
      return arrow::float64();
    return arrow::Status::Invalid("AVG requires a numeric column, got " + type->ToString());
  }
  if (aggregate.function == "min" || aggregate.function == "max") {
    return type;
  }
  return arrow::Status::Invalid("unsupported aggregate function " + aggregate.function);
}

struct ResolvedItem {
  sql::ExprPtr expr;
  std::string alias;
};

std::shared_ptr<LogicalPlan> FindScan(const std::shared_ptr<LogicalPlan>& node,
                                      const std::string& alias) {
  if (!node)
    return nullptr;
  if (node->kind == LogicalPlan::Kind::Scan && IEquals(node->table_alias, alias))
    return node;
  for (const auto& child : node->children) {
    auto found = FindScan(child, alias);
    if (found)
      return found;
  }
  return nullptr;
}

bool HasOuterJoin(const std::shared_ptr<LogicalPlan>& node) {
  if (!node)
    return false;
  if (node->kind == LogicalPlan::Kind::Join && node->left_outer)
    return true;
  for (const auto& child : node->children) {
    if (HasOuterJoin(child))
      return true;
  }
  return false;
}

arrow::Status PushdownFilters(std::shared_ptr<LogicalPlan>& node) {
  if (!node)
    return arrow::Status::OK();
  for (auto& child : node->children) {
    ARROW_RETURN_NOT_OK(PushdownFilters(child));
  }
  if (node->kind != LogicalPlan::Kind::Filter || !node->predicate) {
    return arrow::Status::OK();
  }
  // A predicate above an outer join cannot be pushed into the nullable side
  // without changing semantics. Keep the filter above the join.
  if (HasOuterJoin(node->children[0])) {
    return arrow::Status::OK();
  }

  auto conjuncts = SplitConjuncts(node->predicate);
  std::vector<sql::ExprPtr> residual;
  for (const auto& conjunct : conjuncts) {
    std::set<std::string> refs;
    CollectColumnRefs(conjunct, refs);
    std::set<std::string> aliases;
    for (const std::string& ref : refs) {
      auto dot = ref.find('.');
      if (dot != std::string::npos)
        aliases.insert(ToLower(ref.substr(0, dot)));
    }
    if (aliases.size() == 1) {
      auto scan = FindScan(node->children[0], *aliases.begin());
      if (scan) {
        scan->pushed_filters.push_back(conjunct);
        continue;
      }
    }
    residual.push_back(conjunct);
  }

  if (residual.empty()) {
    node = node->children[0];
  } else {
    node->predicate = CombineConjuncts(residual);
  }
  return arrow::Status::OK();
}

void CollectPlanRefs(const std::shared_ptr<LogicalPlan>& node, std::set<std::string>& refs) {
  if (!node)
    return;
  switch (node->kind) {
  case LogicalPlan::Kind::Scan:
    for (const auto& filter : node->pushed_filters)
      CollectColumnRefs(filter, refs);
    break;
  case LogicalPlan::Kind::Filter:
    CollectColumnRefs(node->predicate, refs);
    break;
  case LogicalPlan::Kind::Join:
    for (const auto& condition : node->join_conditions) {
      CollectColumnRefs(condition.left, refs);
      CollectColumnRefs(condition.right, refs);
    }
    break;
  case LogicalPlan::Kind::Aggregate:
    for (const auto& expr : node->group_by)
      CollectColumnRefs(expr, refs);
    for (const auto& expr : node->aggregates)
      CollectColumnRefs(expr->argument, refs);
    break;
  case LogicalPlan::Kind::Project:
    for (const auto& expr : node->projections)
      CollectColumnRefs(expr, refs);
    break;
  case LogicalPlan::Kind::Sort:
    for (const auto& key : node->sort_keys)
      CollectColumnRefs(key.expression, refs);
    break;
  case LogicalPlan::Kind::Distinct:
    break;
  case LogicalPlan::Kind::Limit:
    break;
  }
  for (const auto& child : node->children)
    CollectPlanRefs(child, refs);
}

arrow::Status PruneNode(const std::shared_ptr<LogicalPlan>& node, const std::set<std::string>& refs,
                        bool allow_empty_scan) {
  if (!node)
    return arrow::Status::OK();
  if (node->kind == LogicalPlan::Kind::Scan) {
    std::vector<std::string> columns;
    for (int i = 0; i < node->file_schema->num_fields(); ++i) {
      const std::string qualified =
          ToLower(node->table_alias + "." + node->file_schema->field(i)->name());
      if (refs.contains(qualified)) {
        columns.push_back(node->file_schema->field(i)->name());
      }
    }
    if (columns.empty() && (!node->pushed_filters.empty() || !allow_empty_scan)) {
      // A residual predicate or pushed filter needs a decoded column.
      if (node->file_schema->num_fields() > 0) {
        columns.push_back(node->file_schema->field(0)->name());
      }
    }
    node->scan_columns = std::move(columns);
    node->schema = MakeQualifiedSchema(node->table_alias, node->file_schema, node->scan_columns);
    return arrow::Status::OK();
  }
  for (const auto& child : node->children) {
    ARROW_RETURN_NOT_OK(PruneNode(child, refs, allow_empty_scan));
  }
  return arrow::Status::OK();
}

bool HasResidualFilter(const std::shared_ptr<LogicalPlan>& node) {
  if (!node)
    return false;
  if (node->kind == LogicalPlan::Kind::Filter && node->predicate)
    return true;
  for (const auto& child : node->children) {
    if (HasResidualFilter(child))
      return true;
  }
  return false;
}

arrow::Status PruneColumns(const std::shared_ptr<LogicalPlan>& root) {
  std::set<std::string> refs;
  CollectPlanRefs(root, refs);
  return PruneNode(root, refs, !HasResidualFilter(root));
}

arrow::Status RebuildSchemas(const std::shared_ptr<LogicalPlan>& node) {
  if (!node)
    return arrow::Status::OK();
  for (const auto& child : node->children) {
    ARROW_RETURN_NOT_OK(RebuildSchemas(child));
  }
  switch (node->kind) {
  case LogicalPlan::Kind::Scan:
    node->schema = MakeQualifiedSchema(node->table_alias, node->file_schema, node->scan_columns);
    break;
  case LogicalPlan::Kind::Filter:
  case LogicalPlan::Kind::Distinct:
  case LogicalPlan::Kind::Sort:
  case LogicalPlan::Kind::Limit:
    node->schema = node->children[0]->schema;
    break;
  case LogicalPlan::Kind::Join: {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& child : node->children) {
      for (int i = 0; i < child->schema->num_fields(); ++i) {
        fields.push_back(child->schema->field(i));
      }
    }
    node->schema = arrow::schema(std::move(fields));
    break;
  }
  case LogicalPlan::Kind::Aggregate:
  case LogicalPlan::Kind::Project:
    // These schemas are built during planning and are independent of pruning.
    break;
  }
  return arrow::Status::OK();
}

void ExplainNode(const std::shared_ptr<LogicalPlan>& node, int depth, std::ostringstream& out) {
  out << std::string(static_cast<size_t>(depth) * 2, ' ');
  switch (node->kind) {
  case LogicalPlan::Kind::Scan: {
    out << "Scan " << node->table_alias << " (" << node->table_name << ")";
    out << " columns=[";
    for (size_t i = 0; i < node->scan_columns.size(); ++i) {
      if (i)
        out << ", ";
      out << node->scan_columns[i];
    }
    out << "] rows=" << node->estimated_rows;
    if (!node->pushed_filters.empty()) {
      out << " pushdown=[";
      for (size_t i = 0; i < node->pushed_filters.size(); ++i) {
        if (i)
          out << ", ";
        out << sql::ExprToString(node->pushed_filters[i]);
      }
      out << "]";
    }
    break;
  }
  case LogicalPlan::Kind::Filter:
    out << "Filter " << sql::ExprToString(node->predicate);
    break;
  case LogicalPlan::Kind::Join: {
    out << (node->left_outer ? "LeftJoin" : "Join") << " on [";
    for (size_t i = 0; i < node->join_conditions.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(node->join_conditions[i].left) << " = "
          << sql::ExprToString(node->join_conditions[i].right);
    }
    out << "]";
    break;
  }
  case LogicalPlan::Kind::Aggregate: {
    out << "Aggregate group=[";
    for (size_t i = 0; i < node->group_by.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(node->group_by[i]);
    }
    out << "] aggregates=[";
    for (size_t i = 0; i < node->aggregates.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(node->aggregates[i]);
    }
    out << "]";
    break;
  }
  case LogicalPlan::Kind::Project:
    out << "Project [";
    for (size_t i = 0; i < node->output_names.size(); ++i) {
      if (i)
        out << ", ";
      out << node->output_names[i];
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Distinct:
    out << "Distinct";
    break;
  case LogicalPlan::Kind::Sort:
    out << "Sort [";
    for (size_t i = 0; i < node->sort_keys.size(); ++i) {
      if (i)
        out << ", ";
      out << sql::ExprToString(node->sort_keys[i].expression)
          << (node->sort_keys[i].descending ? " DESC" : " ASC");
    }
    out << "]";
    break;
  case LogicalPlan::Kind::Limit:
    out << "Limit " << node->limit;
    if (node->offset > 0)
      out << " offset=" << node->offset;
    break;
  }
  out << "\n";
  for (const auto& child : node->children)
    ExplainNode(child, depth + 1, out);
}

} // namespace

arrow::Status Catalog::AddFile(const std::string& name, const std::string& path) {
  if (Find(name)) {
    return arrow::Status::Invalid("table '" + name + "' is already registered");
  }
  ARROW_ASSIGN_OR_RAISE(auto source, ParquetSource::Open(path));
  TableInfo info;
  info.name = name;
  info.path = path;
  info.schema = source->schema();
  info.num_rows = source->num_rows();
  info.num_row_groups = source->num_row_groups();
  info.source = std::move(source);
  tables_.push_back(std::move(info));
  return arrow::Status::OK();
}

void Catalog::Add(TableInfo info) {
  tables_.push_back(std::move(info));
}

const TableInfo* Catalog::Find(const std::string& name) const {
  for (const auto& table : tables_) {
    if (IEquals(table.name, name))
      return &table;
  }
  return nullptr;
}

arrow::Result<QueryPlan> PlanQuery(const sql::SelectStmt& statement, const Catalog& catalog) {
  std::vector<Binding> bindings;

  auto add_binding = [&](const sql::TableRef& ref) -> arrow::Status {
    const TableInfo* info = catalog.Find(ref.name);
    if (!info) {
      std::ostringstream message;
      message << "table '" << ref.name << "' is not registered; available tables: ";
      for (size_t i = 0; i < catalog.tables().size(); ++i) {
        if (i)
          message << ", ";
        message << catalog.tables()[i].name;
      }
      return arrow::Status::Invalid(message.str());
    }
    std::string alias = ref.alias.empty() ? ref.name : ref.alias;
    for (const auto& binding : bindings) {
      if (IEquals(binding.alias, alias)) {
        return arrow::Status::Invalid("duplicate table alias '" + alias + "'");
      }
    }
    Binding binding;
    binding.alias = alias;
    binding.info = info;
    auto scan = std::make_shared<LogicalPlan>();
    scan->kind = LogicalPlan::Kind::Scan;
    scan->table_name = info->name;
    scan->table_path = info->path;
    scan->table_alias = alias;
    scan->table_source = info->source;
    scan->file_schema = info->schema;
    scan->estimated_rows = info->num_rows;
    for (int i = 0; i < info->schema->num_fields(); ++i) {
      scan->scan_columns.push_back(info->schema->field(i)->name());
    }
    scan->schema = MakeQualifiedSchema(alias, info->schema, scan->scan_columns);
    binding.scan = scan;
    bindings.push_back(std::move(binding));
    return arrow::Status::OK();
  };

  ARROW_RETURN_NOT_OK(add_binding(statement.from));
  for (const auto& join : statement.joins) {
    ARROW_RETURN_NOT_OK(add_binding(join.table));
  }

  // Expand SELECT * and resolve every column reference against the table schemas.
  std::vector<ResolvedItem> items;
  for (const auto& item : statement.items) {
    if (item.star) {
      if (item.star_table.empty()) {
        for (const auto& binding : bindings) {
          for (int i = 0; i < binding.info->schema->num_fields(); ++i) {
            const auto& field = binding.info->schema->field(i);
            ResolvedItem resolved;
            auto column = MakeColumn(field->name());
            column->table = binding.alias;
            resolved.expr = column;
            resolved.alias = field->name();
            items.push_back(std::move(resolved));
          }
        }
      } else {
        ARROW_ASSIGN_OR_RAISE(int index, FindBinding(bindings, item.star_table));
        for (int i = 0; i < bindings[index].info->schema->num_fields(); ++i) {
          const auto& field = bindings[index].info->schema->field(i);
          ResolvedItem resolved;
          auto column = MakeColumn(field->name());
          column->table = bindings[index].alias;
          resolved.expr = column;
          resolved.alias = field->name();
          items.push_back(std::move(resolved));
        }
      }
      continue;
    }

    sql::ExprPtr expr = item.expr;
    ARROW_RETURN_NOT_OK(ResolveExpression(expr, bindings, true));
    ResolvedItem resolved;
    resolved.expr = expr;
    if (!item.alias.empty()) {
      resolved.alias = item.alias;
    } else if (expr->kind == sql::ExprKind::Column) {
      resolved.alias = expr->column;
    } else {
      resolved.alias = CanonicalExpression(*expr);
    }
    items.push_back(std::move(resolved));
  }
  if (items.empty()) {
    return arrow::Status::Invalid("query has no output columns");
  }

  if (statement.where) {
    ARROW_RETURN_NOT_OK(ResolveExpression(statement.where, bindings, false));
  }
  if (statement.having) {
    ARROW_RETURN_NOT_OK(ResolveExpression(statement.having, bindings, true));
  }

  std::vector<sql::ExprPtr> group_by;
  std::set<std::string> group_canonicals;
  for (const auto& expr : statement.group_by) {
    sql::ExprPtr resolved = expr;
    ARROW_RETURN_NOT_OK(ResolveExpression(resolved, bindings, false));
    if (resolved->kind != sql::ExprKind::Column) {
      return arrow::Status::Invalid("GROUP BY supports plain columns only; expression " +
                                    CanonicalExpression(*resolved) + " is not supported");
    }
    std::string canonical = CanonicalExpression(*resolved);
    if (group_canonicals.insert(canonical).second) {
      group_by.push_back(resolved);
    }
  }

  // HAVING always makes the query an aggregate query: without GROUP BY it
  // describes the single group containing the whole input. This matches
  // DuckDB even when the select list and HAVING contain no aggregate call.

  std::vector<sql::ExprPtr> order_expressions;
  for (const auto& order : statement.order_by) {
    if (order.ordinal) {
      order_expressions.push_back(nullptr);
      continue;
    }
    // A SELECT alias takes precedence in ORDER BY. Resolve it to the already
    // resolved select expression before trying the table schemas (an alias
    // such as "total" is not a physical column).
    sql::ExprPtr resolved;
    if (order.expr->kind == sql::ExprKind::Column && order.expr->table.empty()) {
      for (const auto& item : items) {
        if (IEquals(item.alias, order.expr->column)) {
          resolved = item.expr;
          break;
        }
      }
    }
    if (!resolved) {
      resolved = order.expr;
      ARROW_RETURN_NOT_OK(ResolveExpression(resolved, bindings, true));
    }
    order_expressions.push_back(resolved);
  }

  // Resolve ON conditions and require equality between the two sides.
  std::vector<std::vector<JoinCondition>> join_conditions(statement.joins.size());
  for (size_t i = 0; i < statement.joins.size(); ++i) {
    if (!statement.joins[i].on) {
      return arrow::Status::Invalid("JOIN requires an ON condition");
    }
    sql::ExprPtr on = statement.joins[i].on;
    ARROW_RETURN_NOT_OK(ResolveExpression(on, bindings, false));
    for (const auto& conjunct : SplitConjuncts(on)) {
      if (conjunct->kind != sql::ExprKind::Binary || conjunct->binary_op != sql::BinaryOp::Eq) {
        return arrow::Status::Invalid(
            "join conditions must be equality comparisons combined with AND");
      }
      if (conjunct->left->kind != sql::ExprKind::Column ||
          conjunct->right->kind != sql::ExprKind::Column) {
        return arrow::Status::Invalid("join conditions must compare two plain columns");
      }
      const std::string right_alias = bindings[i + 1].alias;
      sql::ExprPtr left = conjunct->left;
      sql::ExprPtr right = conjunct->right;
      const bool left_is_right = IEquals(TableOf(*left), right_alias);
      const bool right_is_right = IEquals(TableOf(*right), right_alias);
      if (left_is_right == right_is_right) {
        return arrow::Status::Invalid(
            "each join condition must compare a column from the new table with a column from a "
            "previous table");
      }
      if (left_is_right)
        std::swap(left, right);
      join_conditions[i].push_back(JoinCondition{left, right});
    }
    if (join_conditions[i].empty()) {
      return arrow::Status::Invalid("JOIN requires at least one equality condition");
    }
  }

  const bool has_aggregate =
      !group_by.empty() || statement.having != nullptr ||
      std::any_of(items.begin(), items.end(),
                  [](const ResolvedItem& item) { return ContainsAggregate(item.expr); }) ||
      std::any_of(order_expressions.begin(), order_expressions.end(),
                  [](const sql::ExprPtr& expr) { return ContainsAggregate(expr); });

  std::vector<sql::ExprPtr> aggregates;
  std::set<std::string> aggregate_canonicals;
  if (has_aggregate) {
    for (const auto& item : items) {
      ARROW_RETURN_NOT_OK(ValidateGroupedExpression(item.expr, group_canonicals));
    }
    for (const auto& expr : order_expressions) {
      if (expr)
        ARROW_RETURN_NOT_OK(ValidateGroupedExpression(expr, group_canonicals));
    }
    if (statement.having) {
      ARROW_RETURN_NOT_OK(ValidateHavingExpression(statement.having, group_canonicals));
    }
    for (const auto& item : items)
      CollectAggregates(item.expr, aggregates);
    for (const auto& expr : order_expressions)
      CollectAggregates(expr, aggregates);
    if (statement.having)
      CollectAggregates(statement.having, aggregates);
    for (const auto& aggregate : aggregates) {
      ARROW_RETURN_NOT_OK(ValidateAggregate(*aggregate));
      aggregate_canonicals.insert(CanonicalExpression(*aggregate));
    }
  }

  // Build the FROM / JOIN tree.
  std::shared_ptr<LogicalPlan> current = bindings[0].scan;
  for (size_t i = 0; i < statement.joins.size(); ++i) {
    auto right = bindings[i + 1].scan;
    auto join = std::make_shared<LogicalPlan>();
    join->kind = LogicalPlan::Kind::Join;
    join->children = {current, right};
    join->join_conditions = join_conditions[i];
    join->left_outer = statement.joins[i].left_outer;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& child : join->children) {
      for (int j = 0; j < child->schema->num_fields(); ++j) {
        fields.push_back(child->schema->field(j));
      }
    }
    join->schema = arrow::schema(std::move(fields));
    join->estimated_rows =
        std::min<int64_t>(current->estimated_rows * right->estimated_rows, 1000000000000LL);
    if (join->left_outer) {
      join->estimated_rows = std::max(join->estimated_rows, current->estimated_rows);
    }
    current = join;
  }

  if (statement.where) {
    auto filter = std::make_shared<LogicalPlan>();
    filter->kind = LogicalPlan::Kind::Filter;
    filter->children = {current};
    filter->predicate = statement.where;
    filter->schema = current->schema;
    current = filter;
  }

  if (has_aggregate) {
    auto aggregate = std::make_shared<LogicalPlan>();
    aggregate->kind = LogicalPlan::Kind::Aggregate;
    aggregate->children = {current};
    aggregate->group_by = group_by;
    aggregate->aggregates = aggregates;

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& expr : group_by) {
      std::string name = CanonicalExpression(*expr);
      ARROW_ASSIGN_OR_RAISE(auto type, InferType(*expr, current->schema));
      fields.push_back(arrow::field(name, type));
    }
    for (const auto& expr : aggregates) {
      std::string name = CanonicalExpression(*expr);
      ARROW_ASSIGN_OR_RAISE(auto type, AggregateResultType(*expr, *current->schema));
      aggregate->aggregate_names.push_back(name);
      fields.push_back(arrow::field(name, type));
    }
    aggregate->schema = arrow::schema(std::move(fields));
    aggregate->estimated_rows =
        group_by.empty() ? 1 : std::min<int64_t>(current->estimated_rows, 1 << 20);
    current = aggregate;

    if (statement.having) {
      auto predicate =
          RewriteGroupedExpressionDeep(statement.having, aggregate_canonicals, group_canonicals);
      if (!predicate) {
        return arrow::Status::Invalid("HAVING expression " +
                                      CanonicalExpression(*statement.having) +
                                      " must be a combination of group columns and aggregates");
      }
      auto filter = std::make_shared<LogicalPlan>();
      filter->kind = LogicalPlan::Kind::Filter;
      filter->children = {current};
      filter->predicate = std::move(predicate);
      filter->schema = current->schema;
      current = filter;
    }
  }

  if (statement.distinct) {
    auto project = std::make_shared<LogicalPlan>();
    project->kind = LogicalPlan::Kind::Project;
    project->children = {current};
    std::vector<std::shared_ptr<arrow::Field>> output_fields;
    for (const auto& item : items) {
      sql::ExprPtr expr = item.expr;
      if (has_aggregate) {
        expr = RewriteGroupedExpression(item.expr, aggregate_canonicals, group_canonicals);
        if (!expr) {
          return arrow::Status::Invalid("select expression " + CanonicalExpression(*item.expr) +
                                        " is not valid in a grouped query");
        }
      }
      ARROW_ASSIGN_OR_RAISE(auto type, InferType(*expr, current->schema));
      project->projections.push_back(expr);
      project->output_names.push_back(item.alias);
      output_fields.push_back(arrow::field(item.alias, type));
    }
    project->schema = arrow::schema(std::move(output_fields));

    auto distinct = std::make_shared<LogicalPlan>();
    distinct->kind = LogicalPlan::Kind::Distinct;
    distinct->children = {project};
    distinct->schema = project->schema;
    current = distinct;

    // With DISTINCT, ORDER BY must use expressions from the select list (or
    // ordinals/aliases), because duplicate removal happens before ordering.
    std::vector<SortKey> sort_keys;
    for (size_t i = 0; i < statement.order_by.size(); ++i) {
      const auto& order = statement.order_by[i];
      std::string alias;
      if (order.ordinal) {
        if (order.ordinal < 1 || static_cast<size_t>(order.ordinal) > items.size()) {
          return arrow::Status::Invalid("ORDER BY position " + std::to_string(order.ordinal) +
                                        " is out of range");
        }
        alias = items[static_cast<size_t>(order.ordinal - 1)].alias;
      } else {
        const sql::ExprPtr& source = order_expressions[i];
        if (source->kind == sql::ExprKind::Column && source->table.empty()) {
          for (const auto& item : items) {
            if (IEquals(item.alias, source->column)) {
              alias = item.alias;
              break;
            }
          }
        }
        if (alias.empty()) {
          const std::string canonical = CanonicalExpression(*source);
          for (const auto& item : items) {
            if (CanonicalExpression(*item.expr) == canonical) {
              alias = item.alias;
              break;
            }
          }
        }
        if (alias.empty()) {
          return arrow::Status::Invalid(
              "ORDER BY expression " + CanonicalExpression(*source) +
              " must appear in the select list when SELECT DISTINCT is used");
        }
      }
      sort_keys.push_back(SortKey{MakeColumn(alias), order.descending});
    }

    if (!sort_keys.empty()) {
      auto sort = std::make_shared<LogicalPlan>();
      sort->kind = LogicalPlan::Kind::Sort;
      sort->children = {current};
      sort->sort_keys = std::move(sort_keys);
      if (statement.limit >= 0) {
        int64_t hint = statement.limit;
        if (statement.offset > std::numeric_limits<int64_t>::max() - hint) {
          hint = std::numeric_limits<int64_t>::max();
        } else {
          hint += statement.offset;
        }
        sort->limit_hint = hint;
      }
      sort->schema = current->schema;
      current = sort;
    }

    if (statement.limit >= 0 || statement.offset > 0) {
      auto limit = std::make_shared<LogicalPlan>();
      limit->kind = LogicalPlan::Kind::Limit;
      limit->children = {current};
      limit->limit = statement.limit;
      limit->offset = statement.offset;
      limit->schema = current->schema;
      current = limit;
    }

    ARROW_RETURN_NOT_OK(PushdownFilters(current));
    ARROW_RETURN_NOT_OK(PruneColumns(current));
    ARROW_RETURN_NOT_OK(RebuildSchemas(current));

    QueryPlan result;
    result.root = current;
    result.output_schema = current->schema;
    std::ostringstream explain;
    ExplainNode(current, 0, explain);
    result.explain = explain.str();
    return result;
  }

  // ORDER BY is planned before projection so it can reference columns that are
  // not selected. Sort expressions are evaluated against the pre-projection
  // schema.
  std::vector<SortKey> sort_keys;
  if (!statement.order_by.empty()) {
    for (size_t i = 0; i < statement.order_by.size(); ++i) {
      const auto& order = statement.order_by[i];
      sql::ExprPtr source;
      if (order.ordinal) {
        if (order.ordinal < 1 || static_cast<size_t>(order.ordinal) > items.size()) {
          return arrow::Status::Invalid("ORDER BY position " + std::to_string(order.ordinal) +
                                        " is out of range");
        }
        source = items[static_cast<size_t>(order.ordinal - 1)].expr;
      } else {
        source = order_expressions[i];
        if (source->kind == sql::ExprKind::Column && source->table.empty()) {
          for (const auto& item : items) {
            if (IEquals(item.alias, source->column)) {
              source = item.expr;
              break;
            }
          }
        }
      }

      if (has_aggregate) {
        auto rewritten = RewriteGroupedExpression(source, aggregate_canonicals, group_canonicals);
        if (!rewritten) {
          return arrow::Status::Invalid(
              "ORDER BY expression " + CanonicalExpression(*source) +
              " must be a group column, an aggregate, or a select alias in a grouped query");
        }
        source = rewritten;
      }
      sort_keys.push_back(SortKey{source, order.descending});
    }

    auto sort = std::make_shared<LogicalPlan>();
    sort->kind = LogicalPlan::Kind::Sort;
    sort->children = {current};
    sort->sort_keys = sort_keys;
    if (statement.limit >= 0) {
      int64_t hint = statement.limit;
      if (statement.offset > std::numeric_limits<int64_t>::max() - hint) {
        hint = std::numeric_limits<int64_t>::max();
      } else {
        hint += statement.offset;
      }
      sort->limit_hint = hint;
    }
    sort->schema = current->schema;
    current = sort;
  }

  auto project = std::make_shared<LogicalPlan>();
  project->kind = LogicalPlan::Kind::Project;
  project->children = {current};
  std::vector<std::shared_ptr<arrow::Field>> output_fields;
  for (const auto& item : items) {
    sql::ExprPtr expr = item.expr;
    if (has_aggregate) {
      expr = RewriteGroupedExpression(item.expr, aggregate_canonicals, group_canonicals);
      if (!expr) {
        return arrow::Status::Invalid("select expression " + CanonicalExpression(*item.expr) +
                                      " is not valid in a grouped query");
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto type, InferType(*expr, current->schema));
    project->projections.push_back(expr);
    project->output_names.push_back(item.alias);
    output_fields.push_back(arrow::field(item.alias, type));
  }
  project->schema = arrow::schema(std::move(output_fields));
  current = project;

  if (statement.limit >= 0 || statement.offset > 0) {
    auto limit = std::make_shared<LogicalPlan>();
    limit->kind = LogicalPlan::Kind::Limit;
    limit->children = {current};
    limit->limit = statement.limit;
    limit->offset = statement.offset;
    limit->schema = current->schema;
    current = limit;
  }

  ARROW_RETURN_NOT_OK(PushdownFilters(current));
  ARROW_RETURN_NOT_OK(PruneColumns(current));
  ARROW_RETURN_NOT_OK(RebuildSchemas(current));

  QueryPlan result;
  result.root = current;
  result.output_schema = current->schema;
  std::ostringstream explain;
  ExplainNode(current, 0, explain);
  result.explain = explain.str();
  return result;
}

} // namespace qe
