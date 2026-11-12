#include "engine/expression.h"

#include <arrow/array/util.h>
#include <arrow/compute/api.h>
#include <arrow/type.h>
#include <arrow/util/checked_cast.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
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

bool IsIntegerType(const arrow::DataType& type) {
  switch (type.id()) {
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
    return true;
  default:
    return false;
  }
}

bool IsFloatType(const arrow::DataType& type) {
  return type.id() == arrow::Type::FLOAT || type.id() == arrow::Type::DOUBLE;
}

bool IsNumericType(const arrow::DataType& type) {
  return IsIntegerType(type) || IsFloatType(type);
}

bool IsStringType(const arrow::DataType& type) {
  return type.id() == arrow::Type::STRING || type.id() == arrow::Type::LARGE_STRING ||
         type.id() == arrow::Type::BINARY;
}

arrow::Result<int32_t> ParseDate(const std::string& text) {
  if (text.size() != 10 || text[4] != '-' || text[7] != '-') {
    return arrow::Status::Invalid("date literal must be YYYY-MM-DD: '" + text + "'");
  }
  int year = 0, month = 0, day = 0;
  try {
    year = std::stoi(text.substr(0, 4));
    month = std::stoi(text.substr(5, 2));
    day = std::stoi(text.substr(8, 2));
  } catch (const std::exception&) {
    return arrow::Status::Invalid("invalid date literal: '" + text + "'");
  }
  if (month < 1 || month > 12 || day < 1 || day > 31) {
    return arrow::Status::Invalid("invalid date literal: '" + text + "'");
  }
  year -= month <= 2;
  const int era = (year >= 0 ? year : year - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(year - era * 400);
  const unsigned doy = (153 * static_cast<unsigned>(month > 2 ? month - 3 : month + 9) + 2) / 5 +
                       static_cast<unsigned>(day) - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  const int64_t days = static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
  return static_cast<int32_t>(days);
}

std::shared_ptr<arrow::Scalar> NaturalLiteralScalar(const sql::Literal& literal) {
  switch (literal.type) {
  case sql::LiteralType::Null:
    return arrow::MakeNullScalar(arrow::null());
  case sql::LiteralType::Integer:
    return arrow::MakeScalar(literal.integer);
  case sql::LiteralType::Double:
    return arrow::MakeScalar(literal.real);
  case sql::LiteralType::String:
    return arrow::MakeScalar(literal.string);
  case sql::LiteralType::Boolean:
    return arrow::MakeScalar(literal.boolean);
  }
  return arrow::MakeNullScalar(arrow::null());
}

// Arrow 13 cannot cast strings to dates in the compute plane, so a date
// column compared with a string or integer scalar is handled here. The
// planner resolves the column; the other side is always a literal scalar.
arrow::Status ConvertScalarToDate(arrow::Datum* datum) {
  if (!datum->is_scalar())
    return arrow::Status::OK();
  auto& scalar = *datum->scalar();
  if (scalar.type->id() == arrow::Type::DATE32)
    return arrow::Status::OK();
  if (!scalar.is_valid) {
    *datum = arrow::Datum(arrow::MakeNullScalar(arrow::date32()));
    return arrow::Status::OK();
  }

  int64_t days = 0;
  switch (scalar.type->id()) {
  case arrow::Type::STRING: {
    ARROW_ASSIGN_OR_RAISE(
        auto parsed,
        ParseDate(
            arrow::internal::checked_cast<const arrow::StringScalar&>(scalar).value->ToString()));
    ARROW_ASSIGN_OR_RAISE(auto date_scalar, arrow::MakeScalar(arrow::date32(), parsed));
    *datum = arrow::Datum(std::move(date_scalar));
    return arrow::Status::OK();
  }
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64: {
    switch (scalar.type->id()) {
    case arrow::Type::INT8:
      days = arrow::internal::checked_cast<const arrow::Int8Scalar&>(scalar).value;
      break;
    case arrow::Type::INT16:
      days = arrow::internal::checked_cast<const arrow::Int16Scalar&>(scalar).value;
      break;
    case arrow::Type::INT32:
      days = arrow::internal::checked_cast<const arrow::Int32Scalar&>(scalar).value;
      break;
    default:
      days = arrow::internal::checked_cast<const arrow::Int64Scalar&>(scalar).value;
      break;
    }
    break;
  }
  default:
    return arrow::Status::OK();
  }
  if (days < std::numeric_limits<int32_t>::min() || days > std::numeric_limits<int32_t>::max()) {
    return arrow::Status::Invalid("date literal is outside the date32 range");
  }
  ARROW_ASSIGN_OR_RAISE(auto date_scalar,
                        arrow::MakeScalar(arrow::date32(), static_cast<int32_t>(days)));
  *datum = arrow::Datum(std::move(date_scalar));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Scalar>>
ConvertLiteralScalar(const std::shared_ptr<arrow::Scalar>& scalar,
                     const std::shared_ptr<arrow::DataType>& target) {
  if (scalar->type->Equals(target))
    return scalar;
  return scalar->CastTo(target);
}

arrow::Result<arrow::Datum> CastDatum(const arrow::Datum& datum,
                                      const std::shared_ptr<arrow::DataType>& target) {
  if (datum.type()->Equals(target))
    return datum;
  return arrow::compute::Cast(datum, target);
}

arrow::Status CoercePair(arrow::Datum* left, arrow::Datum* right, bool for_division = false) {
  const auto& left_type = (*left).type();
  const auto& right_type = (*right).type();

  if (left_type->Equals(right_type) && !(for_division && !IsFloatType(*left_type))) {
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::DataType> target;

  if (left_type->id() == arrow::Type::NA) {
    target = right_type;
  } else if (right_type->id() == arrow::Type::NA) {
    target = left_type;
  } else if (left_type->id() == arrow::Type::DATE32 && IsStringType(*right_type)) {
    target = arrow::date32();
  } else if (right_type->id() == arrow::Type::DATE32 && IsStringType(*left_type)) {
    target = arrow::date32();
  } else if (IsNumericType(*left_type) && IsNumericType(*right_type)) {
    if (for_division || IsFloatType(*left_type) || IsFloatType(*right_type)) {
      target = arrow::float64();
    } else {
      target = arrow::int64();
    }
  } else if (IsStringType(*left_type) && IsStringType(*right_type)) {
    target = arrow::utf8();
  } else if (left_type->id() == arrow::Type::BOOL && right_type->id() == arrow::Type::BOOL) {
    target = arrow::boolean();
  }

  if (!target) {
    return arrow::Status::Invalid("cannot combine values of type " + left_type->ToString() +
                                  " and " + right_type->ToString());
  }

  ARROW_ASSIGN_OR_RAISE(*left, CastDatum(*left, target));
  ARROW_ASSIGN_OR_RAISE(*right, CastDatum(*right, target));
  return arrow::Status::OK();
}

// Arrow 13 exposes a LIKE kernel in libarrow, but not its options type in the
// public headers, so the pattern is translated into an anchored regex and run
// through match_substring_regex instead. `%` is "any run", `_` is "one
// character", and every regex metacharacter is escaped so a pattern can only
// mean what SQL LIKE means.
std::string LikePatternToRegex(std::string_view pattern) {
  std::string regex = "^";
  regex.reserve(pattern.size() * 2 + 2);
  for (char c : pattern) {
    switch (c) {
    case '%':
      regex += ".*";
      break;
    case '_':
      regex += '.';
      break;
    case '.':
    case '^':
    case '$':
    case '*':
    case '+':
    case '?':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case '|':
    case '\\':
      regex += '\\';
      regex += c;
      break;
    default:
      regex += c;
      break;
    }
  }
  regex += '$';
  return regex;
}

arrow::Result<arrow::Datum> EvaluateInternal(const sql::Expr& expr,
                                             const arrow::RecordBatch& batch) {
  switch (expr.kind) {
  case sql::ExprKind::Column: {
    ARROW_ASSIGN_OR_RAISE(std::string name, ResolveColumnName(*batch.schema(), expr));
    int index = batch.schema()->GetFieldIndex(name);
    if (index < 0) {
      return arrow::Status::Invalid("column '" + name + "' not found in batch");
    }
    return arrow::Datum(batch.column(index));
  }
  case sql::ExprKind::Literal:
    return arrow::Datum(NaturalLiteralScalar(expr.literal));
  case sql::ExprKind::Star:
    return arrow::Status::Invalid("star expression reached the executor");
  case sql::ExprKind::Aggregate:
    return arrow::Status::Invalid("aggregate '" + expr.function +
                                  "' cannot be evaluated outside an aggregate operator");
  case sql::ExprKind::IsNull: {
    ARROW_ASSIGN_OR_RAISE(auto value, EvaluateInternal(*expr.left, batch));
    return arrow::compute::CallFunction("is_null", {value});
  }
  case sql::ExprKind::Unary: {
    ARROW_ASSIGN_OR_RAISE(auto value, EvaluateInternal(*expr.left, batch));
    if (expr.unary_op == sql::UnaryOp::Not) {
      if (value.type()->id() != arrow::Type::BOOL) {
        return arrow::Status::Invalid("NOT requires a boolean expression");
      }
      return arrow::compute::CallFunction("invert", {value});
    }
    if (!IsNumericType(*value.type())) {
      return arrow::Status::Invalid("unary minus requires a numeric expression");
    }
    arrow::Datum minus_one(arrow::MakeScalar(static_cast<int64_t>(-1)));
    ARROW_ASSIGN_OR_RAISE(auto multiplier, CastDatum(minus_one, value.type()));
    return arrow::compute::CallFunction("multiply", {value, multiplier});
  }
  case sql::ExprKind::Binary: {
    ARROW_ASSIGN_OR_RAISE(auto left, EvaluateInternal(*expr.left, batch));
    ARROW_ASSIGN_OR_RAISE(auto right, EvaluateInternal(*expr.right, batch));

    switch (expr.binary_op) {
    case sql::BinaryOp::And:
    case sql::BinaryOp::Or: {
      if (left.type()->id() != arrow::Type::BOOL || right.type()->id() != arrow::Type::BOOL) {
        return arrow::Status::Invalid("AND/OR require boolean expressions");
      }
      return arrow::compute::CallFunction(
          expr.binary_op == sql::BinaryOp::And ? "and_kleene" : "or_kleene", {left, right});
    }
    case sql::BinaryOp::Eq:
    case sql::BinaryOp::NotEq:
    case sql::BinaryOp::Lt:
    case sql::BinaryOp::LtEq:
    case sql::BinaryOp::Gt:
    case sql::BinaryOp::GtEq: {
      if (left.type()->id() == arrow::Type::DATE32 && right.type()->id() != arrow::Type::DATE32) {
        ARROW_RETURN_NOT_OK(ConvertScalarToDate(&right));
      }
      if (right.type()->id() == arrow::Type::DATE32 && left.type()->id() != arrow::Type::DATE32) {
        ARROW_RETURN_NOT_OK(ConvertScalarToDate(&left));
      }
      ARROW_RETURN_NOT_OK(CoercePair(&left, &right));
      const char* function = nullptr;
      switch (expr.binary_op) {
      case sql::BinaryOp::Eq:
        function = "equal";
        break;
      case sql::BinaryOp::NotEq:
        function = "not_equal";
        break;
      case sql::BinaryOp::Lt:
        function = "less";
        break;
      case sql::BinaryOp::LtEq:
        function = "less_equal";
        break;
      case sql::BinaryOp::Gt:
        function = "greater";
        break;
      case sql::BinaryOp::GtEq:
        function = "greater_equal";
        break;
      default:
        break;
      }
      return arrow::compute::CallFunction(function, {left, right});
    }
    case sql::BinaryOp::Like: {
      if (!IsStringType(*left.type())) {
        return arrow::Status::Invalid("LIKE requires a string expression, got " +
                                      left.type()->ToString());
      }
      // A NULL pattern makes the whole comparison NULL, exactly like any other
      // comparison with NULL.
      if (right.type()->id() == arrow::Type::NA) {
        return arrow::Datum(arrow::MakeNullScalar(arrow::boolean()));
      }
      if (!right.is_scalar() || !IsStringType(*right.type())) {
        return arrow::Status::Invalid(
            "LIKE requires a string literal pattern; a column or expression pattern is not "
            "supported");
      }
      const auto& pattern_scalar = *right.scalar();
      std::string pattern;
      if (pattern_scalar.is_valid) {
        pattern = arrow::internal::checked_cast<const arrow::StringScalar&>(pattern_scalar)
                      .value->ToString();
        // Arrow's match_like treats a backslash as an escape character while
        // DuckDB's LIKE does not, so a pattern containing one would silently
        // change meaning between the two engines.
        if (pattern.find('\\') != std::string::npos) {
          return arrow::Status::Invalid(
              "LIKE patterns with backslash escapes are not supported; remove the escape "
              "character from the pattern");
        }
      }
      ARROW_RETURN_NOT_OK(CoercePair(&left, &right));
      arrow::compute::MatchSubstringOptions options(LikePatternToRegex(pattern));
      return arrow::compute::CallFunction("match_substring_regex", {left}, &options);
    }
    case sql::BinaryOp::Add:
    case sql::BinaryOp::Sub:
    case sql::BinaryOp::Mul:
    case sql::BinaryOp::Div: {
      if (!IsNumericType(*left.type()) || !IsNumericType(*right.type())) {
        return arrow::Status::Invalid("arithmetic requires numeric expressions");
      }
      ARROW_RETURN_NOT_OK(CoercePair(&left, &right, expr.binary_op == sql::BinaryOp::Div));
      const char* function = nullptr;
      switch (expr.binary_op) {
      case sql::BinaryOp::Add:
        function = "add";
        break;
      case sql::BinaryOp::Sub:
        function = "subtract";
        break;
      case sql::BinaryOp::Mul:
        function = "multiply";
        break;
      case sql::BinaryOp::Div:
        function = "divide";
        break;
      default:
        break;
      }
      return arrow::compute::CallFunction(function, {left, right});
    }
    }
    return arrow::Status::Invalid("unsupported binary operator");
  }
  }
  return arrow::Status::Invalid("unsupported expression");
}

} // namespace

arrow::Result<std::string> ResolveColumnName(const arrow::Schema& schema, const sql::Expr& expr) {
  if (expr.kind != sql::ExprKind::Column) {
    return arrow::Status::Invalid("expression is not a column reference");
  }
  const std::string column = ToLower(expr.column);
  const std::string qualified =
      expr.table.empty() ? std::string() : ToLower(expr.table) + "." + column;

  std::vector<std::string> exact;
  std::vector<std::string> suffix;
  for (int i = 0; i < schema.num_fields(); ++i) {
    const std::string field = ToLower(schema.field(i)->name());
    if (!qualified.empty()) {
      if (field == qualified)
        exact.push_back(schema.field(i)->name());
    } else {
      if (field == column) {
        exact.push_back(schema.field(i)->name());
      } else if (field.size() > column.size() &&
                 field.compare(field.size() - column.size(), column.size(), column) == 0 &&
                 field[field.size() - column.size() - 1] == '.') {
        suffix.push_back(schema.field(i)->name());
      }
    }
  }

  const auto& matches = !exact.empty() ? exact : suffix;
  if (matches.empty()) {
    std::ostringstream out;
    out << "column '" << (expr.table.empty() ? "" : expr.table + ".") << expr.column
        << "' does not exist; available columns: ";
    for (int i = 0; i < schema.num_fields(); ++i) {
      if (i)
        out << ", ";
      out << schema.field(i)->name();
    }
    return arrow::Status::Invalid(out.str());
  }
  if (matches.size() > 1) {
    std::ostringstream out;
    out << "column reference '" << (expr.table.empty() ? "" : expr.table + ".") << expr.column
        << "' is ambiguous; qualify it with a table alias. Candidates: ";
    for (size_t i = 0; i < matches.size(); ++i) {
      if (i)
        out << ", ";
      out << matches[i];
    }
    return arrow::Status::Invalid(out.str());
  }
  return matches[0];
}

arrow::Result<arrow::Datum> EvaluateExpression(const sql::Expr& expr,
                                               const arrow::RecordBatch& batch) {
  return EvaluateInternal(expr, batch);
}

arrow::Result<std::shared_ptr<arrow::Array>>
EvaluateExpressionArray(const sql::Expr& expr, const arrow::RecordBatch& batch) {
  ARROW_ASSIGN_OR_RAISE(auto value, EvaluateExpression(expr, batch));
  if (value.is_array())
    return value.make_array();
  if (value.is_scalar()) {
    return arrow::MakeArrayFromScalar(*value.scalar(), batch.num_rows());
  }
  return arrow::Status::Invalid("expression did not produce an array");
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ApplyFilter(const arrow::RecordBatch& batch,
                                                               const sql::Expr& predicate) {
  ARROW_ASSIGN_OR_RAISE(auto value, EvaluateExpression(predicate, batch));
  if (value.type()->id() != arrow::Type::BOOL) {
    return arrow::Status::Invalid("WHERE requires a boolean expression, got " +
                                  value.type()->ToString());
  }

  if (value.is_scalar()) {
    const auto& scalar =
        arrow::internal::checked_cast<const arrow::BooleanScalar&>(*value.scalar());
    if (!scalar.is_valid || !scalar.value)
      return batch.Slice(0, 0);
    return arrow::RecordBatch::Make(batch.schema(), batch.num_rows(), batch.columns());
  }

  auto predicate_array = value.make_array();
  const auto& boolean = arrow::internal::checked_cast<const arrow::BooleanArray&>(*predicate_array);

  arrow::Int32Builder indices_builder;
  ARROW_RETURN_NOT_OK(indices_builder.Reserve(boolean.length()));
  int64_t selected = 0;
  for (int64_t i = 0; i < boolean.length(); ++i) {
    if (boolean.IsValid(i) && boolean.Value(i)) {
      indices_builder.UnsafeAppend(static_cast<int32_t>(i));
      ++selected;
    }
  }
  if (selected == boolean.length()) {
    return arrow::RecordBatch::Make(batch.schema(), batch.num_rows(), batch.columns());
  }
  if (selected == 0)
    return batch.Slice(0, 0);

  ARROW_ASSIGN_OR_RAISE(auto indices, indices_builder.Finish());
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(batch.num_columns());
  for (int i = 0; i < batch.num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto taken, arrow::compute::Take(*batch.column(i), *indices));
    columns.push_back(std::move(taken));
  }
  return arrow::RecordBatch::Make(batch.schema(), selected, std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ApplyProjection(const arrow::RecordBatch& batch, const std::vector<sql::ExprPtr>& expressions,
                const std::vector<std::string>& output_names) {
  if (expressions.size() != output_names.size()) {
    return arrow::Status::Invalid("projection expression/name count mismatch");
  }
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> columns;
  fields.reserve(expressions.size());
  columns.reserve(expressions.size());
  for (size_t i = 0; i < expressions.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto column, EvaluateExpressionArray(*expressions[i], batch));
    fields.push_back(arrow::field(output_names[i], column->type()));
    columns.push_back(std::move(column));
  }
  return arrow::RecordBatch::Make(arrow::schema(std::move(fields)), batch.num_rows(),
                                  std::move(columns));
}

bool ContainsAggregate(const sql::ExprPtr& expr) {
  if (!expr)
    return false;
  if (expr->kind == sql::ExprKind::Aggregate)
    return true;
  if (ContainsAggregate(expr->left) || ContainsAggregate(expr->right))
    return true;
  if (ContainsAggregate(expr->argument))
    return true;
  return false;
}

void CollectColumnRefs(const sql::ExprPtr& expr, std::set<std::string>& out) {
  if (!expr)
    return;
  if (expr->kind == sql::ExprKind::Column) {
    // Plan schemas qualify fields as "<alias>.<column>". Filter pushdown and
    // pruning compare these keys case-insensitively, so normalize them once.
    out.insert(ToLower(expr->table.empty() ? expr->column : expr->table + "." + expr->column));
    return;
  }
  CollectColumnRefs(expr->left, out);
  CollectColumnRefs(expr->right, out);
  CollectColumnRefs(expr->argument, out);
}

std::string CanonicalExpression(const sql::Expr& expr) {
  return sql::ExprToString(std::make_shared<sql::Expr>(expr));
}

bool ExpressionsEqual(const sql::Expr& left, const sql::Expr& right) {
  return CanonicalExpression(left) == CanonicalExpression(right);
}

arrow::Result<std::shared_ptr<arrow::Scalar>> LiteralToScalar(const sql::Literal& literal,
                                                              const arrow::DataType& type) {
  switch (literal.type) {
  case sql::LiteralType::Null:
    return arrow::MakeNullScalar(type.GetSharedPtr());
  case sql::LiteralType::Integer: {
    if (type.id() == arrow::Type::DATE32) {
      return arrow::MakeScalar(arrow::date32(), static_cast<int32_t>(literal.integer));
    }
    auto scalar = arrow::MakeScalar(literal.integer);
    return ConvertLiteralScalar(scalar, type.GetSharedPtr());
  }
  case sql::LiteralType::Double: {
    auto scalar = arrow::MakeScalar(literal.real);
    return ConvertLiteralScalar(scalar, type.GetSharedPtr());
  }
  case sql::LiteralType::String: {
    if (type.id() == arrow::Type::DATE32) {
      ARROW_ASSIGN_OR_RAISE(auto days, ParseDate(literal.string));
      return arrow::MakeScalar(arrow::date32(), days);
    }
    auto scalar = arrow::MakeScalar(literal.string);
    return ConvertLiteralScalar(scalar, type.GetSharedPtr());
  }
  case sql::LiteralType::Boolean: {
    auto scalar = arrow::MakeScalar(literal.boolean);
    return ConvertLiteralScalar(scalar, type.GetSharedPtr());
  }
  }
  return arrow::Status::Invalid("unsupported literal");
}

} // namespace qe
