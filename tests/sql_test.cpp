#include "sql/sql.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void ExpectSqlError(const std::string& sql, const std::string& fragment) {
  try {
    qe::sql::Parser parser(sql);
    parser.Parse();
    std::cerr << "FAIL: expected error for: " << sql << "\n";
    ++failures;
  } catch (const qe::sql::SqlError& error) {
    if (std::string(error.what()).find(fragment) == std::string::npos) {
      std::cerr << "FAIL: error for '" << sql << "' did not contain '" << fragment
                << "': " << error.what() << "\n";
      ++failures;
    }
  }
}

void TestBasicSelect() {
  qe::sql::Parser parser("SELECT region, SUM(amount) AS total FROM sales WHERE amount > 10 "
                         "GROUP BY region ORDER BY total DESC LIMIT 5;");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.items.size() == 2, "two select items");
  Check(stmt.items[0].expr->kind == qe::sql::ExprKind::Column, "first item is column");
  Check(stmt.items[0].expr->column == "region", "column name");
  Check(stmt.items[1].expr->kind == qe::sql::ExprKind::Aggregate, "second item aggregate");
  Check(stmt.items[1].expr->function == "sum", "sum function");
  Check(stmt.items[1].alias == "total", "alias");
  Check(stmt.from.name == "sales", "from table");
  Check(stmt.where != nullptr, "where");
  Check(stmt.where->kind == qe::sql::ExprKind::Binary, "where binary");
  Check(stmt.where->binary_op == qe::sql::BinaryOp::Gt, "where >");
  Check(stmt.group_by.size() == 1, "group by count");
  Check(stmt.order_by.size() == 1, "order by count");
  Check(stmt.order_by[0].descending, "descending");
  Check(stmt.limit == 5, "limit");
}

void TestStarAndJoin() {
  qe::sql::Parser parser(
      "SELECT s.*, c.name FROM sales AS s JOIN customers c ON s.customer_id = c.customer_id");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.items.size() == 2, "two items");
  Check(stmt.items[0].star && stmt.items[0].star_table == "s", "qualified star");
  Check(stmt.items[1].expr->table == "c" && stmt.items[1].expr->column == "name",
        "qualified column");
  Check(stmt.joins.size() == 1, "one join");
  Check(stmt.joins[0].table.alias == "c", "join alias");
  Check(stmt.joins[0].on->kind == qe::sql::ExprKind::Binary, "join on");
}

void TestCountStar() {
  qe::sql::Parser parser("SELECT COUNT(*), COUNT(amount), AVG(amount) FROM sales");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.items.size() == 3, "three aggregates");
  Check(stmt.items[0].expr->star, "count star");
  Check(!stmt.items[1].expr->star, "count column");
  Check(stmt.items[2].expr->function == "avg", "avg function");
}

void TestPrecedence() {
  qe::sql::Parser parser("");
  auto expr = parser.ParseExpressionText("a = 1 OR b = 2 AND c = 3");
  Check(expr->kind == qe::sql::ExprKind::Binary, "top is binary");
  Check(expr->binary_op == qe::sql::BinaryOp::Or, "top is OR");
  Check(expr->right->binary_op == qe::sql::BinaryOp::And, "AND binds tighter");
}

void TestIsNullAndNot() {
  qe::sql::Parser parser("");
  auto expr = parser.ParseExpressionText("NOT (amount IS NOT NULL)");
  Check(expr->kind == qe::sql::ExprKind::Unary, "not at top");
  Check(expr->unary_op == qe::sql::UnaryOp::Not, "unary not");
  Check(expr->left->kind == qe::sql::ExprKind::Unary, "inner not from IS NOT NULL");
}

void TestQuotedAndString() {
  qe::sql::Parser parser("SELECT \"Weird Name\" AS \"Alias Name\" FROM \"my table\" "
                         "WHERE \"Weird Name\" = 'O''Brien'");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.items[0].expr->column == "Weird Name", "quoted column");
  Check(stmt.items[0].alias == "Alias Name", "quoted alias");
  Check(stmt.from.name == "my table", "quoted table");
  Check(stmt.where->right->literal.string == "O'Brien", "escaped string");
}

void TestArithmetic() {
  qe::sql::Parser parser("");
  auto expr = parser.ParseExpressionText("a + 2 * b >= 10");
  Check(expr->binary_op == qe::sql::BinaryOp::GtEq, "comparison at top");
  Check(expr->left->binary_op == qe::sql::BinaryOp::Add, "addition");
  Check(expr->left->right->binary_op == qe::sql::BinaryOp::Mul, "multiplication binds tighter");
}

void TestDistinctHavingLimitOffset() {
  qe::sql::Parser parser("SELECT DISTINCT region, COUNT(*) AS n FROM sales GROUP BY region "
                         "HAVING COUNT(*) > 1 ORDER BY n DESC LIMIT 2 OFFSET 1");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.distinct, "distinct flag");
  Check(stmt.having != nullptr, "having expression");
  Check(stmt.limit == 2, "limit value");
  Check(stmt.offset == 1, "offset value");

  qe::sql::Parser offset_only("SELECT id FROM sales ORDER BY id OFFSET 4");
  qe::sql::SelectStmt offset_stmt = offset_only.Parse();
  Check(offset_stmt.limit == -1, "offset without limit leaves limit unset");
  Check(offset_stmt.offset == 4, "offset without limit");
}

void TestLeftJoin() {
  qe::sql::Parser parser(
      "SELECT l.id, r.id FROM edge_left l LEFT OUTER JOIN edge_right r ON l.id = r.id");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.joins.size() == 1, "one left join");
  Check(stmt.joins[0].left_outer, "left join flag");
  Check(stmt.joins[0].on != nullptr, "left join condition");
}

void TestInAndBetween() {
  qe::sql::Parser parser("");
  auto in = parser.ParseExpressionText("id IN (1, 2)");
  Check(in->kind == qe::sql::ExprKind::Binary && in->binary_op == qe::sql::BinaryOp::Or,
        "IN expands to OR");
  Check(in->left->binary_op == qe::sql::BinaryOp::Eq, "IN comparison is equality");

  auto not_in = parser.ParseExpressionText("id NOT IN (1, 2)");
  Check(not_in->kind == qe::sql::ExprKind::Unary && not_in->unary_op == qe::sql::UnaryOp::Not,
        "NOT IN wraps the expanded OR in NOT");

  auto between = parser.ParseExpressionText("id BETWEEN 2 AND 4");
  Check(between->binary_op == qe::sql::BinaryOp::And, "BETWEEN expands to AND");
  Check(between->left->binary_op == qe::sql::BinaryOp::GtEq, "BETWEEN lower bound");
  Check(between->right->binary_op == qe::sql::BinaryOp::LtEq, "BETWEEN upper bound");

  auto not_between = parser.ParseExpressionText("id NOT BETWEEN 2 AND 4");
  Check(not_between->binary_op == qe::sql::BinaryOp::Or, "NOT BETWEEN expands to OR");
  Check(not_between->left->binary_op == qe::sql::BinaryOp::Lt, "NOT BETWEEN below range");
  Check(not_between->right->binary_op == qe::sql::BinaryOp::Gt, "NOT BETWEEN above range");
}

void TestLike() {
  qe::sql::Parser parser("");
  auto like = parser.ParseExpressionText("label LIKE 'x%'");
  Check(like->kind == qe::sql::ExprKind::Binary, "LIKE is binary");
  Check(like->binary_op == qe::sql::BinaryOp::Like, "LIKE operator");
  Check(like->left->column == "label", "LIKE left operand");
  Check(like->right->literal.string == "x%", "LIKE pattern literal");

  auto not_like = parser.ParseExpressionText("label NOT LIKE '%x'");
  Check(not_like->kind == qe::sql::ExprKind::Unary && not_like->unary_op == qe::sql::UnaryOp::Not,
        "NOT LIKE wraps LIKE in NOT");
  Check(not_like->left->binary_op == qe::sql::BinaryOp::Like, "NOT LIKE inner operator");

  // A non-literal pattern still parses; the executor rejects it with a message
  // that names the requirement, so the failure is not a vague type error.
  auto column_pattern = parser.ParseExpressionText("label LIKE other");
  Check(column_pattern->binary_op == qe::sql::BinaryOp::Like, "column pattern parses");
}

void TestLikeInsidePredicate() {
  qe::sql::Parser parser("SELECT id FROM t WHERE label LIKE 'a_c' AND id > 2");
  qe::sql::SelectStmt stmt = parser.Parse();
  Check(stmt.where->binary_op == qe::sql::BinaryOp::And, "LIKE contained in AND");
  Check(stmt.where->left->binary_op == qe::sql::BinaryOp::Like, "LIKE conjunct");
}

void TestRejections() {
  ExpectSqlError("SELECT * FROM t RIGHT JOIN u ON t.id = u.id", "RIGHT JOIN");
  ExpectSqlError("SELECT * FROM t FULL JOIN u ON t.id = u.id", "FULL JOIN");
  ExpectSqlError("SELECT * FROM t CROSS JOIN u", "CROSS JOIN");
  ExpectSqlError("SELECT * FROM t NATURAL JOIN u", "NATURAL JOIN");
  ExpectSqlError("SELECT * FROM t JOIN u USING (id)", "USING");
  ExpectSqlError("SELECT * FROM t ORDER BY a NULLS FIRST", "NULLS");
  ExpectSqlError("SELECT * FROM t UNION SELECT * FROM u", "UNION");
  ExpectSqlError("SELECT SUM(*) FROM t", "only COUNT(*)");
  ExpectSqlError("SELECT COUNT(DISTINCT a) FROM t", "DISTINCT");
  ExpectSqlError("SELECT * FROM t WHERE a = b = c", "chained");
  ExpectSqlError("SELECT * FROM t WHERE a IN (SELECT a FROM u)", "subqueries");
  ExpectSqlError("SELECT * FROM t WHERE a IN ()", "IN list");
  ExpectSqlError("SELECT * FROM t WHERE a BETWEEN 1 OR 2", "AND");
  ExpectSqlError("SELECT * FROM t GROUP WHERE a = 1", "BY");
  ExpectSqlError("SELECT * FROM t LEFT WHERE a = 1", "JOIN");
  ExpectSqlError("SELECT * FROM t LIMIT 1.5", "integer");
  ExpectSqlError("SELECT 1", "FROM");
  ExpectSqlError("DELETE FROM t", "read-only");
  ExpectSqlError("SELECT * FROM t; garbage", "unexpected token");
  ExpectSqlError("SELECT \"unterminated FROM t", "unterminated");
  ExpectSqlError("SELECT * FROM t WHERE", "expected an expression");
  ExpectSqlError("SELECT lower(region) FROM sales", "unsupported function");
}

void TestOrderOrdinal() {
  qe::sql::Parser parser("SELECT region, COUNT(*) FROM sales GROUP BY region ORDER BY 2 DESC");
  auto stmt = parser.Parse();
  Check(stmt.order_by.size() == 1 && stmt.order_by[0].ordinal == 2, "order by ordinal");
  Check(stmt.order_by[0].descending, "ordinal descending");
}

} // namespace

int main() {
  TestBasicSelect();
  TestStarAndJoin();
  TestCountStar();
  TestPrecedence();
  TestIsNullAndNot();
  TestQuotedAndString();
  TestArithmetic();
  TestDistinctHavingLimitOffset();
  TestLeftJoin();
  TestInAndBetween();
  TestLike();
  TestLikeInsidePredicate();
  TestRejections();
  TestOrderOrdinal();

  if (failures == 0) {
    std::cout << "sql parser tests passed\n";
    return 0;
  }
  std::cerr << failures << " test(s) failed\n";
  return 1;
}
