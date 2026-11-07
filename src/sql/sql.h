#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qe::sql {

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

enum class ExprKind {
  Column,
  Literal,
  Binary,
  Unary,
  IsNull,
  Aggregate,
  Star,
};

enum class LiteralType { Null, Integer, Double, String, Boolean };

struct Literal {
  LiteralType type = LiteralType::Null;
  int64_t integer = 0;
  double real = 0.0;
  std::string string;
  bool boolean = false;
};

enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  NotEq,
  Lt,
  LtEq,
  Gt,
  GtEq,
  And,
  Or,
};

enum class UnaryOp { Negate, Not };

struct Expr {
  ExprKind kind = ExprKind::Literal;

  // Column
  std::string column;
  std::string table;

  // Literal
  Literal literal;

  // Binary / unary
  BinaryOp binary_op = BinaryOp::Eq;
  UnaryOp unary_op = UnaryOp::Not;
  ExprPtr left;
  ExprPtr right;

  // Aggregate
  std::string function;
  bool distinct = false;
  bool star = false;
  ExprPtr argument;
};

struct SelectItem {
  ExprPtr expr;
  std::string alias;
  bool star = false;
  std::string star_table; // empty means all tables
};

struct TableRef {
  std::string name;
  std::string alias; // defaults to name
};

struct JoinClause {
  TableRef table;
  ExprPtr on;
  bool left_outer = false;
};

struct OrderItem {
  ExprPtr expr;
  bool descending = false;
  // ORDER BY <ordinal> is stored as a 1-based ordinal and expr is null.
  int ordinal = 0;
};

struct SelectStmt {
  std::vector<SelectItem> items;
  bool distinct = false;
  TableRef from;
  std::vector<JoinClause> joins;
  ExprPtr where;
  ExprPtr having;
  std::vector<ExprPtr> group_by;
  std::vector<OrderItem> order_by;
  int64_t limit = -1;
  int64_t offset = 0;
};

// Thrown for any lexing, parsing, or semantic error in the SQL subset.
struct SqlError : std::runtime_error {
  explicit SqlError(std::string message, size_t position = 0)
      : std::runtime_error(std::move(message)), position(position) {}
  size_t position;
};

class Lexer {
public:
  explicit Lexer(std::string input);
  struct Token {
    enum class Type { Identifier, Number, String, Symbol, Keyword, End };
    Type type = Type::End;
    std::string text;
    size_t position = 0;
    bool is_keyword = false;
  };
  Token Next();
  const std::string& input() const {
    return input_;
  }

private:
  char Peek(size_t ahead = 0) const;
  char Advance();
  void SkipWhitespaceAndComments();

  std::string input_;
  size_t position_ = 0;
};

class Parser {
public:
  explicit Parser(std::string input);
  SelectStmt Parse();

  // Exposed for unit tests: parses a standalone expression (used by ORDER BY tests).
  ExprPtr ParseExpressionText(const std::string& text);

private:
  using Token = Lexer::Token;
  using TokenType = Token::Type;

  const Token& Peek(size_t ahead = 0);
  Token Consume();
  bool Match(TokenType type, std::string_view text = {});
  bool MatchKeyword(std::string_view keyword);
  Token Expect(TokenType type, std::string_view what);
  Token ExpectIdentifier(std::string_view what);
  void ExpectEnd();

  SelectStmt ParseSelect();
  SelectItem ParseSelectItem();
  TableRef ParseTableRef();
  JoinClause ParseJoin();
  ExprPtr ParseExpression();
  ExprPtr ParseOr();
  ExprPtr ParseAnd();
  ExprPtr ParseNot();
  ExprPtr ParseComparison();
  ExprPtr ParseAdditive();
  ExprPtr ParseMultiplicative();
  ExprPtr ParseUnary();
  ExprPtr ParsePrimary();
  ExprPtr ParseFunctionOrColumn();
  std::vector<ExprPtr> ParseExpressionList();
  OrderItem ParseOrderItem();

  Literal ParseNumberLiteral();
  [[noreturn]] void Error(const std::string& message, size_t position) const;

  std::string input_;
  Lexer lexer_;
  std::vector<Token> tokens_;
  size_t cursor_ = 0;
  bool tokenized_ = false;
};

std::string ExprToString(const ExprPtr& expr);
std::string StatementToString(const SelectStmt& stmt);

} // namespace qe::sql
