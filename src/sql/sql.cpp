#include "sql/sql.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace qe::sql {
namespace {

const std::unordered_set<std::string>& Keywords() {
  static const std::unordered_set<std::string> keywords = {
      "SELECT",  "FROM",  "WHERE",    "GROUP",    "BY",     "ORDER",  "ASC",    "DESC",  "LIMIT",
      "JOIN",    "INNER", "LEFT",     "RIGHT",    "FULL",   "OUTER",  "CROSS",  "ON",    "AS",
      "AND",     "OR",    "NOT",      "IS",       "NULL",   "TRUE",   "FALSE",  "COUNT", "SUM",
      "AVG",     "MIN",   "MAX",      "DISTINCT", "HAVING", "UNION",  "OFFSET", "LIKE",  "IN",
      "BETWEEN", "CASE",  "WHEN",     "THEN",     "ELSE",   "END",    "WITH",   "CAST",  "NULLS",
      "FIRST",   "LAST",  "EXISTS",   "ALL",      "ANY",    "ASOF",   "SEMI",   "ANTI",  "USING",
      "NATURAL", "TOP",   "INSERT",   "UPDATE",   "DELETE", "CREATE", "DROP",   "ALTER", "EXPLAIN",
      "PRAGMA",  "SHOW",  "DESCRIBE",
  };
  return keywords;
}

std::string ToUpper(std::string_view text) {
  std::string result(text);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return result;
}

bool IsIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}
bool IsIdentifierBody(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string QuoteIdentifier(const std::string& name) {
  bool simple = !name.empty() &&
                (IsIdentifierStart(name[0]) || std::isdigit(static_cast<unsigned char>(name[0])));
  for (char c : name) {
    if (!IsIdentifierBody(c)) {
      simple = false;
      break;
    }
  }
  if (simple)
    return name;
  std::string out = "\"";
  for (char c : name) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

std::string QuoteString(const std::string& value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'')
      out += "''";
    else
      out += c;
  }
  out += "'";
  return out;
}

std::string LiteralToString(const Literal& literal) {
  switch (literal.type) {
  case LiteralType::Null:
    return "NULL";
  case LiteralType::Integer:
    return std::to_string(literal.integer);
  case LiteralType::Double: {
    std::ostringstream out;
    out << literal.real;
    return out.str();
  }
  case LiteralType::String:
    return QuoteString(literal.string);
  case LiteralType::Boolean:
    return literal.boolean ? "TRUE" : "FALSE";
  }
  return "?";
}

std::string BinaryOpToString(BinaryOp op) {
  switch (op) {
  case BinaryOp::Add:
    return "+";
  case BinaryOp::Sub:
    return "-";
  case BinaryOp::Mul:
    return "*";
  case BinaryOp::Div:
    return "/";
  case BinaryOp::Eq:
    return "=";
  case BinaryOp::NotEq:
    return "!=";
  case BinaryOp::Lt:
    return "<";
  case BinaryOp::LtEq:
    return "<=";
  case BinaryOp::Gt:
    return ">";
  case BinaryOp::GtEq:
    return ">=";
  case BinaryOp::And:
    return "AND";
  case BinaryOp::Or:
    return "OR";
  }
  return "?";
}

ExprPtr MakeBinary(BinaryOp op, ExprPtr left, ExprPtr right) {
  auto expr = std::make_shared<Expr>();
  expr->kind = ExprKind::Binary;
  expr->binary_op = op;
  expr->left = std::move(left);
  expr->right = std::move(right);
  return expr;
}

ExprPtr MakeNot(ExprPtr expr) {
  auto result = std::make_shared<Expr>();
  result->kind = ExprKind::Unary;
  result->unary_op = UnaryOp::Not;
  result->left = std::move(expr);
  return result;
}

} // namespace

Lexer::Lexer(std::string input) : input_(std::move(input)) {}

char Lexer::Peek(size_t ahead) const {
  if (position_ + ahead >= input_.size())
    return '\0';
  return input_[position_ + ahead];
}

char Lexer::Advance() {
  return input_[position_++];
}

void Lexer::SkipWhitespaceAndComments() {
  while (position_ < input_.size()) {
    char c = input_[position_];
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++position_;
    } else if (c == '-' && Peek(1) == '-') {
      position_ += 2;
      while (position_ < input_.size() && input_[position_] != '\n')
        ++position_;
    } else if (c == '/' && Peek(1) == '*') {
      position_ += 2;
      while (position_ < input_.size() && !(input_[position_] == '*' && Peek(1) == '/')) {
        ++position_;
      }
      if (position_ >= input_.size()) {
        throw SqlError("unterminated block comment", position_);
      }
      position_ += 2;
    } else {
      break;
    }
  }
}

Lexer::Token Lexer::Next() {
  SkipWhitespaceAndComments();
  Token token;
  token.position = position_;
  if (position_ >= input_.size()) {
    token.type = Token::Type::End;
    return token;
  }

  char c = Peek();
  if (IsIdentifierStart(c)) {
    size_t start = position_;
    while (position_ < input_.size() && IsIdentifierBody(input_[position_]))
      ++position_;
    token.text = input_.substr(start, position_ - start);
    std::string upper = ToUpper(token.text);
    if (Keywords().contains(upper)) {
      token.type = Token::Type::Keyword;
      token.text = upper;
      token.is_keyword = true;
    } else {
      token.type = Token::Type::Identifier;
    }
    return token;
  }

  if (c == '"') {
    ++position_;
    std::string value;
    while (position_ < input_.size()) {
      char ch = Advance();
      if (ch == '"') {
        if (Peek() == '"') {
          ++position_;
          value += '"';
          continue;
        }
        token.type = Token::Type::Identifier;
        token.text = std::move(value);
        return token;
      }
      value += ch;
    }
    throw SqlError("unterminated quoted identifier", token.position);
  }

  if (c == '\'') {
    ++position_;
    std::string value;
    while (position_ < input_.size()) {
      char ch = Advance();
      if (ch == '\'') {
        if (Peek() == '\'') {
          ++position_;
          value += '\'';
          continue;
        }
        token.type = Token::Type::String;
        token.text = std::move(value);
        return token;
      }
      value += ch;
    }
    throw SqlError("unterminated string literal", token.position);
  }

  if (std::isdigit(static_cast<unsigned char>(c))) {
    size_t start = position_;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
    if (Peek() == '.' && std::isdigit(static_cast<unsigned char>(Peek(1)))) {
      ++position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
      }
    }
    if (Peek() == 'e' || Peek() == 'E') {
      size_t save = position_;
      ++position_;
      if (Peek() == '+' || Peek() == '-')
        ++position_;
      if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
        position_ = save;
      } else {
        while (position_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[position_]))) {
          ++position_;
        }
      }
    }
    token.type = Token::Type::Number;
    token.text = input_.substr(start, position_ - start);
    return token;
  }

  auto symbol = [&](std::string_view text, size_t width) {
    token.type = Token::Type::Symbol;
    token.text = std::string(text);
    position_ += width;
    return token;
  };
  if (c == '!' && Peek(1) == '=')
    return symbol("!=", 2);
  if (c == '<' && Peek(1) == '>')
    return symbol("<>", 2);
  if (c == '<' && Peek(1) == '=')
    return symbol("<=", 2);
  if (c == '>' && Peek(1) == '=')
    return symbol(">=", 2);
  if (c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' ||
      c == '(' || c == ')' || c == ',' || c == '.' || c == ';') {
    return symbol(std::string(1, c), 1);
  }

  throw SqlError(std::string("unexpected character '") + c + "'", token.position);
}

Parser::Parser(std::string input) : input_(std::move(input)), lexer_(input_) {}

const Parser::Token& Parser::Peek(size_t ahead) {
  if (!tokenized_) {
    tokens_.push_back(lexer_.Next());
    tokenized_ = true;
  }
  while (tokens_.size() <= cursor_ + ahead) {
    tokens_.push_back(lexer_.Next());
  }
  return tokens_[cursor_ + ahead];
}

Parser::Token Parser::Consume() {
  const Token& token = Peek();
  ++cursor_;
  return token;
}

bool Parser::Match(TokenType type, std::string_view text) {
  const Token& token = Peek();
  if (token.type != type)
    return false;
  if (!text.empty() && token.text != text)
    return false;
  ++cursor_;
  return true;
}

bool Parser::MatchKeyword(std::string_view keyword) {
  return Match(TokenType::Keyword, keyword);
}

Parser::Token Parser::Expect(TokenType type, std::string_view what) {
  const Token& token = Peek();
  // Keywords and symbols name the exact token that must appear. Numbers and
  // identifiers use `what` only as a description, so their text is not checked.
  const bool exact_text = type == TokenType::Keyword || type == TokenType::Symbol;
  if (token.type != type || (exact_text && !what.empty() && token.text != what)) {
    Error("expected " + std::string(what) + ", found '" +
              (token.type == TokenType::End ? std::string("<end of query>") : token.text) + "'",
          token.position);
  }
  return Consume();
}

Parser::Token Parser::ExpectIdentifier(std::string_view what) {
  return Expect(TokenType::Identifier, what);
}

void Parser::ExpectEnd() {
  if (Match(TokenType::Symbol, ";")) {
    // trailing semicolon allowed
  }
  const Token& token = Peek();
  if (token.type != TokenType::End) {
    Error("unexpected token '" + token.text + "' after end of query", token.position);
  }
}

void Parser::Error(const std::string& message, size_t position) const {
  size_t line = 1;
  size_t column = 1;
  for (size_t i = 0; i < position && i < input_.size(); ++i) {
    if (input_[i] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
  }
  throw SqlError("line " + std::to_string(line) + ", column " + std::to_string(column) + ": " +
                     message,
                 position);
}

SelectStmt Parser::Parse() {
  if (!MatchKeyword("SELECT")) {
    const Token& token = Peek();
    if (token.type == TokenType::Keyword &&
        (token.text == "INSERT" || token.text == "UPDATE" || token.text == "DELETE")) {
      Error(token.text + " statements are not supported; this engine is read-only", token.position);
    }
    Error("expected SELECT at the start of the query", token.position);
  }
  SelectStmt stmt = ParseSelect();
  ExpectEnd();
  return stmt;
}

SelectStmt Parser::ParseSelect() {
  SelectStmt stmt;

  if (Peek().type == TokenType::Keyword && Peek().text == "DISTINCT") {
    Consume();
    stmt.distinct = true;
  }

  // Select list.
  while (true) {
    stmt.items.push_back(ParseSelectItem());
    if (!Match(TokenType::Symbol, ","))
      break;
  }

  if (!MatchKeyword("FROM")) {
    Error("expected FROM (queries must read at least one Parquet table)", Peek().position);
  }
  stmt.from = ParseTableRef();

  while (true) {
    if (Peek().type != TokenType::Keyword)
      break;
    const std::string& word = Peek().text;
    if (word == "JOIN" || word == "INNER" || word == "LEFT" || word == "RIGHT" || word == "FULL" ||
        word == "CROSS" || word == "NATURAL" || word == "SEMI" || word == "ANTI" ||
        word == "ASOF") {
      stmt.joins.push_back(ParseJoin());
    } else if (word == "OUTER") {
      Error("OUTER JOIN without INNER or LEFT is not supported", Peek().position);
    } else {
      break;
    }
  }

  if (MatchKeyword("WHERE")) {
    stmt.where = ParseExpression();
  }

  if (MatchKeyword("GROUP")) {
    Expect(TokenType::Keyword, "BY");
    stmt.group_by = ParseExpressionList();
  }

  if (MatchKeyword("HAVING")) {
    stmt.having = ParseExpression();
  }

  if (MatchKeyword("ORDER")) {
    Expect(TokenType::Keyword, "BY");
    while (true) {
      stmt.order_by.push_back(ParseOrderItem());
      if (!Match(TokenType::Symbol, ","))
        break;
    }
  }

  auto parse_nonnegative = [&](const char* clause) -> int64_t {
    Token number = Expect(TokenType::Number, std::string("a non-negative integer after ") + clause);
    if (number.text.find_first_of(".eE") != std::string::npos) {
      Error(std::string(clause) + " must be an integer", number.position);
    }
    try {
      int64_t value = std::stoll(number.text);
      if (value < 0)
        Error(std::string(clause) + " must be non-negative", number.position);
      return value;
    } catch (const std::exception&) {
      Error(std::string(clause) + " value is out of range", number.position);
    }
  };

  if (MatchKeyword("LIMIT")) {
    stmt.limit = parse_nonnegative("LIMIT");
  }
  if (MatchKeyword("OFFSET")) {
    stmt.offset = parse_nonnegative("OFFSET");
  }

  if (Peek().type == TokenType::Keyword && (Peek().text == "UNION" || Peek().text == "EXCEPT" ||
                                            Peek().text == "INTERSECT" || Peek().text == "WITH")) {
    Error(Peek().text + " is not supported", Peek().position);
  }

  return stmt;
}

SelectItem Parser::ParseSelectItem() {
  SelectItem item;
  if (Match(TokenType::Symbol, "*")) {
    item.star = true;
    return item;
  }
  if (Peek().type == TokenType::Identifier && Peek(1).type == TokenType::Symbol &&
      Peek(1).text == "." && Peek(2).type == TokenType::Symbol && Peek(2).text == "*") {
    item.star = true;
    item.star_table = Consume().text;
    Consume(); // .
    Consume(); // *
    return item;
  }

  item.expr = ParseExpression();
  if (MatchKeyword("AS")) {
    item.alias = ExpectIdentifier("an alias after AS").text;
  } else if (Peek().type == TokenType::Identifier) {
    item.alias = Consume().text;
  }
  if (item.alias.empty() && item.expr->kind == ExprKind::Column) {
    item.alias = item.expr->column;
  }
  return item;
}

TableRef Parser::ParseTableRef() {
  Token name = ExpectIdentifier("a table name");
  TableRef ref;
  ref.name = name.text;
  ref.alias = name.text;
  if (MatchKeyword("AS")) {
    ref.alias = ExpectIdentifier("an alias after AS").text;
  } else if (Peek().type == TokenType::Identifier) {
    ref.alias = Consume().text;
  }
  return ref;
}

JoinClause Parser::ParseJoin() {
  JoinClause join;
  if (MatchKeyword("LEFT")) {
    join.left_outer = true;
    MatchKeyword("OUTER");
    Expect(TokenType::Keyword, "JOIN");
  } else if (MatchKeyword("INNER")) {
    Expect(TokenType::Keyword, "JOIN");
  } else if (MatchKeyword("JOIN")) {
    // plain JOIN
  } else {
    const Token& token = Peek();
    Error(token.text + " JOIN is not supported (INNER and LEFT joins only)", token.position);
  }
  join.table = ParseTableRef();
  if (MatchKeyword("USING")) {
    Error("JOIN ... USING is not supported; use ON", tokens_[cursor_ - 1].position);
  }
  Expect(TokenType::Keyword, "ON");
  join.on = ParseExpression();
  return join;
}

std::vector<ExprPtr> Parser::ParseExpressionList() {
  std::vector<ExprPtr> expressions;
  while (true) {
    expressions.push_back(ParseExpression());
    if (!Match(TokenType::Symbol, ","))
      break;
  }
  return expressions;
}

OrderItem Parser::ParseOrderItem() {
  OrderItem item;
  if (Peek().type == TokenType::Number && Peek().text.find_first_of(".eE") == std::string::npos) {
    Token number = Consume();
    try {
      item.ordinal = std::stoi(number.text);
    } catch (const std::exception&) {
      Error("ORDER BY position is out of range", number.position);
    }
    if (item.ordinal < 1)
      Error("ORDER BY position must be >= 1", number.position);
  } else {
    item.expr = ParseExpression();
  }
  if (MatchKeyword("ASC")) {
    item.descending = false;
  } else if (MatchKeyword("DESC")) {
    item.descending = true;
  }
  if (MatchKeyword("NULLS")) {
    Error("NULLS FIRST/LAST is not supported", tokens_[cursor_ - 1].position);
  }
  return item;
}

ExprPtr Parser::ParseExpression() {
  return ParseOr();
}

ExprPtr Parser::ParseOr() {
  ExprPtr left = ParseAnd();
  while (MatchKeyword("OR")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Binary;
    expr->binary_op = BinaryOp::Or;
    expr->left = left;
    expr->right = ParseAnd();
    left = expr;
  }
  return left;
}

ExprPtr Parser::ParseAnd() {
  ExprPtr left = ParseNot();
  while (MatchKeyword("AND")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Binary;
    expr->binary_op = BinaryOp::And;
    expr->left = left;
    expr->right = ParseNot();
    left = expr;
  }
  return left;
}

ExprPtr Parser::ParseNot() {
  if (MatchKeyword("NOT")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Unary;
    expr->unary_op = UnaryOp::Not;
    expr->left = ParseNot();
    return expr;
  }
  return ParseComparison();
}

ExprPtr Parser::ParseComparison() {
  ExprPtr left = ParseAdditive();

  const Token& token = Peek();
  if (token.type == TokenType::Symbol &&
      (token.text == "=" || token.text == "!=" || token.text == "<>" || token.text == "<" ||
       token.text == "<=" || token.text == ">" || token.text == ">=")) {
    Consume();
    BinaryOp op = BinaryOp::Eq;
    if (token.text == "=")
      op = BinaryOp::Eq;
    else if (token.text == "!=" || token.text == "<>")
      op = BinaryOp::NotEq;
    else if (token.text == "<")
      op = BinaryOp::Lt;
    else if (token.text == "<=")
      op = BinaryOp::LtEq;
    else if (token.text == ">")
      op = BinaryOp::Gt;
    else
      op = BinaryOp::GtEq;

    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Binary;
    expr->binary_op = op;
    expr->left = left;
    expr->right = ParseAdditive();

    if (Peek().type == TokenType::Symbol &&
        (Peek().text == "=" || Peek().text == "!=" || Peek().text == "<>" || Peek().text == "<" ||
         Peek().text == "<=" || Peek().text == ">" || Peek().text == ">=")) {
      Error("chained comparisons are not supported; use AND", Peek().position);
    }
    return expr;
  }

  if (MatchKeyword("IS")) {
    bool negated = MatchKeyword("NOT");
    if (!MatchKeyword("NULL")) {
      Error("expected NULL after IS", Peek().position);
    }
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::IsNull;
    expr->left = left;
    if (negated) {
      auto not_expr = std::make_shared<Expr>();
      not_expr->kind = ExprKind::Unary;
      not_expr->unary_op = UnaryOp::Not;
      not_expr->left = expr;
      return not_expr;
    }
    return expr;
  }

  bool negated = false;
  if (Peek().type == TokenType::Keyword && Peek().text == "NOT" &&
      Peek(1).type == TokenType::Keyword &&
      (Peek(1).text == "IN" || Peek(1).text == "BETWEEN" || Peek(1).text == "LIKE")) {
    Consume(); // NOT
    negated = true;
  }

  if (MatchKeyword("IN")) {
    if (!Match(TokenType::Symbol, "(")) {
      Error("expected '(' after IN", Peek().position);
    }
    if (Peek().type == TokenType::Keyword && Peek().text == "SELECT") {
      Error("subqueries are not supported", Peek().position);
    }
    if (Match(TokenType::Symbol, ")")) {
      Error("IN list must contain at least one value", tokens_[cursor_ - 1].position);
    }
    ExprPtr result;
    while (true) {
      ExprPtr value = ParseExpression();
      auto comparison = MakeBinary(BinaryOp::Eq, left, std::move(value));
      result = result ? MakeBinary(BinaryOp::Or, std::move(result), std::move(comparison))
                      : std::move(comparison);
      if (!Match(TokenType::Symbol, ","))
        break;
    }
    Expect(TokenType::Symbol, ")");
    return negated ? MakeNot(std::move(result)) : result;
  }

  if (MatchKeyword("BETWEEN")) {
    ExprPtr low = ParseAdditive();
    Expect(TokenType::Keyword, "AND");
    ExprPtr high = ParseAdditive();
    if (negated) {
      auto below = MakeBinary(BinaryOp::Lt, left, std::move(low));
      auto above = MakeBinary(BinaryOp::Gt, left, std::move(high));
      return MakeBinary(BinaryOp::Or, std::move(below), std::move(above));
    }
    auto at_least = MakeBinary(BinaryOp::GtEq, left, std::move(low));
    auto at_most = MakeBinary(BinaryOp::LtEq, left, std::move(high));
    return MakeBinary(BinaryOp::And, std::move(at_least), std::move(at_most));
  }

  if (Peek().type == TokenType::Keyword && Peek().text == "LIKE") {
    Error("LIKE is not supported", Peek().position);
  }
  if (negated) {
    Error("expected IN, BETWEEN, or LIKE after NOT", Peek().position);
  }

  return left;
}

ExprPtr Parser::ParseAdditive() {
  ExprPtr left = ParseMultiplicative();
  while (true) {
    BinaryOp op;
    if (Match(TokenType::Symbol, "+"))
      op = BinaryOp::Add;
    else if (Match(TokenType::Symbol, "-"))
      op = BinaryOp::Sub;
    else
      break;
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Binary;
    expr->binary_op = op;
    expr->left = left;
    expr->right = ParseMultiplicative();
    left = expr;
  }
  return left;
}

ExprPtr Parser::ParseMultiplicative() {
  ExprPtr left = ParseUnary();
  while (true) {
    BinaryOp op;
    if (Match(TokenType::Symbol, "*"))
      op = BinaryOp::Mul;
    else if (Match(TokenType::Symbol, "/"))
      op = BinaryOp::Div;
    else
      break;
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Binary;
    expr->binary_op = op;
    expr->left = left;
    expr->right = ParseUnary();
    left = expr;
  }
  return left;
}

ExprPtr Parser::ParseUnary() {
  if (Match(TokenType::Symbol, "-")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Unary;
    expr->unary_op = UnaryOp::Negate;
    expr->left = ParseUnary();
    return expr;
  }
  if (Match(TokenType::Symbol, "+")) {
    return ParseUnary();
  }
  return ParsePrimary();
}

Literal Parser::ParseNumberLiteral() {
  Token token = Expect(TokenType::Number, "a number");
  Literal literal;
  if (token.text.find_first_of(".eE") == std::string::npos) {
    try {
      literal.type = LiteralType::Integer;
      literal.integer = std::stoll(token.text);
      return literal;
    } catch (const std::exception&) {
      Error("integer literal is out of range", token.position);
    }
  }
  try {
    literal.type = LiteralType::Double;
    literal.real = std::stod(token.text);
    return literal;
  } catch (const std::exception&) {
    Error("invalid numeric literal", token.position);
  }
}

ExprPtr Parser::ParsePrimary() {
  const Token& token = Peek();
  if (token.type == TokenType::Number) {
    Literal literal = ParseNumberLiteral();
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Literal;
    expr->literal = std::move(literal);
    return expr;
  }
  if (token.type == TokenType::String) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Literal;
    expr->literal.type = LiteralType::String;
    expr->literal.string = Consume().text;
    return expr;
  }
  if (MatchKeyword("NULL")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Literal;
    expr->literal.type = LiteralType::Null;
    return expr;
  }
  if (MatchKeyword("TRUE") || MatchKeyword("FALSE")) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Literal;
    expr->literal.type = LiteralType::Boolean;
    expr->literal.boolean = tokens_[cursor_ - 1].text == "TRUE";
    return expr;
  }
  if (Match(TokenType::Symbol, "(")) {
    ExprPtr expr = ParseExpression();
    Expect(TokenType::Symbol, ")");
    return expr;
  }
  if (token.type == TokenType::Symbol && token.text == "*") {
    Error("'*' is only supported in the SELECT list and inside COUNT(*)", token.position);
  }
  if (token.type == TokenType::Identifier ||
      (token.type == TokenType::Keyword &&
       (token.text == "COUNT" || token.text == "SUM" || token.text == "AVG" ||
        token.text == "MIN" || token.text == "MAX"))) {
    return ParseFunctionOrColumn();
  }
  Error("expected an expression, found '" +
            (token.type == TokenType::End ? std::string("<end of query>") : token.text) + "'",
        token.position);
}

ExprPtr Parser::ParseFunctionOrColumn() {
  Token name;
  if (Peek().type == TokenType::Keyword &&
      (Peek().text == "COUNT" || Peek().text == "SUM" || Peek().text == "AVG" ||
       Peek().text == "MIN" || Peek().text == "MAX")) {
    name = Consume();
  } else {
    name = ExpectIdentifier("a column or function name");
  }
  if (Match(TokenType::Symbol, "(")) {
    std::string function = ToUpper(name.text);
    if (function != "COUNT" && function != "SUM" && function != "AVG" && function != "MIN" &&
        function != "MAX") {
      Error("unsupported function '" + name.text +
                "'; supported aggregates are COUNT, SUM, AVG, MIN, MAX",
            name.position);
    }
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::Aggregate;
    expr->function = std::string(name.text);
    std::transform(expr->function.begin(), expr->function.end(), expr->function.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (MatchKeyword("DISTINCT")) {
      Error(function + "(DISTINCT ...) is not supported", tokens_[cursor_ - 1].position);
    }
    if (Match(TokenType::Symbol, "*")) {
      if (function != "COUNT") {
        Error(function + "(*) is not valid; only COUNT(*) accepts *", name.position);
      }
      expr->star = true;
    } else {
      expr->argument = ParseExpression();
    }
    Expect(TokenType::Symbol, ")");
    return expr;
  }

  auto expr = std::make_shared<Expr>();
  expr->kind = ExprKind::Column;
  if (Match(TokenType::Symbol, ".")) {
    expr->table = name.text;
    expr->column = ExpectIdentifier("a column name after '.'").text;
  } else {
    expr->column = name.text;
  }
  return expr;
}

ExprPtr Parser::ParseExpressionText(const std::string& text) {
  Parser parser(text);
  ExprPtr expr = parser.ParseExpression();
  parser.ExpectEnd();
  return expr;
}

std::string ExprToString(const ExprPtr& expr) {
  if (!expr)
    return "<null>";
  switch (expr->kind) {
  case ExprKind::Column:
    return expr->table.empty() ? QuoteIdentifier(expr->column)
                               : QuoteIdentifier(expr->table) + "." + QuoteIdentifier(expr->column);
  case ExprKind::Literal:
    return LiteralToString(expr->literal);
  case ExprKind::Star:
    return "*";
  case ExprKind::Binary: {
    std::string op = BinaryOpToString(expr->binary_op);
    return "(" + ExprToString(expr->left) + " " + op + " " + ExprToString(expr->right) + ")";
  }
  case ExprKind::Unary:
    if (expr->unary_op == UnaryOp::Negate)
      return "-" + ExprToString(expr->left);
    return "NOT (" + ExprToString(expr->left) + ")";
  case ExprKind::IsNull:
    return ExprToString(expr->left) + " IS NULL";
  case ExprKind::Aggregate:
    return expr->function + "(" + (expr->star ? "*" : ExprToString(expr->argument)) + ")";
  }
  return "?";
}

std::string StatementToString(const SelectStmt& stmt) {
  std::ostringstream out;
  out << "SELECT ";
  if (stmt.distinct)
    out << "DISTINCT ";
  for (size_t i = 0; i < stmt.items.size(); ++i) {
    if (i)
      out << ", ";
    const SelectItem& item = stmt.items[i];
    if (item.star) {
      if (!item.star_table.empty())
        out << QuoteIdentifier(item.star_table) << ".";
      out << "*";
    } else {
      out << ExprToString(item.expr);
      if (!item.alias.empty() && item.alias != item.expr->column) {
        out << " AS " << QuoteIdentifier(item.alias);
      }
    }
  }
  out << " FROM " << QuoteIdentifier(stmt.from.name);
  if (stmt.from.alias != stmt.from.name)
    out << " AS " << QuoteIdentifier(stmt.from.alias);
  for (const JoinClause& join : stmt.joins) {
    out << (join.left_outer ? " LEFT JOIN " : " JOIN ") << QuoteIdentifier(join.table.name);
    if (join.table.alias != join.table.name)
      out << " AS " << QuoteIdentifier(join.table.alias);
    out << " ON " << ExprToString(join.on);
  }
  if (stmt.where)
    out << " WHERE " << ExprToString(stmt.where);
  if (!stmt.group_by.empty()) {
    out << " GROUP BY ";
    for (size_t i = 0; i < stmt.group_by.size(); ++i) {
      if (i)
        out << ", ";
      out << ExprToString(stmt.group_by[i]);
    }
  }
  if (stmt.having)
    out << " HAVING " << ExprToString(stmt.having);
  if (!stmt.order_by.empty()) {
    out << " ORDER BY ";
    for (size_t i = 0; i < stmt.order_by.size(); ++i) {
      if (i)
        out << ", ";
      if (stmt.order_by[i].ordinal)
        out << stmt.order_by[i].ordinal;
      else
        out << ExprToString(stmt.order_by[i].expr);
      if (stmt.order_by[i].descending)
        out << " DESC";
    }
  }
  if (stmt.limit >= 0)
    out << " LIMIT " << stmt.limit;
  if (stmt.offset > 0)
    out << " OFFSET " << stmt.offset;
  return out.str();
}

} // namespace qe::sql
