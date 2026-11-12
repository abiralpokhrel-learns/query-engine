#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli/printer.h"
#include "engine/executor.h"
#include "engine/plan.h"
#include "gpu/cuda_runtime.h"
#include "sql/sql.h"

#ifndef QE_VERSION
#define QE_VERSION "0.2.0"
#endif
#ifndef QE_ARROW_VERSION
#define QE_ARROW_VERSION "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Options {
  std::string path;
  std::vector<std::pair<std::string, std::string>> named_tables;
  std::string query;
  std::string format = "table";
  int64_t batch_size = 8192;
  int64_t max_rows = -2; // -2 means: 50 for tables, all rows for JSON.
  bool max_rows_set = false;
  bool explain = false;
  int repeat = 1;
  int threads = 1;
  bool show_stats = true;
  bool use_gpu = false;
};

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::string Lower(std::string_view text) {
  std::string result(text);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

std::string CudaVersion(int version) {
  if (version <= 0)
    return "unknown";
  std::ostringstream text;
  text << version / 1000 << '.' << (version % 1000) / 10;
  return text.str();
}

int64_t ParseInteger(const std::string& text, const std::string& option) {
  size_t consumed = 0;
  int64_t value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw UsageError("invalid value for " + option + ": '" + text + "'");
  }
  if (consumed != text.size()) {
    throw UsageError("invalid value for " + option + ": '" + text + "'");
  }
  return value;
}

std::pair<std::string, std::string> ParseTableSpec(const std::string& spec) {
  const size_t equals = spec.find('=');
  if (equals == std::string::npos || equals == 0 || equals + 1 == spec.size()) {
    throw UsageError("--table expects NAME=PATH, got '" + spec + "'");
  }
  return {spec.substr(0, equals), spec.substr(equals + 1)};
}

std::string DeriveTableName(const std::string& path) {
  std::string stem = std::filesystem::path(path).stem().string();
  if (stem.empty())
    stem = "table";
  for (char& c : stem) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      c = '_';
  }
  if (std::isdigit(static_cast<unsigned char>(stem.front())))
    stem = "t_" + stem;
  return stem;
}

bool NameExists(const std::vector<std::pair<std::string, std::string>>& tables,
                const std::string& name) {
  const std::string lowered = Lower(name);
  return std::any_of(tables.begin(), tables.end(),
                     [&](const auto& entry) { return Lower(entry.first) == lowered; });
}

void PrintUsage(std::ostream& out) {
  out << "qe-cli " << QE_VERSION << " - columnar SQL query engine\n\n"
      << "Usage:\n"
      << "  qe-cli [options] <file.parquet> [SQL]\n"
      << "  qe-cli [options] --table NAME=PATH [--table NAME=PATH ...] --query SQL\n\n"
      << "Options:\n"
      << "  --table NAME=PATH     Register a Parquet file under a SQL table name.\n"
      << "                        Repeat for joins. A positional .parquet path is\n"
      << "                        registered automatically under its file stem.\n"
      << "  --query SQL           SQL text. May also be the second positional argument.\n"
      << "  --format table|json   Output format (default: table).\n"
      << "  --max-rows N          Maximum result rows to display; -1 shows all.\n"
      << "                        Defaults to 50 for tables and all for JSON.\n"
      << "  --batch-size N        Vectorized batch size (default: 8192).\n"
      << "  --threads N           Parallel scan workers for row-group decoding\n"
      << "                        (default: 1). Results never depend on this.\n"
      << "  --explain             Print the optimized plan and do not execute.\n"
      << "  --repeat N            Execute the query N times and report the last result.\n"
      << "  --gpu                 Use the optional CUDA global-aggregate path where\n"
      << "                        supported; the CPU path remains the default.\n"
      << "  --no-stats            Do not print scan statistics to stderr.\n"
      << "  -h, --help            Show this help.\n"
      << "  --version             Show the version.\n\n"
      << "Timings and scan statistics are written to stderr; query results go to stdout.\n\n"
      << "Examples:\n"
      << "  qe-cli data/sales.parquet \"SELECT region, SUM(amount) FROM sales GROUP BY "
         "region\"\n"
      << "  qe-cli --table a=a.parquet --table b=b.parquet \\\n"
      << "         --query \"SELECT a.k, b.v FROM a JOIN b ON a.id = b.id LIMIT 10\"\n";
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  std::string positional_query;
  bool query_set = false;

  auto require_value = [&](int& index, const std::string& option) -> std::string {
    if (index + 1 >= argc)
      throw UsageError(option + " requires a value");
    return argv[++index];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(std::cout);
      std::exit(0);
    }
    if (arg == "--version") {
      std::cout << "qe-cli " << QE_VERSION << '\n';
      std::cout << "arrow: " << QE_ARROW_VERSION << '\n';
      const qe::gpu::DeviceInfo device = qe::gpu::QueryDevice();
      if (device.available) {
        std::cout << "cuda: enabled (" << device.name << ", "
                  << (device.memory_bytes / (1024 * 1024)) << " MiB, compute " << device.major
                  << "." << device.minor
                  << ", cuda_driver_api=" << CudaVersion(device.driver_version) << ")\n";
        std::cout << "cuda_ops: global-aggregate\n";
      } else if (device.build_enabled) {
        std::cout << "cuda: build enabled but unavailable";
        if (!device.error.empty())
          std::cout << " (" << device.error << ")";
        std::cout << '\n';
      } else {
        std::cout << "cuda: unavailable (CPU-only build)\n";
      }
      std::exit(0);
    }
    if (arg == "--explain") {
      options.explain = true;
      continue;
    }
    if (arg == "--no-stats") {
      options.show_stats = false;
      continue;
    }
    if (arg == "--gpu") {
      options.use_gpu = true;
      continue;
    }
    if (arg == "--table" || arg.rfind("--table=", 0) == 0) {
      std::string spec;
      if (arg == "--table") {
        spec = require_value(i, "--table");
      } else {
        spec = arg.substr(std::string("--table=").size());
      }
      options.named_tables.push_back(ParseTableSpec(spec));
      continue;
    }
    if (arg == "--query" || arg.rfind("--query=", 0) == 0) {
      if (query_set)
        throw UsageError("--query may only be specified once");
      options.query = arg == "--query" ? require_value(i, "--query")
                                       : arg.substr(std::string("--query=").size());
      query_set = true;
      continue;
    }
    if (arg == "--format" || arg.rfind("--format=", 0) == 0) {
      options.format = arg == "--format" ? require_value(i, "--format")
                                         : arg.substr(std::string("--format=").size());
      continue;
    }
    if (arg == "--batch-size" || arg.rfind("--batch-size=", 0) == 0) {
      const std::string value = arg == "--batch-size"
                                    ? require_value(i, "--batch-size")
                                    : arg.substr(std::string("--batch-size=").size());
      options.batch_size = ParseInteger(value, "--batch-size");
      continue;
    }
    if (arg == "--threads" || arg.rfind("--threads=", 0) == 0) {
      const std::string value = arg == "--threads" ? require_value(i, "--threads")
                                                   : arg.substr(std::string("--threads=").size());
      const int64_t threads = ParseInteger(value, "--threads");
      if (threads < 1 || threads > 1024) {
        throw UsageError("--threads must be between 1 and 1024");
      }
      options.threads = static_cast<int>(threads);
      continue;
    }
    if (arg == "--max-rows" || arg.rfind("--max-rows=", 0) == 0) {
      const std::string value = arg == "--max-rows" ? require_value(i, "--max-rows")
                                                    : arg.substr(std::string("--max-rows=").size());
      options.max_rows = Lower(value) == "all" ? -1 : ParseInteger(value, "--max-rows");
      options.max_rows_set = true;
      continue;
    }
    if (arg == "--repeat" || arg.rfind("--repeat=", 0) == 0) {
      const std::string value = arg == "--repeat" ? require_value(i, "--repeat")
                                                  : arg.substr(std::string("--repeat=").size());
      const int64_t repeat = ParseInteger(value, "--repeat");
      if (repeat < 1 || repeat > 100000) {
        throw UsageError("--repeat must be between 1 and 100000");
      }
      options.repeat = static_cast<int>(repeat);
      continue;
    }
    if (!arg.empty() && arg[0] == '-' && arg != "-") {
      throw UsageError("unknown option '" + arg + "' (try --help)");
    }

    if (options.path.empty() && (options.named_tables.empty() || std::filesystem::exists(arg) ||
                                 Lower(arg).ends_with(".parquet"))) {
      options.path = arg;
    } else {
      if (!positional_query.empty())
        positional_query += ' ';
      positional_query += arg;
    }
  }

  if (!positional_query.empty()) {
    if (query_set) {
      throw UsageError("SQL was provided both positionally and with --query");
    }
    options.query = positional_query;
  }
  if (!options.path.empty()) {
    if (NameExists(options.named_tables, DeriveTableName(options.path))) {
      throw UsageError("table name '" + DeriveTableName(options.path) +
                       "' is already registered; use --table to choose a different name");
    }
    options.named_tables.emplace_back(DeriveTableName(options.path), options.path);
  }
  if (options.named_tables.empty()) {
    throw UsageError("no Parquet input; pass a path or --table NAME=PATH");
  }
  if (options.query.empty()) {
    throw UsageError("no SQL query; pass SQL text or --query SQL");
  }

  options.format = Lower(options.format);
  if (options.format != "table" && options.format != "json") {
    throw UsageError("--format must be 'table' or 'json'");
  }
  if (options.batch_size <= 0) {
    throw UsageError("--batch-size must be positive");
  }
  if (options.max_rows_set && options.max_rows < -1) {
    throw UsageError("--max-rows must be -1 or non-negative");
  }
  return options;
}

void PrintScanStats(const qe::ExecutionStats& stats) {
  const int64_t row_groups_total = stats.row_groups_total;
  std::cerr << "scan: rows_scanned=" << stats.rows_scanned
            << " rows_after_pushdown=" << stats.rows_after_pushdown
            << " row_groups_read=" << stats.row_groups_read
            << " row_groups_skipped=" << stats.row_groups_skipped
            << " row_groups_total=" << row_groups_total << " batches=" << stats.batches_scanned
            << " rows_from_metadata=" << stats.rows_from_metadata << '\n';
}

void PrintGpuStats(const qe::ExecutionStats& stats) {
  std::cerr << "gpu: used=" << (stats.gpu_used ? 1 : 0);
  if (stats.gpu_used) {
    std::cerr << " batches=" << stats.gpu_batches << " transfer_ms=" << stats.gpu_transfer_ms
              << " kernel_ms=" << stats.gpu_kernel_ms;
  } else if (stats.gpu_attempted) {
    std::cerr << " reason=no-rows";
  } else {
    std::cerr << " reason=no-supported-gpu-operator";
  }
  std::cerr << '\n';
}

double Median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 == 1)
    return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

int Run(int argc, char** argv) {
  const auto program_start = Clock::now();
  Options options;
  try {
    options = ParseOptions(argc, argv);
  } catch (const UsageError& error) {
    std::cerr << "error: " << error.what() << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }

  int64_t max_rows = options.max_rows;
  if (!options.max_rows_set)
    max_rows = options.format == "table" ? 50 : -1;

  qe::Catalog catalog;
  const auto load_start = Clock::now();
  for (const auto& [name, path] : options.named_tables) {
    auto status = catalog.AddFile(name, path);
    if (!status.ok()) {
      std::cerr << "error: " << status.ToString() << '\n';
      return 1;
    }
  }
  const auto load_end = Clock::now();

  const auto parse_start = Clock::now();
  qe::sql::SelectStmt statement;
  try {
    qe::sql::Parser parser(options.query);
    statement = parser.Parse();
  } catch (const qe::sql::SqlError& error) {
    std::cerr << "SQL parse error at byte " << error.position << ": " << error.what() << '\n';
    return 1;
  }
  const auto parse_end = Clock::now();

  const auto plan_start = Clock::now();
  auto planned = qe::PlanQuery(statement, catalog);
  if (!planned.ok()) {
    std::cerr << "error: " << planned.status().ToString() << '\n';
    return 1;
  }
  const auto plan_end = Clock::now();
  qe::QueryPlan plan = std::move(planned).ValueOrDie();

  if (options.explain) {
    std::cout << plan.explain;
    const double load_ms = Milliseconds(load_start, load_end);
    const double parse_ms = Milliseconds(parse_start, parse_end);
    const double plan_ms = Milliseconds(plan_start, plan_end);
    std::cerr << std::fixed << std::setprecision(3) << "timings: catalog_ms=" << load_ms
              << " parse_ms=" << parse_ms << " plan_ms=" << plan_ms << '\n';
    return 0;
  }

  qe::ExecutionOptions execution_options;
  execution_options.batch_size = options.batch_size;
  execution_options.use_gpu = options.use_gpu;
  execution_options.threads = options.threads;
  if (options.use_gpu) {
    const qe::gpu::DeviceInfo device = qe::gpu::QueryDevice();
    if (!device.build_enabled) {
      std::cerr << "error: --gpu requires a build configured with -DQE_ENABLE_CUDA=ON\n";
      return 1;
    }
    if (!device.available) {
      std::cerr << "error: --gpu requested but no usable CUDA device: " << device.error << '\n';
      return 1;
    }
    // Run one real kernel before the timed queries. This validates the device
    // path and loads the CUDA module so the first timed query is not dominated
    // by one-time runtime initialization.
    const qe::gpu::SelfTest warmup = qe::gpu::RunSelfTest();
    if (!warmup.ok) {
      std::cerr << "error: GPU self-test failed: " << warmup.message << '\n';
      return 1;
    }
  }

  qe::ExecutionResult result;
  std::vector<double> execute_times;
  execute_times.reserve(static_cast<size_t>(options.repeat));
  for (int run = 0; run < options.repeat; ++run) {
    auto executed = qe::ExecutePlan(plan.root, execution_options);
    if (!executed.ok()) {
      std::cerr << "error: " << executed.status().ToString() << '\n';
      return 1;
    }
    result = std::move(executed).ValueOrDie();
    execute_times.push_back(result.execute_ms);
  }

  if (options.format == "json") {
    std::cout << qe::TableToJson(*result.table, max_rows) << '\n';
  } else {
    qe::PrintTable(*result.table, max_rows, std::cout);
  }

  const auto end = Clock::now();
  const double load_ms = Milliseconds(load_start, load_end);
  const double parse_ms = Milliseconds(parse_start, parse_end);
  const double plan_ms = Milliseconds(plan_start, plan_end);
  const double median_ms = Median(execute_times);
  const double total_ms = Milliseconds(program_start, end);

  std::cerr << std::fixed << std::setprecision(3);
  if (options.repeat == 1) {
    std::cerr << "timings: catalog_ms=" << load_ms << " parse_ms=" << parse_ms
              << " plan_ms=" << plan_ms << " execute_ms=" << median_ms << " total_ms=" << total_ms
              << '\n';
  } else {
    std::cerr << "runs_ms:";
    for (double time : execute_times)
      std::cerr << ' ' << time;
    std::cerr << '\n';
    std::cerr << "timings: catalog_ms=" << load_ms << " parse_ms=" << parse_ms
              << " plan_ms=" << plan_ms << " execute_ms=" << median_ms
              << " execute_min_ms=" << *std::min_element(execute_times.begin(), execute_times.end())
              << " execute_max_ms=" << *std::max_element(execute_times.begin(), execute_times.end())
              << " total_ms=" << total_ms << " runs=" << options.repeat << '\n';
  }
  if (options.use_gpu)
    PrintGpuStats(result.stats);
  if (options.show_stats)
    PrintScanStats(result.stats);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "error: unknown failure\n";
    return 1;
  }
}
