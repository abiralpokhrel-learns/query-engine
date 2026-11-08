# Security policy

`qe` is a local, read-only analytical query engine. It parses SQL and reads
Parquet files supplied by the caller; it does not open network ports, execute
user code, or write to the input files.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |
| < 0.2 | No |

## Reporting a vulnerability

Please do not open a public issue for a security concern. Use GitHub's private
security advisory flow:

<https://github.com/abiralpokhrel-learns/query-engine/security/advisories/new>

Include the affected version, a minimal reproducer, and the expected impact if
you can. Reports are reviewed on a best-effort basis. Please allow time for a fix
and a coordinated disclosure before publishing details.

## Scope

In scope:

- memory-safety or denial-of-service bugs in the SQL parser, planner, operators,
  Parquet reader, or CUDA runtime;
- malformed Parquet files or SQL that cause an out-of-bounds access, an
  unchecked allocation, or a crash outside the documented error path;
- build or packaging changes that would execute untrusted code.

Out of scope:

- vulnerabilities in Apache Arrow, Parquet, DuckDB, the CUDA toolkit, or the
  operating system; report those to the upstream project, though a version bump
  in this repository may still be appropriate;
- resource-exhaustion reports that only require running the CLI on an
  intentionally huge query or dataset without a memory limit;
- benchmark timing differences or unsupported SQL behavior.

## Dependency versions

The pinned development environment is in `environment.yml`; CI installs from the
same file and runs a separate AddressSanitizer / UndefinedBehaviorSanitizer job.
