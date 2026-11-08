## Summary

<!-- What changed, and why? Keep this focused on one logical change. -->

## Related issue

<!-- Link the issue, or write "none". -->

## Changes

- 

## Verification

<!-- Paste the commands you ran and their results. -->

```bash
scripts/build.sh
ctest --test-dir build --output-on-failure
python -m pytest tests -q
```

## Benchmarks

<!--
Only required for performance-sensitive changes. State whether the benchmark
harness was run, and include the relevant before/after numbers. Never hand-type
or estimate a result.
-->

- [ ] Not applicable (no performance-sensitive code changed)
- [ ] Benchmark run attached or summarized below

## Checklist

- [ ] The change is limited to one logical feature or fix.
- [ ] Tests cover the new behavior, including NULL, empty-input, and multi-batch cases where relevant.
- [ ] Parser tests include a rejection case for malformed or unsupported SQL.
- [ ] User-visible SQL behavior is documented in `README.md`.
- [ ] `CHANGELOG.md` is updated under `Unreleased`.
- [ ] `clang-format` and `ruff format` have been run.
- [ ] README media, if changed, was regenerated from its source script rather than hand-edited.
- [ ] CI is expected to pass, including ASan/UBSan and coverage.
- [ ] No benchmark, GPU, or performance number is fabricated or estimated.

## Notes for reviewers

<!-- Call out risky areas, follow-up work, or intentionally deferred scope. -->
