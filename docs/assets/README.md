# Visual assets

These files are generated from real repository output; they are not mockups or
hand-drawn performance claims.

| Asset | Source | Regenerate with |
| --- | --- | --- |
| `demo.gif` | Animated terminal transcript of `build/qe-cli --version` and a selective `GROUP BY` query on the sample dataset. The script captures the CLI's actual stdout/stderr. | `python scripts/render_demo_gif.py` |
| `demo-poster.png` | Static final frame of the same transcript, used as an accessible preview and for links where animation is not appropriate. | `python scripts/render_demo_gif.py` |
| [`benchmarks/charts/latest.png`](../../benchmarks/charts/latest.png) | Chart rendered from `benchmarks/results/latest.json`. GPU bars appear only when the harness verified a GPU query. | `python scripts/plot_benchmarks.py benchmarks/results/latest.json` |

The demo uses the CPU build and 200,000 sample rows from `scripts/demo.sh`.
Timings in the terminal are real output from the machine that generated the
asset; they are illustrative and vary by host. Benchmark claims in the README
come from the committed JSON and Markdown report, not from the GIF.
