# Kwauntdream v2.5.1

Experimental model exploring how wave interference patterns emerge in networks and how topology shapes correlation strength between nodes.
## Contents

- `Kwauntdream_v2_5_1 2.py` — simulation & analysis script.
- `v2_5_1_nodes_agg.csv` — node-level aggregated metrics across random seeds.
- `v2_5_1_summary.csv` — configuration-level summary (graph_type × pseudo_count).
## Quick Start

```bash
# Python 3.10+ is recommended
pip install numpy scipy pandas networkx matplotlib

# Option 1: run as a script (file name contains a space, so use quotes)
python "Kwauntdream_v2_5_1 2.py"
```
## Reproducibility

Running `run_test_suite(outdir=...)` will:
- generate random graphs of different types (e.g., small-world, Erdős–Rényi),
- simulate wave-like evolution over the graph Laplacian,
- compute pairwise correlation metrics between indices,
- write aggregated CSVs matching the ones in this deposit.

Outputs:
- `v2_5_1_nodes_agg.csv` — per-index aggregates
- `v2_5_1_summary.csv` — per-configuration summary
- optional figures in `outdir` (if enabled inside the script)

## Method 

- Build graph Laplacians for selected network models.
- Evolve complex amplitudes over time via `exp(-i L t)` (wave-like propagation on graphs).
- Form 2×2 pseudo-states from amplitudes to score inter-index correlation strength.
- Optimize a simple phase transform on a grid to maximize a CHSH-like bound.
- Compute negativity-like correlation and average it per index.
- Aggregate across random seeds; summarize per configuration.
## Data Dictionary

### v2_5_1_nodes_agg.csv
- `idx` — pseudo-index (0-based)
- `avg_neg_mean` — mean correlation (negativity-like) across pairs and seeds
- `avg_neg_std` — std. dev. of the above across seeds
- `degree_mean` — mean node degree (from Laplacian diagonal) across seeds
- `fiedler_mean` — mean |Fiedler vector component| across seeds
- `graph_type` — network model label (e.g., `smallworld`, `erdos`)
- `pseudo_count` — number of indices used to form pseudo-states

### v2_5_1_summary.csv
- `graph_type` — network model (`smallworld`, `erdos`)
- `pseudo_count` — number of indices
- `mean_avg_neg_mean` — mean of `avg_neg_mean` over indices
- `max_avg_neg_mean` — max of `avg_neg_mean` over indices
- `std_avg_neg_mean` — std. dev. of `avg_neg_mean` over indices

# Parameters

Typical knobs inside the script:
- `graph_type`: `"smallworld"` or `"erdos"`
- `pseudo_count`: e.g., 24, 32, 40
- `seeds`: number and values of random seeds
- `steps`, `t`, `alpha`: evolution and dephasing controls
- output directory for CSVs and figures

## Minimal Analysis Example

```python
import pandas as pd

nodes = pd.read_csv("v2_5_1_nodes_agg.csv")
summary = pd.read_csv("v2_5_1_summary.csv")

# top indices by avg_neg_mean
print(nodes.sort_values("avg_neg_mean", ascending=False).head(10))

# compare models
print(summary.groupby("graph_type")["mean_avg_neg_mean"].mean())

```
## License

Licensed under the **Apache License, Version 2.0** (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

> https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## Author

**Bato Naidanov**  
📧 bnaydanov@gmail.com 
