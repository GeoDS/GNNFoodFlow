# Multi-Type Food Flow Prediction using Graph Neural Networks

An [NSF-AI Institute ICICLE project](https://icicle.osu.edu/) leveraging Graph Neural Networks (GNNs) to predict food flows between counties and FAF zones for economic planning, infrastructure development, and policy-making. This repository contains the core implementation of **MT-LocalizedGCN**, a spatially localized multi-task graph convolutional network that predicts food trade flows across seven SCTG commodity categories (SCTG 01–07) simultaneously. The model is trained on Freight Analysis Framework (FAF) zone origin–destination (OD) flows and can be transferred to county features for county-to-county cross-scale inference. It addresses the sparsity of trade data with a two-stage hurdle formulation that distinguishes the presence of trade from its magnitude.

Tags: Smart-Foodsheds, AI4CI, Food-Systems, Food-Access

Great News! The portal is now online at https://gnnfoodflowportal.pods.icicleai.tapis.io/

Please go to https://github.com/ICICLE-ai/GNNFoodFlowPortal/ for more updated information and better accessability

> **Note on model versions.** This is a new multi-task model: a *single*
> `MTLocalizedGCN` now predicts all seven SCTG categories at once,
> replacing the earlier approach that trained one checkpoint per SCTG code.
> The original single-task implementation is archived in [`legacy/`](legacy/)
> for reference.

## Model Details
- **Developed by**: Qianheng Zhang & ICICLE Food Systems Team led by Professor Song Gao
- **Funded by**: NSF AI Institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)
- **Model type**: Multi-task Graph Neural Network — **MT-LocalizedGCN** (a distance-kNN localized GCN; `MTEdgeMLP` non-spatial baseline also provided)
- **Language(s)**: English (for documentation and metadata)
- **License**: MIT License
- **Framework**: PyTorch, PyTorch Geometric

This repository contains the minimal core implementation for the multi-task FoodFlow GNN model. It intentionally excludes the web portal and large precomputed county prediction CSVs — the portal lives in a separate repository, and full prediction outputs should be generated locally or distributed through an external data host. _Comprehensive usage instructions and tutorials on Hugging Face are being updated for the multi-task (MT-LocalizedGCN) model and will be linked here once available._

## Contents

```text
code/
├── model.py                    # MTEdgeMLP and MTLocalizedGCN
├── dataset.py                  # FAF SCTG 01-07 multitask loader
├── train.py                    # multitask hurdle loss and metrics
├── run_multitask.py            # training entry point
├── run_inference.py            # county cross-scale inference entry point
├── sample_multitask.ipynb      # minimal usage notebook
├── data/                       # compact FAF/county input data
├── artifacts/                  # county inference artifacts and edge scaler
├── models/                     # saved multitask checkpoint
└── results/                    # small metric-summary CSVs only
legacy/                         # archived single-task (per-SCTG) implementation
```

The interactive visualization portal now lives in its own repository:
[ICICLE-ai/GNNFoodFlowPortal](https://github.com/ICICLE-ai/GNNFoodFlowPortal/).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a one-epoch smoke test:

```bash
python code/run_multitask.py --epochs 1 --hidden 32 --log_every 1 --no_pos_weight --no_save
```

Run the default training protocol:

```bash
python code/run_multitask.py
```

Generate county-to-county predictions:

```bash
python code/run_inference.py --all-county-crossscale --out predictions_county_crossscale.csv
```

The generated county prediction CSV is large and is not committed here. See [`REPRODUCE.md`](REPRODUCE.md) for the full reproduction protocol.

## Model Notes

- Food categories: SCTG 01–07.
- Default architecture: `MTLocalizedGCN` (distance-kNN localized graph, `k=5`).
- Task heads: one hurdle head per SCTG category (seven total), sharing one GCN backbone.
- County-scale outputs are model-inferred flows, not observed county ground truth.
- The bundled summary CSVs in `code/results/` are lightweight checks, not the complete paper evidence archive.

## Legacy (Single-Task) Model

The original implementation trained a separate GAT/GCN checkpoint per SCTG
commodity code (`best_model1_gcn.pth` … `best_model7_gcn.pth`). That code,
its notebooks, checkpoints, and original documentation are preserved under
[`legacy/`](legacy/). It is kept for reference and is not guaranteed to run
against the current top-level layout.

## License
MIT

## Reference
If you find our code or ideas useful for your research, please cite our paper:

> **The multi-task (MT-LocalizedGCN) paper — _Multi-Type Food Flow Prediction and Scenario Analysis between U.S. Counties using Graph Neural Networks_ (SIGSPATIAL '26) — is to be added.**

In the meantime, please cite the earlier single-task work:

*Zhang, Q., Dev, P., Miller, M., Morales, A., &  Gao, S. (2025). [Scalable Inter-County Food Flow Prediction Using Graph Neural Networks](https://doi.org/10.1145/3748636.3764168). ACM SIGSPATIAL '25, November 3–6, 2025, Minneapolis, MN, USA , 1-10.*

```
@article{zhang2025scalable,
  title={Scalable Inter-County Food Flow Prediction Using Graph Neural Networks},
  author={Qianheng Zhang and Dev Paul and Michelle Miller and Alfonso Morales and Song Gao.},
  journal={The 33rd ACM International Conference on Advances in Geographic
Information Systems (SIGSPATIAL ’25)},
  volume={2025},
  number={1},
  pages={1--10},
  year={2025},
  doi={10.1145/3748636.3764168}
  publisher={ACM}
}
```

## For Detailed Usage
> _Detailed setup instructions, tutorials, data download links, and troubleshooting on Hugging Face are being updated for the multi-task (MT-LocalizedGCN) model and will be linked here once available._

For the interactive visualization portal, see the **[GNN Food Flow Portal](https://gnnfoodflowportal.pods.icicleai.tapis.io/)** and its repository at [ICICLE-ai/GNNFoodFlowPortal](https://github.com/ICICLE-ai/GNNFoodFlowPortal/).

### Direct Use
- Predicting food flows between regions using node (county/FAF zone) and edge features
- Modeling economic connectivity and transportation dependency

### Visualization Portal
Please visit the new online portal here: **[GNN Food Flow Portal](https://gnnfoodflowportal.pods.icicleai.tapis.io/)** for the latest interactive visualization and download features.

Please go to https://github.com/ICICLE-ai/GNNFoodFlowPortal/ for more updated information and better accessability

### Downstream Use
- Spatial forecasting of trade changes under policy shifts
- Identifying critical counties for supply chain resilience

### Out-of-Scope Use
- Real-time food trade forecasting
- Non-U.S. geographic settings without retraining

## Bias, Risks, and Limitations
- **Bias**: Model predictions depend on historical FAF data and may not reflect unexpected future disruptions (e.g., disasters, pandemics)
- **Limitations**: Prediction is limited to the predefined commodity codes SCTG 01–07
- **Data quality**: Assumes accuracy of FAF flow data and economic indicators

## Recommendations
Users should:
- Evaluate model generalizability before applying it to non-FAF settings
- Interpret sparse predictions carefully—zeros may result from missing data, not true absence

## Reference

### Data Sources
- **Trade Data**: [FAF5.6.1 SCTG 01–07 commodity flow data](https://faf.ornl.gov/faf5/) (`code/data/FAF5_SCTG1.csv` … `code/data/FAF5_SCTG7.csv`)
- **Geographic Information** (used for visualization; hosted in the [portal repository](https://github.com/ICICLE-ai/GNNFoodFlowPortal/)):
  - County shapefiles (`cb_2017_us_county_500k`)
  - State shapefiles (`cb_2018_us_state_20m`)
  - FAF zones shapefiles (`2017_CFS_Metro_Areas_with_FAF`)
- **FAF Economic Indicators**: FAF-level economic data (`code/data/faf_features_aligned_filtered.csv`)
- **County Economic Indicators**: County-level economic data (`code/data/county_aligned_filtered.csv`)
- **Distance Information**: FAF Distance Matrix (`code/data/FAF_distance_matrix.csv`)
Due to the low efficiency of separating county level distance matrix, the distance information is included within county level feature information

### Acknowledgements
National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)

### Future Work
- Refining the multi-task model to capture more granular food trade patterns
- Extending cross-scale transfer to additional commodity groups and finer spatial resolutions
