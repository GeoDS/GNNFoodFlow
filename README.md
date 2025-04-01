# County-Level Trade Flow Prediction using Graph Neural Networks

## Motivation
Understanding and predicting trade flows between counties is crucial for economic planning, infrastructure development, and policy-making. Traditional statistical methods often fail to capture the complex relationships and dependencies in trade networks. This project leverages modern Graph Neural Networks (GNNs) to better model these intricate patterns of inter-county trade.

## Goal
The primary objectives of this project are to:
- Develop accurate predictions of trade flows between counties using both Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN)
- Implement a hurdle model approach that can:
  1. Predict whether trade exists between two counties
  2. Estimate the volume of trade when it exists
- Compare the performance of different GNN architectures for trade flow prediction
- Create a robust model that accounts for both geographical and economic features

## Data
This project utilizes the Freight Analysis Framework (FAF5) dataset, which provides comprehensive data on freight movement between regions in the United States. Key data sources include:
- **Trade Data**: FAF5 SCTG1 commodity flow data (`code/data/FAF5_SCTG1.csv`)
- **Geographic Information**:
  - County shapefiles (`code/data/shapefiles/cb_2017_us_county_500k/cb_2017_us_county_500k.shp`)
  - State shapefiles (`code/data/shapefiles/cb_2018_us_state_20m/cb_2018_us_state_20m.shp`)
  - FAF zones shapefiles (`code/data/shapefiles/2017_CFS_Metro_Areas_with_FAF/2017_CFS_Metro_Areas_with_FAF.shp`)
- **Economic Indicators**: County-level economic data (`code/data/faf_features.csv`)
- **Distance Information**: FAF Distance Matrix (`code/data/FAF_Distance_Matrix.csv`)

## Features
- **Hurdle Model Approach**: Combined classification and regression for better handling of zero-trade flows
- **Rich Feature Set**: Incorporates:
  - County-level economic indicators (population, employment_rate, median_income, industry_diversity)
  - Geographic distance and transportation modes (distance, dms_mode, dist_band)
- **FAF Zone Level Estimation**: Estimates trade flows at the FAF zone level, especially for the trade without detailed county-level data

## How to Use

### Prerequisites
- Python 3.9
- PyTorch (>=1.9.0)
- Pandas (>=1.3.0)
- NumPy (>=1.20.0)
- Matplotlib (>=3.4.0)
- Scikit-learn (>=0.24.0)
- NetworkX (>=2.6.0)
- GeoPandas (>=0.9.0)
- PyTorch Geometric (>=2.0.0)
- tqdm (>=4.62.0)

## Acknowledgements
This project is supported by ICICLE, a research project funnded by Ohio State University.