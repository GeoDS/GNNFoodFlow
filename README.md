# County-Level Trade Flow Prediction using Graph Neural Networks
A project leveraging Graph Neural Networks (GNNs) to predict trade flows between counties and FAF zones for economic planning, infrastructure development, and policy-making.

Tags: Smart-Foodsheds, AI4CI

## Tutorials

### Getting Started with Trade Flow Prediction
- **Prerequisites**:
  - Python 3.9
  - PyTorch (>=1.9.0)
  - PyTorch Geometric (>=2.0.0)
  - Pandas (>=1.3.0)
  - NumPy (>=1.20.0)
  - Matplotlib (>=3.4.0)
  - Scikit-learn (>=0.24.0)
  - NetworkX (>=2.6.0)
  - GeoPandas (>=0.9.0)
  - tqdm (>=4.62.0)
- **Steps to run the model**:
  1. Clone the repository
  2. Install dependencies
  3. Prepare your data following the format in the data section
  4. Run the training script
  5. Evaluate model performance

## How-To Guides

### How to Implement a Hurdle Model for Trade Prediction
- **Problem**: Predicting trade flows with many zero values
- **Solution**: Implement a two-stage hurdle model that:
  1. First predicts whether trade exists between two regions
  2. Then estimates the volume of trade when it exists
- **Code example**: See the GAT model implementation in `code/model.py`

### How to Incorporate Geographic and Economic Features
- **Problem**: Capturing complex relationships in trade networks
- **Solution**: Combine multiple feature types:
  - County-level economic indicators (population, employment_rate, median_income)
  - Geographic distance and transportation modes
  - Industry diversity metrics

## Explanation

### Graph Neural Networks for Trade Flow Prediction
Our approach uses Graph Neural Networks to model trade relationships between counties and FAF zones. The model architecture includes:
- Graph Attention Networks (GAT) to capture the importance of different connections
- Graph Convolutional Networks (GCN) for comparison
- A hurdle model approach that separates the prediction into classification and regression tasks

This design better handles the sparse nature of trade networks where many county pairs have zero trade. The model learns both from geographic proximity and economic similarity, providing more accurate predictions than traditional statistical methods.

## Reference

### Data Sources
- **Trade Data**: FAF5 SCTG1 commodity flow data (`code/data/FAF5_SCTG1.csv`)
- **Geographic Information**:
  - County shapefiles (`code/data/shapefiles/cb_2017_us_county_500k/cb_2017_us_county_500k.shp`)
  - State shapefiles (`code/data/shapefiles/cb_2018_us_state_20m/cb_2018_us_state_20m.shp`)
  - FAF zones shapefiles (`code/data/shapefiles/2017_CFS_Metro_Areas_with_FAF/2017_CFS_Metro_Areas_with_FAF.shp`)
- **Economic Indicators**: County-level economic data (`code/data/faf_features.csv`)
- **Distance Information**: FAF Distance Matrix (`code/data/FAF_Distance_Matrix.csv`)

### Acknowledgements
National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)

### Future Work
- Extending the model to handle inter-county trade flow predictions
- Refining the model to capture more granular trade patterns
- Implementing visualization tools for inter-county trade networks
