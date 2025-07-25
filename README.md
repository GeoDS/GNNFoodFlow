# Multi-Scale Food Flow Prediction using Graph Neural Networks
A project leveraging Graph Neural Networks (GNNs) to predict food flows between counties and FAF zones for economic planning, infrastructure development, and policy-making. This model predicts food trade flows between U.S. counties and Freight Analysis Framework (FAF) zones using Graph Neural Networks (GNNs). It addresses the challenges of sparsity in trade data by applying a two-stage hurdle model that distinguishes between the presence and magnitude of trade.

Tags: Smart-Foodsheds, AI4CI

## Model Details
- **Developed by**: Qianheng Zhang & ICICLE Team
- **Funded by**: NSF AI Institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)
- **Model type**: Graph Neural Network (GAT and GCN variants)
- **Language(s)**: English (for documentation and metadata)
- **License**: MIT License
- **Framework**: PyTorch, PyTorch Geometric
This repository contains the core implementation for the FoodFlow GNN model. For comprehensive usage instructions, detailed tutorials, and visualization tools, please visit: [https://huggingface.co/ICICLE-AI/FoodFlow_GNN_Model](https://huggingface.co/ICICLE-AI/FoodFlow_GNN_Model)

## License
MIT

## How-To Guides

### How to Implement a Hurdle Model for Trade Prediction
- **Problem**: Predicting trade flows with many zero values
- **Solution**: Implement a two-stage hurdle model that:
  1. First predicts whether trade exists between two regions
  2. Then estimates the volume of trade when it exists
- **Code example**: See the GAT model implementation in `code/model.py`
- **Results**: See the resulted models in `code/models`, note that models for different SCTG02 codes has to be trained separately

### Prerequisites
- Python 3.9+
- PyTorch (>=1.9.0)
- PyTorch Geometric (>=2.0.0)
- Other dependencies (see Hugging Face page for complete list)

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

## For Detailed Usage
- **Complete setup instructions**: [Hugging Face - Setup Instructions](https://huggingface.co/ICICLE-AI/FoodFlow_GNN_Model)
- **Data download links**: [FoodFlow Inference Data](https://drive.google.com/drive/u/0/folders/1mnlbiRvBHw4Hy2iU-i1IvElLw3Uuf0aV)
- **Information for Visualization portal**: [FoodFlowPortal](https://huggingface.co/ICICLE-AI/FoodFlow_GNN_Model)
- **Troubleshooting**: [Hugging Face - Troubleshooting](https://huggingface.co/ICICLE-AI/FoodFlow_GNN_Model)

### Direct Use
- Predicting food flows between regions using node (county/FAF zone) and edge features
- Modeling economic connectivity and transportation dependency

### Visulization Portal
- For visualization, download [processed_results](https://drive.google.com/drive/u/1/folders/1r6hdnym5wU3DOawecwknBP13VDMOcB7c) from Google Drive and save in cleaned_data folder.

- Install Dependencies

- streamlit run app.py

### Downstream Use
- Spatial forecasting of trade changes under policy shifts
- Identifying critical counties for supply chain resilience

### Out-of-Scope Use
- Real-time food trade forecasting
- Non-U.S. geographic settings without retraining

## Bias, Risks, and Limitations
- **Bias**: Model predictions depend on historical FAF data and may not reflect unexpected future disruptions (e.g., disasters, pandemics)
- **Limitations**: Prediction is limited to predefined commodity codes (SCTG1)
- **Data quality**: Assumes accuracy of FAF flow data and economic indicators

## Recommendations
Users should:
- Evaluate model generalizability before applying it to non-FAF settings
- Interpret sparse predictions carefully—zeros may result from missing data, not true absence

## Reference

### Data Sources
- **Trade Data**: [FAF5.6.1 SCTG1 commodity flow data](https://faf.ornl.gov/faf5/) (`code/data/FAF5_SCTG1.csv`)
- **Geographic Information**:
  - County shapefiles (`code/data/shapefiles/cb_2017_us_county_500k/cb_2017_us_county_500k.shp`)
  - State shapefiles (`code/data/shapefiles/cb_2018_us_state_20m/cb_2018_us_state_20m.shp`)
  - FAF zones shapefiles (`code/data/shapefiles/2017_CFS_Metro_Areas_with_FAF/2017_CFS_Metro_Areas_with_FAF.shp`)
- **Economic Indicators**: County-level economic data (`code/data/faf_features.csv`)
- **Distance Information**: FAF Distance Matrix (`code/data/FAF_Distance_Matrix.csv`)

### Acknowledgements
National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)

### Future Work
- Extending the model to handle inter-county food trade flow predictions
- Refining the model to capture more granular food trade patterns

