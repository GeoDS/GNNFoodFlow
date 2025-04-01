import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.nn import GATConv, GCNConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, edge_dim=edge_feature_dim)
        self.conv2 = GATConv(hidden_channels, hidden_channels * 2, edge_dim=edge_feature_dim)
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels * 4, edge_dim=edge_feature_dim)

        # Common feature extraction layers
        self.edge_features = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 2 + edge_feature_dim, hidden_channels * 4),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LeakyReLU(0.01)
        )

        # Binary classification (hurdle) layer - predicts if trade exists
        self.trade_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # Regression layer - predicts trade value if trade exists
        self.value_regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels, 1)
        )

        # Initialize weights
        for layer_group in [self.edge_features, self.trade_classifier, self.value_regressor]:
            for layer in layer_group:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, mode='fan_in',
                                        nonlinearity='leaky_relu' if isinstance(layer, nn.LeakyReLU) else 'linear')

    def forward(self, x, edge_index, edge_attr):

        # Update node embeddings through GNN layers
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = F.leaky_relu(h1, negative_slope=0.01)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = F.leaky_relu(h2, negative_slope=0.01)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = F.leaky_relu(h3, negative_slope=0.01)

        # For each edge, get source and target node embeddings
        src, dst = edge_index
        src_feat = h3[src]
        dst_feat = h3[dst]

        # Concatenate source, target embeddings with edge features
        edge_features = torch.cat([src_feat, dst_feat, edge_attr], dim=1)

        # Extract common features
        common_features = self.edge_features(edge_features)
        # Predict whether trade exists (binary classification)
        trade_exists = self.trade_classifier(common_features)

        # Predict trade value (regression)
        trade_value = self.value_regressor(common_features)
        # Apply hurdle model: final prediction is trade_exists * trade_value
        final_pred = trade_exists * trade_value

        return final_pred, trade_exists

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim):
        super(GCN, self).__init__()

        # GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.LeakyReLU(0.01)
        )

        # Edge features processing
        self.edge_features = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 2 + hidden_channels, hidden_channels * 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        # Trade existence classifier
        self.trade_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # Trade value regressor
        self.value_regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for layer_group in [self.edge_encoder, self.edge_features, self.trade_classifier, self.value_regressor]:
            for layer in layer_group:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, x, edge_index, edge_attr):
        # Process edge features separately
        edge_features_encoded = self.edge_encoder(edge_attr)

        # Node embedding updates (without edge features in GCNConv)
        h1 = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), negative_slope=0.01)
        h3 = F.leaky_relu(self.conv3(h2, edge_index), negative_slope=0.01)

        # Get source and target node features for each edge
        src, dst = edge_index
        src_feat = h3[src]
        dst_feat = h3[dst]

        # Combine node features with encoded edge features
        combined_features = torch.cat([src_feat, dst_feat, edge_features_encoded], dim=1)

        # Process combined features
        common_features = self.edge_features(combined_features)

        # Predict trade existence
        trade_exists = self.trade_classifier(common_features)

        # Predict trade value
        trade_value = self.value_regressor(common_features)

        # Hurdle model: final prediction is trade_exists * trade_value
        final_pred = trade_exists * trade_value

        return final_pred, trade_exists