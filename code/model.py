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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
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

        # Trade existence classifier (returning logits)
        self.trade_classifier_logits = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels, 1)  # No sigmoid here
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
        for layer_group in [self.edge_encoder, self.edge_features,
                            self.trade_classifier_logits, self.value_regressor]:
            for layer in layer_group:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, mode='fan_in',
                                         nonlinearity='leaky_relu')

    def forward(self, x, edge_index, edge_attr, return_logits=False):
        # Process edge features
        edge_features_encoded = self.edge_encoder(edge_attr)

        # GCN layers
        h1 = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), negative_slope=0.01)
        h3 = F.leaky_relu(self.conv3(h2, edge_index), negative_slope=0.01)

        # Prepare edge-level features
        src, dst = edge_index
        src_feat = h3[src]
        dst_feat = h3[dst]
        combined_features = torch.cat([src_feat, dst_feat, edge_features_encoded], dim=1)

        # Edge-wise hidden representation
        common_features = self.edge_features(combined_features)

        # Predict trade existence (logits)
        trade_logits = self.trade_classifier_logits(common_features)
        trade_probs = torch.sigmoid(trade_logits)

        # Predict trade value
        trade_value = self.value_regressor(common_features)
        
        # Final prediction = trade probability * value
        final_pred = trade_probs * trade_value

        if return_logits:
            return final_pred, trade_probs, trade_logits
        else:
            return final_pred, trade_probs



# Improve the model architecture with better design
class ImprovedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim, num_layers=3, dropout=0.2):
        super(ImprovedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Multiple GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Edge feature processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4)
        )

        # Hurdle model components
        # Binary classifier for trade existence
        self.trade_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + hidden_channels // 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        # Regression for trade value (when trade exists)
        self.value_regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + hidden_channels // 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.ReLU()
        )

        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings through GCN layers
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = self.batch_norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection (except for first layer)
            if i > 0 and h.shape[1] == h_new.shape[1]:
                h = h + h_new
            else:
                h = h_new

        # Process edge features
        edge_emb = self.edge_mlp(edge_attr)

        # Get source and target node embeddings
        src_emb = h[edge_index[0]]
        dst_emb = h[edge_index[1]]

        # Combine embeddings
        edge_input = torch.cat([src_emb, dst_emb, edge_emb], dim=1)

        # Hurdle model predictions
        trade_exists = self.trade_classifier(edge_input)
        trade_value = self.value_regressor(edge_input)

        # Final prediction: probability of trade * predicted value
        final_pred = trade_exists * trade_value

        return final_pred, trade_exists