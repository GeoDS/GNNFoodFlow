import torch
import torch.nn.functional as F

# Training function
def train(model, train_data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # Get both regression output and binary classification output
    pred_value, pred_exists = model(train_data.x, train_data.edge_index, train_data.edge_attr)
    trade_exists_target = (train_data.y > 0).float()

    # Calculate regression loss (MSE) for non-zero trades
    regression_loss = criterion(pred_value, train_data.y)

    # Calculate binary classification loss (BCE)
    classification_loss = F.binary_cross_entropy(pred_exists, trade_exists_target)

    # Combined loss
    alpha = classification_loss / (classification_loss + regression_loss + 1e-8)
    loss = alpha * classification_loss + (1 - alpha) * regression_loss


    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
@torch.no_grad()
def test(model, test_data):
    model.eval()
    pred_value, pred_exists = model(test_data.x, test_data.edge_index, test_data.edge_attr)

    # Create binary target: 1 if trade value > 0, else 0
    trade_exists_target = (test_data.y > 0).float()

    # Calculate regression metrics
    mse = F.mse_loss(pred_value, test_data.y).item()
    r2 = 1 - mse / torch.var(test_data.y.reshape(pred_value.shape) if pred_value.shape[0] != test_data.y.shape[0] else test_data.y).item()

    # Calculate classification metrics
    pred_binary = (pred_exists > 0.5).float()
    accuracy = (pred_binary == trade_exists_target).float().mean().item()
    return mse, r2, accuracy

