import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def get_weights(self):
        with torch.no_grad():
            return self.linear.weight.data.clone().cpu().numpy().flatten(), self.linear.bias.data.clone().cpu().item()

    def set_weights(self, weight, bias):
        with torch.no_grad():
            self.linear.weight.data = torch.tensor(weight, dtype=torch.float32).unsqueeze(0).to(self.linear.weight.device)
            self.linear.bias.data = torch.tensor([bias], dtype=torch.float32).to(self.linear.bias.device)

    def get_named_parameters(self):
        return list(self.named_parameters())
