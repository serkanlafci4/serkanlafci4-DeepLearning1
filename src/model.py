import torch.nn as nn


class MLP(nn.Module):
    """Tam bağlantılı (fully-connected) çok katmanlı algılayıcı.

    Girdi: 28x28 gri görüntü -> 784 boyutlu vektöre düzleştirilir.
    Her hidden layer: Linear -> ReLU -> (opsiyonel) Dropout
    Çıkış: 10 sınıf için logit (softmax CrossEntropyLoss içinde uygulanır).
    """

    def __init__(self, hidden_sizes=(256, 128), dropout=0.0, input_size=28 * 28, num_classes=10):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("Ödev gereği en az 2 hidden layer kullanılmalıdır.")

        layers = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
