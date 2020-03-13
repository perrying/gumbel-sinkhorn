import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, hid_c, out_c):
        super().__init__()

        self.g1 = nn.Sequential(
            nn.Conv1d(1, hid_c, 1),
            nn.ReLU(True)
        )

        self.g2 = nn.Sequential(
            nn.Conv1d(hid_c, out_c, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # NOTE: x is 2-dim or 3-dim torch.Tensor, i.e., (batch, 1, numbers)
        if x.dim() == 2:
            x = x[:, None]
        h = self.g1(x)
        log_alpha = self.g2(h).transpose(1,2).contiguous()
        return log_alpha