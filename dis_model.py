import torch
import torch.nn as nn

class Dis_Classifier(nn.Module):
    def __init__(self, feature_dim: int, n_convolutions: int, kernel_size: int, hidden_dim: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_convolutions):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=feature_dim,
                        out_channels=feature_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int((kernel_size - 1) / 2),
                        dilation=1,
                        # w_init_gain='relu'
                    ),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                )
            )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm.flatten_parameters() # to optimize weights and operations (keep in forward method if using nn.DataParallel)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, n_frames, n_feats] -> [B, n_feats, n_frames]
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2) # [B, n_feats, n_frames] -> [B, n_frames, n_feats]
        outputs, (h_n, c_n) = self.lstm(x) # [B, L, D * H_out], ([D * n_layers, N, H_out], [D * n_layers, N, H_cell])
        h_n = h_n.transpose(0, 1) # [2, B, hidden_dim] -> [B, 2, hidden_dim]
        h_n = torch.flatten(h_n, start_dim=1) # [B, 2, hidden_dim] -> [B, 2 * hidden_dim]
        output = self.fc(h_n).reshape(-1)
        return output
