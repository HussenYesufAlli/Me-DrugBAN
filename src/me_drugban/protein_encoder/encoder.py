import torch
import torch.nn as nn
import torch.nn.functional as F
# Encode each integer-encoded protein sequence into a learnable representation using an embedding and 1D convolutional layers.
# Reference: # Original DrugBAN: models.py lines 91-112

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_filters=[32,64,96], kernel_size=[7,7,7], padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernel_size[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())      # (batch, seq, embed)
        v = v.transpose(2, 1)             # (batch, embed, seq)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)  # (batch, seq', channels)
        return v