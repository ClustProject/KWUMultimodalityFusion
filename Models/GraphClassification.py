import torch.nn as nn
from torch_geometric.nn import GCNConv

class MSGCN(nn.Module):
    def __init__(self, num_nodes, num_features, gcn1_channels, gcn2_channels, gcn3_channels, fc1_channels, out_channels,
                 edge_weight, batch_size, learnable=False):
        super(MSGCN, self).__init__()
        self.in_channels = num_features
        self.gcn1_out_channels = gcn1_channels
        self.gcn2_out_channels = gcn2_channels
        #         self.fc1_in_channels = (gcn1_channels + gcn2_channels) * num_nodes
        self.fc1_in_channels = gcn2_channels * num_nodes
        self.fc1_out_channels = fc1_channels
        self.out_channels = out_channels

        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learnable)
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy = nn.CrossEntropyLoss()
        self.activation = nn.CELU()
        self.dropout = nn.Dropout(0.1)

        self.bn1d = nn.BatchNorm1d(self.in_channels)
        self.gcn1 = GCNConv(self.in_channels, self.gcn1_out_channels)
        self.gcn2 = GCNConv(self.gcn1_out_channels, self.gcn2_out_channels)
        self.fc1 = nn.Linear(self.fc1_in_channels, self.fc1_out_channels)
        self.fc2 = nn.Linear(self.fc1_out_channels, self.out_channels)
        self.fc_module = nn.Sequential(self.fc1, self.dropout, self.activation, self.fc2)

    def forward(self, data, _type=None):
        data.edge_attr = self.edge_weight.data.repeat(self.batch_size)

        bn = self.bn1d(data.x)
        gcn1 = self.gcn1(bn, data.edge_index, data.edge_attr)
        gcn2 = self.gcn2(self.activation(gcn1), data.edge_index, data.edge_attr)

        """need to be checked"""
        gcn = self.activation(gcn2).reshape(self.batch_size, -1)

        logits = self.fc_module(gcn)
        #         print("edge: ", data.edge_attr)
        #         print("data: ", data.x)
        #         print("BN: ", bn)
        #         print("gcn1: ", gcn1)
        #         print("gcn2: ", gcn2)
        #         print("gcn3: ", gcn3, gcn3.shape)
        #         print("flatten: ", gcn, gcn.shape)
        #         print("fc: ", fc)

        return logits