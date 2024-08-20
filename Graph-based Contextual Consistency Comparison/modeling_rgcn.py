import torch
from torch.nn import Module
from torch_geometric.nn import RGCNConv

class RGCN(Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, 16, num_relations)
        self.conv2 = RGCNConv(16,out_channels, num_relations)
        # self.linear = nn.Linear(16 , out_channels)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        # x = global_mean_pool(x, torch.zeros(x.size(0),dtype=torch.long))
        # x = x.view(-1, 16)
        # x = self.linear(x)
        return x
