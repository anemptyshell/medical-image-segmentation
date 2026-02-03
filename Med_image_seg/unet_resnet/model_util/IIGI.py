import torch.nn as nn
import torch
import math
import os
# import h5py
import torch.nn.functional as F

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class Exchange(nn.Module):
    def __init__(self,in_channels):
        super(Exchange, self).__init__()
        self.in_channel = in_channels

    def forward(self, x, bn, bn_threshold):
        bn1, bn2, bn3 = bn[0].weight.abs(), bn[1].weight.abs(), bn[2].weight.abs()
        x1, x2, x3 = torch.zeros_like(x[0]), torch.zeros_like(x[1]), torch.zeros_like(x[2])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = (x[1][:, bn1 < bn_threshold] + x[2][:, bn1 < bn_threshold])/2
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = (x[0][:, bn2 < bn_threshold] + x[2][:, bn2 < bn_threshold])/2
        x3[:, bn3 >= bn_threshold] = x[2][:, bn3 >= bn_threshold]
        x3[:, bn3 < bn_threshold] = (x[0][:, bn3 < bn_threshold] + x[1][:, bn3 < bn_threshold])/2

        return [x1, x2, x3]

class intra_graph(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(intra_graph, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv2 = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))
        self.graphnet = GraphNet(32,dim)

    def forward(self, x):
        batch, channel, h, w = x.size()
        nodes_graph,assign = self.graphnet(x)
        nodes_graph = self.conv1(nodes_graph.unsqueeze(3)).squeeze(3)
        nodes_graph = self.gcn(nodes_graph)
        x_out = nodes_graph.bmm(assign) # reprojection
        x_out = self.conv2(x_out.unsqueeze(3)).squeeze(3)
        x_out = x + x_out.view(batch, channel, h, w)
        return x_out

class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    # def init(self, initcache):
    #     if not os.path.exists(initcache):
    #         print(initcache + ' not exist!!!\n')
    #     else:
    #         with h5py.File(initcache, mode='r') as h5:
    #             clsts = h5.get("centroids")[...]
    #             traindescs = h5.get("descriptors")[...]
    #             self.init_params(clsts, traindescs)
    #             del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) #across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert (loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k
        x = self.relu(x)
        return x

class Interaction(nn.Module):
    def __init__(self, in_channels, out_channels, num_parallel=3, bn_threshold=2e-2):
        super(Interaction, self).__init__()
        self.inp_dim = in_channels
        self.num_parallel = num_parallel
        self.bn_threshold = bn_threshold

        self.conv1 = ModuleParallel(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = ModuleParallel(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3 = ModuleParallel(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))

        self.relu = ModuleParallel(nn.ReLU(inplace=False))

        self.exchange = Exchange(in_channels)

        self.bn1 = BatchNorm2dParallel(in_channels)
        self.bn2 = BatchNorm2dParallel(in_channels)
        self.bn3 = BatchNorm2dParallel(out_channels)

        self.intra_GCA1 = intra_graph(out_channels)
        self.intra_GCA2 = intra_graph(out_channels)
        self.intra_GCA3 = intra_graph(out_channels)
        self.inter_GCA = inter_gcn(out_channels)

        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        # print(len(out))  ## 2
        # print(out[0].size())  ## torch.Size([512, 64, 64])
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)
        ## cbr * 2

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        Fe_v,Fe_d,Fe_t = self.inter_GCA(out[0], out[1], out[2])


        intra_out1 = self.intra_GCA1(Fe_v)
        intra_out2 = self.intra_GCA2(Fe_d)
        intra_out3 = self.intra_GCA3(Fe_t)

        out = [intra_out1, intra_out2, intra_out3]


        return out


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=3):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class inter_gcn(nn.Module):
    def __init__(self, in_channel, GNN=False):
        super(inter_gcn, self).__init__()
        self.in_channel = in_channel
        self.gcn = Channel_gcn(in_channel, 48)

    def forward(self, F_v,F_d,F_t):
        Fe_v,Fe_d,Fe_t = self.gcn(F_v,F_d,F_t)
        Fe_v = F_v + Fe_v
        Fe_d = F_d + Fe_d
        Fe_t = F_t + Fe_t

        return Fe_v,Fe_d,Fe_t

class Channel_gcn(nn.Module):
    def __init__(self, dim_a, num):  # 64 64 64 64
        super(Channel_gcn, self).__init__()
        dim = dim_a * 3  # 64*6

        self.gcn = CascadeGCNet(dim//num, loop=2)  # 邻接矩阵size 输入节点维度 输出节点维度 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc_1 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )

        self.num = num
        self.dim_a = dim_a

    def forward(self, F_v,F_d,F_t):
        batch, channel, _, _ = F_v.size()
        combined = torch.cat([F_v, F_d, F_t], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, self.num, -1).permute(0, 2, 1)

        feat_cat = self.gcn(combined_fc)
        feat_cat = feat_cat.view(batch, -1)

        excitation1 = self.fc_1(feat_cat[:,:self.dim_a]).view(batch, channel, 1, 1)
        excitation2 = self.fc_2(feat_cat[:,self.dim_a:self.dim_a*2]).view(batch, channel, 1, 1)
        excitation3 = self.fc_3(feat_cat[:,self.dim_a*2:]).view(batch, channel, 1, 1)

        return excitation1 * F_v, excitation2 * F_d, excitation3 * F_t
    


## 
# interaction = Interaction(in_channels=64, out_channels=64)
# a = torch.rand(1, 64, 64, 64)
# input = [a, a, a]
# output = interaction(input)
# print(len(output))
# print(output[0].size())

