import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

class Layer_AGG(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.6,weight=1,num_layers =2,layers_tree=2):
        super(Layer_AGG, self).__init__()
        self.drop_rate = drop_rate
        self.weight = weight
        self.num_layers = num_layers
        self.layers_tree = layers_tree
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_feat if i==0 else out_feat
            self.convs.append(SAGEConv(in_channels,out_feat))
        self.conv_tree = nn.ModuleList()
        self.gating_networks = nn.ModuleList()
        for i in range(0,layers_tree):
            self.conv_tree.append(SAGEConv(in_feat,out_feat))
            self.gating_networks.append(nn.Linear(out_feat, 1))
        self.bias = nn.Parameter(torch.zeros(layers_tree))  
        


    def forward(self, x, edge_index):
        h = x
        layer_outputs = []
        
        # 计算每层的节点嵌入
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index[0])
            if i != self.num_layers - 1:  # No activation and dropout on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        # 遍历层级树的卷积操作
        for i in range(0,self.layers_tree):
            temp = self.conv_tree[i](h,edge_index[1][i])
            temp = F.relu(temp)
            temp = F.dropout(temp,p=self.drop_rate,training=self.training)
            layer_outputs.append(temp)
        # print(layer_outputs[0].shape)


        
        # print(weighted_sums[0].shape)
        
        if self.layers_tree >= 1:
            weighted_sums = [self.gating_networks[i](layer_outputs[i]) for i in range(self.layers_tree)]
            alpha = F.softmax(torch.stack(weighted_sums, dim=-1), dim=-1)

            # print(alpha.shape)
            x_tree = torch.zeros_like(layer_outputs[0])  
            for i in range(self.layers_tree):
            
                weight = alpha[:, :, i]  
                x_tree += layer_outputs[i] * weight

            return x+self.weight*x_tree
        else:
            return x


class Layer_AGG_Edge(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.6,num_layers=2,):
        super(Layer_AGG_Edge, self).__init__()
        self.drop_rate = drop_rate
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_feat if i==0 else out_feat
            self.convs.append(GCNConv(in_channels,out_feat))


    def forward(self, x, edge_index, edge_attr):
        # print("-----------------")
        # print(x)
        # print(edge_index)
        # print(edge_attr)
        # print("====================")
        # print(f"Edge attributes shape: {edge_attr.shape}")
        # print(f"Edge index shape: {edge_index.shape}")
        # print(f"Edge attributes shape: {edge_attr.shape}")
        # 确保 edge_attr 是 Tensor 且形状正确
        if not isinstance(edge_attr, torch.Tensor):
            raise TypeError(f"Expected edge_attr to be a torch.Tensor, but got {type(edge_attr)}")
        
        # edge_attr 是 [num_edges, 1] 或 [num_edges] 的形状，无需额外转换
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)  # 如果是 1D, 则加一个维度，变成 2D
        
        # 计算每层的节点嵌入
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_attr)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        # print("-----------------")
        # print(x)
        # print("====================")
        # sys.exit(0)
        return x


class GCNWithEdgeWeights(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.6, layers = 2, relation_nums = 3):
        super(GCNWithEdgeWeights, self).__init__()
        self.relation_nums = relation_nums
        self.drop_rate = drop_rate
        for i in range(relation_nums):
            setattr(self,'graph'+str(i),Layer_AGG_Edge(in_feat,out_feat,self.drop_rate,layers))
        

    def forward(self, x, edge_index, edge_attr):
        
        layer_outputs = []
        
        for i in range(self.relation_nums):
            layer_output = getattr(self, 'graph' + str(i))(x[i], edge_index[i][0], edge_attr[i])
            layer_outputs.append(layer_output)
            
        return layer_outputs



class multi_SEFraud_Model(nn.Module):
    def __init__(self,in_feat,out_feat,relation_nums = 3, hidden = 32,drop_rate=0.6,weight = 1,num_layers = 2,layers_tree=2, hetero_layers=2):
        super(multi_SEFraud_Model, self).__init__()
        self.relation_nums=relation_nums
        self.drop_rate = drop_rate
        self.weight = weight
        self.layers_tree = layers_tree
        self.hetero_layers = hetero_layers
        for i in range(relation_nums):
            setattr(self,'Layers'+str(i),Layer_AGG(in_feat,hidden,self.drop_rate,self.weight,num_layers,self.layers_tree))
        self.linear=nn.Linear(hidden*relation_nums,out_feat)
        
        # 设置节点类型编码
        self.node_type_embedding = nn.Embedding(self.relation_nums, hidden)
        
        # 设置边类型编码
        self.edge_type_embedding = nn.Embedding(self.relation_nums, hidden)
        
        # 设置属性类型 和边类型 掩码
        for i in range(self.relation_nums):
            setattr(self, 'feat_mask_' + str(i), nn.Linear(2*hidden+in_feat,in_feat))
            
            setattr(self, 'edge_mask_' + str(i), nn.Linear(2*hidden+in_feat,1))
        
        # 用于计算边属性的 MLP
        self.edge_attr_mlp = nn.Sequential(
            nn.Linear(3 * hidden, in_feat),
            nn.ReLU(),
            nn.Linear(in_feat, 1)
        )
        
        self.hetro_conv = GCNWithEdgeWeights(in_feat, hidden,self.drop_rate,self.hetero_layers,self.relation_nums)


    def forward(self, x, edge_index):

        layer_outputs = []
        edge_attrs = []  # 存储每层计算的边属性

        for i in range(self.relation_nums):

            node_type = self.node_type_embedding(torch.LongTensor([i]).to(device=x.device))
            edge_type = self.edge_type_embedding(torch.LongTensor([i]).to(device=x.device))
            
            node_type = node_type.repeat(x.shape[0], 1)
            edge_type = edge_type.repeat(edge_index[i][0].shape[1], 1)
            
            layer_output = getattr(self, 'Layers' + str(i))(x, edge_index[i])
                    
            # 计算边的属性
            src, dst = edge_index[i][0]  # 获取边的起点和终点索引
            edge_features = torch.cat([layer_output[src], layer_output[dst], edge_type], dim=-1)  # 拼接两端点特征
            edge_mask = self.edge_attr_mlp(edge_features)  # 使用 MLP 映射边特征
               
            # 将node_type的形状变成和x一致
            # 或者使用 repeat 方法
            # repeat：数据被真正复制，适合后续修改或独立处理每一行的场景。
            # expand：不会复制数据，节省内存，适用于无需修改每行的场景。
            # layer_output 拼接上x 和节点类型编码
            # print(f"Layer output shape: {layer_output.shape}")

            hetro_output = torch.cat([layer_output, x, node_type], dim=1)           

            feat_mask = getattr(self, 'feat_mask_' + str(i))(hetro_output)
            # print(f"Feature mask shape: {feat_mask.shape}")
            # sys.exit(0)
            
            # print(f"Feature mask: {feat_mask}")
            layer_outputs.append(x*feat_mask)
            
            # edge_attrs.append(edge_mask)  # 记录每层的边属性
            
            edge_attr_min = torch.min(edge_mask)
            edge_attr_max = torch.max(edge_mask)
            edge_attr_normalized = (edge_mask - edge_attr_min) / (edge_attr_max - edge_attr_min)
            
            edge_attrs.append(edge_attr_normalized)  # 记录每层的边属性
            
            # print(f"edge_attr mean: {torch.mean(edge_mask)}")
            # print(f"edge_attr std: {torch.std(edge_mask)}")
            # print(f"edge_attr min: {torch.min(edge_mask)}")
            # print(f"edge_attr max: {torch.max(edge_mask)}")


        # 这里之前都是有数值的，过了下面的代码就没有数值了
        hetro_conv_output = self.hetro_conv(layer_outputs, edge_index, edge_attrs)
        # print(f"Hetero conv output shape: {len(hetro_conv_output)}")
        # print(f"Hetero conv output shape: {hetro_conv_output[0].shape}")
        # print(f"Hetero conv output: {hetro_conv_output[0]}")
        # print("--------------------------------------")

        x_temp = torch.cat(hetro_conv_output, dim=1)

        x = self.linear(x_temp)
        
        x = torch.clamp(x, min=-1e10, max=1e10)  # 将x的值限制在合理的范围内
        x = F.log_softmax(x, dim=1)
        return x,x_temp


class Graphsage(nn.Module):
    def __init__(self, in_feat,out_feat):
        super(Graphsage, self).__init__()
        self.conv1 = SAGEConv(in_feat, out_feat)
        self.conv2 = SAGEConv(out_feat, out_feat)
        # self.conv1 = GCNConv(in_feat, out_feat)
        # self.conv2 = GCNConv(out_feat, out_feat)
        self.linear = nn.Linear(out_feat,2)


    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x,dim=1)
        return x