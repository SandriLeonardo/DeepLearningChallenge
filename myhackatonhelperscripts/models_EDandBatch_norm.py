import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import dropout_edge

from conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5,
                    JK = "last", graph_pooling = "mean",edge_drop_ratio = 0.1, batch_norm = False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
            batch_norm (bool): whether to add batch normalization after GNN layers  # ADDED: documentation
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        self.edge_drop_ratio = edge_drop_ratio
        self.batch_norm = batch_norm

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

        # Batch normalization layer for node embeddings
        if self.batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(emb_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        if self.training and self.edge_drop_ratio > 0:
            current_edge_attr = getattr(batched_data, 'edge_attr', None)

            # dropout_edge in your PyG version likely returns (edge_index, edge_mask)
            # It does not take edge_attr as a direct argument.
            new_edge_index, edge_mask = dropout_edge(
                batched_data.edge_index,
                p=self.edge_drop_ratio,
                force_undirected=True,  # Ensure this is appropriate for your graphs
                training=self.training
            )
            
            batched_data.edge_index = new_edge_index

            if current_edge_attr is not None:
                # Apply the mask to filter the edge attributes
                batched_data.edge_attr = current_edge_attr[edge_mask]
            # If current_edge_attr was None, batched_data.edge_attr remains None (or undefined)
            # and the mask isn't applied, which is correct.

        h_node = self.gnn_node(batched_data)

        # Batch normalization for node embeddings
        if self.batch_norm:
            h_node = self.batch_norm_layer(h_node)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)