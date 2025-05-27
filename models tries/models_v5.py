import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import degree # We might need this if x features are degrees, but notebook uses zeros

# We are replacing GNN_node and GNN_node_Virtualnode for the GIN case
# by implementing the GIN stack directly in the GNN class.
# If you had other GNN types (GCN, etc.) you wanted to keep via GNN_node,
# you'd need a conditional logic, but the request is to make GNN a GIN structure.
# from src.conv import GNN_node, GNN_node_Virtualnode # No longer needed if GNN is purely GIN

class GNN(torch.nn.Module):
    def __init__(self, num_class, num_layer=5, emb_dim=300,
                 virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        '''
        GIN-specific Graph Neural Network.
        num_class (int): number of labels to be predicted
        num_layer (int): number of GIN layers
        emb_dim (int): dimensionality of embeddings
        virtual_node (bool): whether to add virtual node or not
        residual (bool): whether to add residual connections
        drop_ratio (float): dropout ratio
        JK (str): Jumping Knowledge connection type ("last", "sum", "cat")
        graph_pooling (str): type of graph pooling ("sum", "mean", "max", "attention", "set2set")
        '''
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        self.residual = residual
        self.virtual_node = virtual_node

        if self.num_layer < 1: # Can be 1 if JK is 'last' and no complex interactions are needed
            raise ValueError("Number of GNN layers must be at least 1.")

        # Initial node embedding
        # The notebook uses add_zeros, making data.x = torch.zeros(num_nodes, dtype=torch.long)
        # This means all nodes initially get the same embedding learned for index 0.
        self.atom_encoder = torch.nn.Embedding(1, emb_dim)

        # List of GIN convolutions
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layer):
            # MLP for GINConv: maps emb_dim to emb_dim
            # A common GIN MLP structure: Linear -> BN -> ReLU -> Linear
            # Sometimes a deeper MLP is used for more expressiveness.
            mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim), # Using BatchNorm in MLP for stability
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )
            # train_eps=True allows learning the epsilon parameter in GIN
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # Virtual node components
        if self.virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0) # Initialize virtual node to 0

            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(self.num_layer):
                # MLP to update virtual node embedding from aggregated graph features
                self.mlp_virtualnode_list.append(
                    nn.Sequential(
                        nn.Linear(emb_dim, 2 * emb_dim),
                        nn.BatchNorm1d(2 * emb_dim),
                        nn.ReLU(),
                        nn.Linear(2 * emb_dim, emb_dim),
                        nn.BatchNorm1d(emb_dim),
                        nn.ReLU()
                    )
                )

        # Jumping Knowledge aggregation
        if self.JK == "cat":
            jk_input_dim = emb_dim * num_layer
        else: # "last" or "sum"
            jk_input_dim = emb_dim
        
        # If JK is "cat", we might need a linear layer to project back to emb_dim before pooling
        # or adjust the input to the graph_pred_linear accordingly.
        # For simplicity, if JK is "cat", let's assume pooling happens on concatenated features,
        # and graph_pred_linear input dim will be jk_input_dim (or 2*jk_input_dim for set2set).
        # However, typical JK is applied on node embeddings before pooling.
        # Let's keep it simple: if JK="cat", the pool function will get concatenated features.


        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            # Attention mechanism input dim depends on JK output
            self.pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(jk_input_dim, 2 * jk_input_dim),
                                                              nn.BatchNorm1d(2*jk_input_dim), nn.ReLU(),
                                                              nn.Linear(2 * jk_input_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(jk_input_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # Final linear layer for graph classification
        if self.graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * jk_input_dim, self.num_class)
        else:
            self.graph_pred_linear = nn.Linear(jk_input_dim, self.num_class)

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        # Initial node features
        # x from add_zeros is [num_nodes_in_batch], dtype=torch.long
        # Squeeze if it has an extra dimension, e.g. [N, 1] -> [N]
        if x.dim() > 1 and x.size(1) == 1:
            x = x.squeeze(1)
        h = self.atom_encoder(x) # Now h has shape [num_nodes_in_batch, emb_dim]

        if self.virtual_node:
            # Initialize virtual node embedding for each graph in the batch
            virtualnode_emb = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        h_list = [h] # For Jumping Knowledge. initialize with first embedding

        for layer in range(self.num_layer):
            if self.virtual_node:
                h_list[layer] = h_list[layer] + virtualnode_emb[batch] # to add BEFORE conv
            
            h = self.convs[layer](h_list[layer], edge_index) # GIN convolution
            h = self.batch_norms[layer](h) # Batch normalization

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training) # Dropout only on last layer
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training) # ReLU and dropout on intermediate layers

            if self.residual:
                h = h + h_list[layer]  # Residual connection

            h_list.append(h)  # Store the output for Jumping Knowledge

            if self.virtual_node and layer < self.num_layer - 1:
                # Update virtual node for next layer
                virtualnode_emb_temp = global_add_pool(h_list[layer], batch) + virtualnode_emb
                virtualnode_emb = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_emb_temp),
                    self.drop_ratio, training=self.training
                )

        # Jumping Knowledge aggregation
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.stack(h_list, dim=0).sum(dim=0)
        elif self.JK == "cat":
            node_representation = torch.cat(h_list, dim=1)
        else: # Should not happen if JK is validated in init
            node_representation = h_list[-1]


        # Graph pooling
        h_graph = self.pool(node_representation, batch)

        return self.graph_pred_linear(h_graph)