from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
import torch as th

class NodeEdge(nn.Module):
    def __init__(self,
                 embedding_dim: int = 8) -> None:
        super().__init__()

        # Aggregating node embeddings to create edge embeddings
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self,
                h_nodes: th.Tensor,
                adj_i: th.Tensor,
                adj_j: th.Tensor) -> th.Tensor:
        
        num_nodes = h_nodes.shape[1]
        # Get node embeddings using edge embeddings
        adj_i_long = adj_i[0].squeeze().long()
        batch_indices = th.arange(h_nodes.shape[0]).view(-1, 1).to(adj_i.device)
        batch_indices = batch_indices.expand(-1, adj_i_long.size(0))
        adj_i_expanded = adj_i_long.expand(batch_indices.size(0), -1)
        h_nodes_1 = h_nodes[batch_indices, adj_i_expanded]
        adj_j_long = adj_j[0].squeeze().long()
        batch_indices = th.arange(h_nodes.shape[0]).view(-1, 1).to(adj_j.device)
        batch_indices = batch_indices.expand(-1, adj_j_long.size(0))
        adj_j_expanded = adj_j_long.expand(batch_indices.size(0), -1)
        h_nodes_2 = h_nodes[batch_indices, adj_j_expanded]
        # h_nodes_1 = th.gather(h_nodes, 1, adj_i.unsqueeze(-1).expand(-1,-1,h_nodes.size(-1)).to(th.int64))
        # h_nodes_2 = th.gather(h_nodes, 1, adj_j.unsqueeze(-1).expand(-1,-1,h_nodes.size(-1)).to(th.int64))

        # new_h_nodes = th.zeros_like(h_nodes)
        # for node in range(num_nodes):
        #     adj = th.zeros(num_nodes, 1)
        #     adj[adj_i[th.where(adj_j==node)].to(th.int64)] = 1
        #     adj[adj_j[th.where(adj_i==node)].to(th.int64)] = 1
        #     adj[node] = 1
        #     th.mean(h_nodes*adj, dim=1)
        #     new_h_nodes[:,node] = th.mean(h_nodes*adj, dim=1)
        
        if not hasattr(self, 'adj'):
            adj_i = adj_i.long()
            adj_j = adj_j.long()
            adj = th.zeros(num_nodes, num_nodes)
            adj[adj_i, adj_j] = 1
            adj[adj_j, adj_i] = 1
            adj += th.eye(adj.size(0))
            adj = adj / adj.sum(dim=-1, keepdim=True)
            # adj = adj.unsqueeze(0)  # shape: (1, node_num, node_num)

            setattr(self, 'adj', adj)
        # Repeat B to match the batch size of A
        adj = getattr(self, 'adj')
        # adj = adj.repeat(h_nodes.size(0), 1, 1)  # shape: (batch_size, node_num, node_num)
        # Computing mean values
        #new_h_nodes = th.matmul(adj, h_nodes)
        new_h_nodes = th.einsum('ik,bkj->bij', adj, h_nodes)

        return self.fc(h_nodes_1+h_nodes_2), new_h_nodes


# Feature extractor
class CircuitExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Box,
                 num_gcn_layer: int = 3,
                 features_dim: int = 144, #embedding_dim*(1+1+node_num)
                 embedding_dim: int = 8) -> None:
        super().__init__(observation_space, features_dim)


        self.metadata_encoder = nn.Sequential(
            nn.Linear(10,embedding_dim), nn.ReLU()
        )

        # 3 iteration of node<->edge embedding update
        self.feature_encoder = []
        for _ in range(num_gcn_layer):
            self.feature_encoder.append(NodeEdge())
        self.feature_encoder = nn.ModuleList(self.feature_encoder)

        self.atten_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)


    def forward(self, observations: Dict) -> th.Tensor:
        features = []

        # h_metadata = self.metadata_encoder(observations["metadata"])
        # features.append(h_metadata)

        # Only use reduced mean for edge embeddings
        for layer in self.feature_encoder:
            h_edges, h_nodes = layer(observations["nodes"], observations["adj_i"], observations["adj_j"])
        h_edges_mean = th.mean(h_edges, dim=1).unsqueeze(1)
        features.append(h_edges_mean)

        # Use current node as feature
        h_current_node = th.gather(h_nodes, 1, observations["current_node"].unsqueeze(-1).expand(-1,-1,h_nodes.size(-1)).to(th.int64))
        features.append(h_current_node)

        # Use self-attention layer to use all node information
        h_atten, _ = self.atten_layer(h_nodes, h_current_node, h_current_node)
        # features.append(h_atten)
        features.append(h_atten[:,:16])

        return th.cat(features, dim=1).reshape(-1,144)


class CircuitNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int = 8,
            hidden_dim: int = 64,
            last_layer_dim_pi: int = 1024,
            last_layer_dim_vf: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # GAN-like deconvolution network
        # Creates placement image with shape of chip canvas
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 2, stride=2),
            nn.Flatten()
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim,1)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class CircuitActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CircuitNetwork(self.features_dim)