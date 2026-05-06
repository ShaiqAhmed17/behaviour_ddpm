import torch
from torch import nn
from torch import Tensor as _T

from typing import List


class VectoralResidualModel(nn.Module):
    """
    ϵ_θ (x_t, t, Z)
    """

    def __init__(
        self,
        state_space_size: int,
        recurrence_hidden_layers: List[int],
        input_size: int,
        time_embedding_size: int,
        nonlin_first: bool,
        include_time: bool = False
    ) -> None:

        super().__init__()

        self.input_size = input_size
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        self.time_embedding_size = time_embedding_size
        self.nonlin_first = nonlin_first
        self.include_time = include_time

        recurrence_layers = self.generate_layers(recurrence_hidden_layers, include_time)
        self.layers = nn.ModuleList(recurrence_layers)  # R^N -> R^N

    def generate_layers(self, recurrence_hidden_layers, include_time = True) -> List[nn.Module]:
        all_layer_sizes = recurrence_hidden_layers + [self.state_space_size]
        recurrence_layers = []
        total_input_size = self.state_space_size + self.input_size
        if include_time:
            total_input_size = total_input_size + self.time_embedding_size
        recurrence_layers.extend([nn.Linear(total_input_size, all_layer_sizes[0]), nn.Softplus()])  # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            recurrence_layers.extend(
                [nn.Linear(all_layer_sizes[i], op_s), nn.Softplus()]
            )
        recurrence_layers = recurrence_layers[:-1]

        return recurrence_layers

    @staticmethod
    def unsqueeze_start_dims(tensor: _T, start_dims: List[int]):
        return tensor[*[None for _ in start_dims]].expand(
            *start_dims, *tensor.shape
        )

    def concatenate_with_time_and_input(
        self, x: _T, t_embeddings_schedule: _T, input_vector: _T
    ) -> _T:
        """
        x of shape [..., T, layer_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size]
        """
        if self.include_time:
            reshaped_t_schedule = self.unsqueeze_start_dims(
                t_embeddings_schedule, x.shape[:-2]
            )
            x_concat = torch.concat(
                [
                    x,
                    reshaped_t_schedule.to(x.device, x.dtype),
                    input_vector.to(x.device, x.dtype),
                ],
                -1,
            )
        else:
            x_concat = torch.concat(
                [
                    x,
                    input_vector.to(x.device, x.dtype),
                ],
                -1,
            )
        return x_concat

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size],             <s> passed to all </s>

        x[...,i,:] is x_{t_embeddings_schedule[i]}
            e.g.
                if t_embeddings_schedule[i] = 1, then x[...,i,:] is x_1
                if t_embeddings_schedule[i] = 5, then x[...,i,:] is x_5
            But actually, the numerical values of t_embeddings_schedule won't be the integer values printed here of course
        """
        if self.nonlin_first:
            x = torch.nn.functional.softplus(x)
        x = self.concatenate_with_time_and_input(x, t_embeddings_schedule, input_vector)
        for layer in self.layers:
            x = layer(x)
        return x



class DynamicalSystemTransitionModel(nn.Module):
    """
    Designed to be used for the HVAE setting, where ϵ_θ is interpreted as the transition kernel,
        i.e. k[x]_u^n
    
    Requires k, n and tau parameters, and never has deep layers.

    Takes in t_embeddings_schedule to forward but ignores it.
    Takes in input_vector but just linearly adds it, as in the dynamical system case

    Shapes of forward same as above
    """
    
    def __init__(
        self,
        state_space_size: int,
        input_size: int,
        time_embedding_size: int,
        k: float = 1.0,
        n: int = 1,
        tau: float = 0.1,
        init_spectral_radius: float = 0.9,
    ) -> None:
        
        super().__init__()
        
        self.state_space_size = state_space_size
        self.input_size = input_size
        self.time_embedding_size = time_embedding_size
        self.init_spectral_radius = init_spectral_radius
        
        # Dynamical system parameters
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor(n, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))

        # Weight matrix W: [state_space_size, state_space_size]
        self.W = nn.Parameter(torch.randn(state_space_size, state_space_size))
        
        # Input projection: h -> state_space (if input_size != state_space_size)
        if input_size != state_space_size:
            self.input_projection = nn.Linear(input_size, state_space_size, bias=False)
        else:
            self.input_projection = None
            
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize W to be stable with specified spectral radius"""
        with torch.no_grad():
            # Initialize W with desired spectral radius
            eigvals = torch.linalg.eigvals(self.W)
            spectral_radius = eigvals.abs().max()
            self.W.data = self.W.data * (self.init_spectral_radius / spectral_radius)
    
    def f(self, u: torch.Tensor) -> torch.Tensor:
        """
        Threshold power-law nonlinearity: f(u) = k * [u]_+^n
        
        Args:
            u: [..., state_space_size]
        Returns:
            r: [..., state_space_size]
        """
        if self.n == -1:  # Linear case
            return self.k * u
        else:
            # Threshold power law: k * [u]_+^n
            u_thresh = torch.clamp(u, min=0.0)
            return self.k * torch.pow(u_thresh, self.n)
    
    def transition_kernel(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the dynamical system transition: k[x]_u^n
        
        This represents the "drift" term in the SDE:
        du/dt = (-u + h + W*f(u))/tau
        
        Args:
            u: [..., state_space_size] - current state
            h: [..., state_space_size] - external input
        Returns:
            transition: [..., state_space_size] - transition kernel output
        """
        # Apply nonlinearity
        r = self.f(u)  # [..., state_space_size]
        
        # Compute recurrent term: W @ r
        # Handle batch dimensions by using einsum or matrix multiply
        W_r = torch.einsum('ij,...j->...i', self.W, r)
        
        # Dynamical system equation: (-u + h + W*r) / tau
        transition = (-u + h + W_r) / self.tau
        
        return transition
    
    @staticmethod
    def unsqueeze_start_dims(tensor: torch.Tensor, start_dims: List[int]) -> torch.Tensor:
        return tensor[*[None for _ in start_dims]].expand(
            *start_dims, *tensor.shape
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        t_embeddings_schedule: torch.Tensor, 
        input_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the dynamical system transition model.
        
        Args:
            x: [..., T, state_space_size] - state trajectory
            t_embeddings_schedule: [T, time_emb_size] - time embeddings (ignored)
            input_vector: [..., T, input_size] - external inputs
            
        Returns:
            transition: [..., T, state_space_size] - transition kernel outputs
        """
        # Project input if necessary
        if self.input_projection is not None:
            h = self.input_projection(input_vector)  # [..., T, state_space_size]
        else:
            h = input_vector  # [..., T, state_space_size]
        
        # Apply transition kernel to each time step
        # x: [..., T, state_space_size]
        # h: [..., T, state_space_size]
        
        batch_shape = x.shape[:-2]  # [...] 
        T = x.shape[-2]
        
        # Reshape for processing: [batch_total, T, state_space_size]
        x_flat = x.view(-1, T, self.state_space_size)
        h_flat = h.view(-1, T, self.state_space_size)
        
        # Apply transition kernel to each time step
        transitions = []
        for t in range(T):
            u_t = x_flat[:, t, :]  # [batch_total, state_space_size]
            h_t = h_flat[:, t, :]  # [batch_total, state_space_size]
            
            transition_t = self.transition_kernel(u_t, h_t)  # [batch_total, state_space_size]
            transitions.append(transition_t)
        
        # Stack and reshape back: [..., T, state_space_size]
        transitions = torch.stack(transitions, dim=1)  # [batch_total, T, state_space_size]
        transitions = transitions.view(*batch_shape, T, self.state_space_size)
        
        return transitions

    def get_W_matrix(self) -> torch.Tensor:
        """Get the current weight matrix"""
        return self.W.data
