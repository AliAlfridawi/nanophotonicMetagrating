import torch
import torch.nn as nn

class SurrogateMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) designed to act as a surrogate electromagnetic solver.
    It takes 1D metagrating geometric parameters and predicts the optical spectra.
    """
    def __init__(self, input_dim=6, output_dim=22):
        super(SurrogateMLP, self).__init__()
        
        # We use nn.Sequential to keep the architecture clean and readable.
        # The hidden layer sizes (64, 128) can be tuned later (hyperparameter optimization).
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, 64),
            nn.GELU(),  # GELU often performs slightly better than ReLU in physics-informed tasks
            
            # Hidden Layer 1
            nn.Linear(64, 128),
            nn.GELU(),
            
            # Hidden Layer 2
            nn.Linear(128, 128),
            nn.GELU(),
            
            # Hidden Layer 3
            nn.Linear(128, 64),
            nn.GELU(),
            
            # Output Layer
            nn.Linear(64, output_dim),
            
            # The Sigmoid activation ensures the final output values are strictly 
            # bounded between 0.0 and 1.0, matching the physical reality of 
            # transmission and reflection efficiencies.
            nn.Sigmoid() 
        )

    def forward(self, x):
        """
        The forward pass of the network. 
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, input_dim) containing scaled geometries.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, output_dim) containing predicted spectra.
        """
        return self.network(x)

# --- Quick Test Block ---
# This block only runs if you execute this specific file directly.
# It's a great way to verify your tensor shapes before plugging it into the main training loop.
if __name__ == "__main__":
    # Initialize the model
    model = SurrogateMLP()
    
    # Create a dummy batch of 32 random metagrating geometries (scaled 0 to 1)
    # Shape: (Batch Size, Number of Geometric Parameters)
    dummy_inputs = torch.rand(32, 6) 
    
    # Run the forward pass
    predicted_spectra = model(dummy_inputs)
    
    print(f"Input shape: {dummy_inputs.shape}")
    print(f"Output shape: {predicted_spectra.shape}")
    print("If Output shape is torch.Size([32, 22]), your architecture is set up perfectly!")