import torch
import torch.nn as nn
from torch.autograd import Function


class NestedDropoutFunction(Function):
    @staticmethod
    def forward(ctx, input, noise):
        # Apply noise directly to input
        output = input * noise

        # Save noise for backward pass
        ctx.save_for_backward(noise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        noise, = ctx.saved_tensors
        # Apply noise to the gradient during backpropagation
        grad_input = grad_output * noise
        return grad_input, None  # Return None for the second argument because noise doesn't require gradients


class NestedDropoutLayer(nn.Module):
    def __init__(self, k_dim, p=0.02):
        super(NestedDropoutLayer, self).__init__()
        self.k_dim = k_dim
        self.noise = None  # Initialize noise attribute for later access
        self.p = p
        print(self.p)
        self.n_converged = 0

    def forward(self, input):
        # Generate noise during the forward pass
        noise = torch.ones_like(input)
        for n in range(noise.size(0)):
            # Randomly select a cutoff index, keeping only the first b elements
            b = torch.distributions.Geometric(probs=self.p).sample().item()
            b = int(self.n_converged + b + 1)
            noise[n, b:] = 0  # Zero out the elements after the cutoff point

        # Store noise for future access
        self.noise = noise
        # input = torch.cat((input[:, :self.n_converged].detach(), input[:, self.n_converged:]), dim=1)

        # Use the custom function to apply the noise and return the result
        return NestedDropoutFunction.apply(input, noise)

    def get_noise(self):
        """Access the noise tensor after the forward pass."""
        return self.noise.bool().detach().clone()

    def get_n_converged(self):
        """
        Get the number of converged units.

        Returns:
        --------
        int
            The number of units that have converged and are no longer updated.

        Expected Output Shape:
        ----------------------
        Returns an integer value.
        """
        return self.n_converged

    def set_n_converged(self, idx):
        """
        Set the number of converged units, which will not be dropped in subsequent passes.

        Parameters:
        -----------
        idx : int
            The index of the first unit to be excluded from dropout (i.e., the
            number of units that have converged).

        Expected Input Shape:
        ---------------------
        No tensor input required. Sets an integer value.
        """
        self.n_converged = idx


class NestedDropoutUniformLayer(nn.Module):
    def __init__(self, k_dim):
        super(NestedDropoutUniformLayer, self).__init__()
        self.k_dim = k_dim
        self.noise = None  # Initialize noise attribute for later access
        self.min_fb = 1
    def forward(self, input):
        # Generate noise during the forward pass
        noise = torch.ones_like(input)
        for n in range(noise.size(0)):
            # Randomly select a cutoff index, keeping only the first b elements
            b = torch.randint(self.min_fb, self.k_dim + 1, (1,)).item()
            noise[n, b:] = 0  # Zero out the elements after the cutoff point

        # Store noise for future access
        self.noise = noise

        # Use the custom function to apply the noise and return the result
        return NestedDropoutFunction.apply(input, noise)

    def get_noise(self):
        """Access the noise tensor after the forward pass."""
        return self.noise.bool().detach().clone()

# # Example usage
# x = torch.randn(5, 80)
# enc = nn.Linear(80, 80)
# nd = NestedDropoutLayer(20)
# dec = nn.Linear(80, 80)
#
# x = enc(x)
# x = x.view(5, 20, 4)
# x = nd(x)
# x = x.view(5, -1)
# y = dec(x)
#
# loss = nn.MSELoss()(y, x)
# loss.backward()
