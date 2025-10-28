# software/services/pinn_forward/freezing_soil_pinn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os


# -------------------------
# Utility: Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Neural Network Definition
# -------------------------
class PhaseChangePINN(nn.Module):
    """Feedforward neural network for temperature field T(x,t)."""

    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Physics-Informed Model
# -------------------------
class FreezingSoilPINN:
    """Physics-Informed Neural Network for freezing soil heat transfer."""

    def __init__(self, params, device="cpu", log_callback=None):
        self.device = torch.device(device)
        self.params = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in params.items()}
        self.log = log_callback or (lambda msg: print(msg))

        # Build model and optimizer
        self.model = PhaseChangePINN([2, 50, 50, 50, 1]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # Logging buffers
        self.loss_history = []
        self.components = {"pde": [], "bc0": [], "bcL": [], "ic": []}
        self.T_max = None

    # -------------------------
    # Physical Subroutines
    # -------------------------
    def _pore_water(self, T):
        T_nabla, b = self.params["T_nabla"], self.params["b"]
        return torch.where(T >= T_nabla, torch.ones_like(T), torch.abs(T_nabla) ** b * torch.abs(T) ** (-b))

    def _unfrozen_water(self, T):
        return self.params["eta"] * self._pore_water(T)

    def _effective_heat_capacity(self, T):
        phi = self._pore_water(T)
        return (
            (1 - self.params["eta"]) * self.params["C_f"]
            + self.params["eta"] * (phi * self.params["C_l"] + (1 - phi) * self.params["C_i"])
        )

    def _effective_conductivity(self, T):
        phi = self._pore_water(T)
        return (
            (1 - self.params["eta"]) * self.params["lambda_f"]
            + self.params["eta"] * (phi * self.params["lambda_l"] + (1 - phi) * self.params["lambda_i"])
        )

    def _boundary_temp(self, t):
        return 4.03 + 16.11 * torch.sin(2 * np.pi * t / 365 - 1.709)

    def _initial_temp(self, x):
        z1, z2 = 0.0, 5.0
        T1 = self._boundary_temp(torch.tensor(0.0, device=self.device)).item()
        T2 = 1.0
        return T1 + (T2 - T1) / (z2 - z1) * x

    # -------------------------
    # PDE Loss Computation
    # -------------------------
    def compute_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        T_pred = self.model(torch.cat([x, t], dim=1))

        dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred),
                                    create_graph=True, retain_graph=True)[0]
        dT_dx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred),
                                    create_graph=True, retain_graph=True)[0]
        d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx),
                                      create_graph=True, retain_graph=True)[0]

        C_eff = self._effective_heat_capacity(T_pred)
        lambda_eff = self._effective_conductivity(T_pred)
        theta = self._unfrozen_water(T_pred)
        dtheta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta),
                                        create_graph=True, retain_graph=True)[0]

        # PDE residual
        pde_res = C_eff * dT_dt - lambda_eff * d2T_dx2 - self.params["L"] * dtheta_dt
        pde_loss = torch.mean(pde_res ** 2)

        # Boundary and initial conditions
        bc0_mask = (x == 0).squeeze()
        bcL_mask = (x == 5).squeeze()
        ic_mask = (t == 0).squeeze()

        bc0_pred = T_pred[bc0_mask]
        bcL_pred = T_pred[bcL_mask]
        ic_pred = T_pred[ic_mask]

        bc0_true = self._boundary_temp(t[bc0_mask]) / self.T_max
        bcL_true = torch.ones_like(bcL_pred) / self.T_max
        ic_true = self._initial_temp(x[ic_mask]).unsqueeze(1) / self.T_max

        bc0_loss = torch.mean((bc0_pred - bc0_true) ** 2)
        bcL_loss = torch.mean((bcL_pred - bcL_true) ** 2)
        ic_loss = torch.mean((ic_pred - ic_true) ** 2)

        total_loss = 1e2 * (pde_loss + bc0_loss + bcL_loss) + 1 * ic_loss
        return total_loss, pde_loss, bc0_loss, bcL_loss, ic_loss

    # -------------------------
    # Training Routine
    # -------------------------
    def train(self, x_domain, t_domain, epochs=50000, n_samples=5000):
        """Train the PINN model on domain [x_domain] × [t_domain]."""
        set_seed(42)

        # Determine normalization factor
        t_test = torch.linspace(t_domain[0], t_domain[1], 1000, device=self.device)
        self.T_max = torch.max(self._boundary_temp(t_test)).item()
        self.log(f"T_max scaling factor = {self.T_max:.2f}")

        # Sample points
        x = torch.rand(n_samples, 1, device=self.device) * (x_domain[1] - x_domain[0]) + x_domain[0]
        t = torch.rand(n_samples, 1, device=self.device) * (t_domain[1] - t_domain[0]) + t_domain[0]

        # Add boundaries
        bc0_points = torch.zeros(500, 1, device=self.device)
        bc0_times = torch.rand(500, 1, device=self.device) * t_domain[1]
        bcL_points = torch.ones(500, 1, device=self.device) * 5.0
        bcL_times = torch.rand(500, 1, device=self.device) * t_domain[1]
        ic_points = torch.rand(1000, 1, device=self.device) * 5.0
        ic_times = torch.zeros(1000, 1, device=self.device)

        x = torch.cat([x, bc0_points, bcL_points, ic_points])
        t = torch.cat([t, bc0_times, bcL_times, ic_times])

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss, pde_loss, bc0_loss, bcL_loss, ic_loss = self.compute_loss(x, t)
            weighted_loss = 1e2 * (pde_loss + bc0_loss + bcL_loss) + 1e3 * ic_loss
            weighted_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 1000 == 0:
                self.log(f"[{epoch:05d}] Loss={total_loss.item():.3e} "
                         f"PDE={pde_loss.item():.3e} BC0={bc0_loss.item():.3e} "
                         f"BCL={bcL_loss.item():.3e} IC={ic_loss.item():.3e}")
            self.loss_history.append(total_loss.item())

        self.log("Training completed!")

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, x, t):
        with torch.no_grad():
            xt = torch.cat([x, t], dim=1)
            return self.model(xt)

    def save(self, model_dir="models/pinn_forward"):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "freezing_soil_pinn.pt")
        torch.save(self.model.state_dict(), path)
        self.log(f"Model saved to {path}")
