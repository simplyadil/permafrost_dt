# software/services/pinn_inversion/inversion_pinn_model.py
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
from scipy.interpolate import interp1d


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PhaseChangePINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class InversionFreezingSoilPINN:
    """
    PINN-based inversion of soil thermal parameters using observed temperatures.
    """

    def __init__(self, known_params, param_bounds, device="cpu", log_callback=None):
        self.device = torch.device(device)
        self.log = log_callback or print

        # Fixed known parameters
        self.params = {k: torch.tensor(v, device=self.device) for k, v in known_params.items()}
        self.param_bounds = param_bounds

        # Trainable physical parameters
        self.param_names = list(param_bounds.keys())
        self.lambda_f = nn.Parameter(torch.tensor(np.mean(param_bounds["lambda_f"]), device=self.device))
        self.C_f = nn.Parameter(torch.tensor(np.mean(param_bounds["C_f"]), device=self.device))
        self.eta = nn.Parameter(torch.tensor(np.mean(param_bounds["eta"]), device=self.device))
        self.b = nn.Parameter(torch.tensor(np.mean(param_bounds["b"]), device=self.device))
        self.T_nabla = nn.Parameter(torch.tensor(np.mean(param_bounds["T_nabla"]), device=self.device))

        self.model = PhaseChangePINN([2, 50, 50, 50, 1]).to(self.device)
        self.optimizer = optim.Adam(
            [{"params": self.model.parameters(), "lr": 1e-4},
             {"params": [self.lambda_f, self.C_f, self.eta, self.b, self.T_nabla], "lr": 1e-5}]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        self.loss_history: List[float] = []
        self.loss_components: Dict[str, List[float]] = {
            "physics": [],
            "boundary": [],
            "data": [],
        }
        self.param_history = {name: [] for name in self.param_names}
        self.boundary_data = None
        self.T_max = 20.0
        self.boundary_interp_func = None
        self._last_summary: Dict[str, object] | None = None

    # ---- Physics model helpers ----
    def _pore_water(self, T):
        return torch.where(
            T >= self.T_nabla, torch.ones_like(T),
            torch.abs(self.T_nabla)**self.b * torch.abs(T)**(-self.b)
        )

    def _unfrozen_water(self, T):
        return self.eta * self._pore_water(T)

    def _effective_heat_capacity(self, T):
        phi = self._pore_water(T)
        return (1 - self.eta) * self.C_f + self.eta * (phi * self.params["C_l"] + (1 - phi) * self.params["C_i"])

    def _effective_conductivity(self, T):
        phi = self._pore_water(T)
        return (1 - self.eta) * self.lambda_f + self.eta * (phi * self.params["lambda_l"] + (1 - phi) * self.params["lambda_i"])

    def _create_boundary_interp_func(self):
        if self.boundary_data is not None:
            time_points = self.boundary_data["time_days"].values
            temp_values = self.boundary_data["temperature_0m"].values
            self.boundary_interp_func = interp1d(time_points, temp_values, kind="linear",
                                                 bounds_error=False, fill_value="extrapolate")

    def _boundary_temp(self, t):
        if isinstance(t, torch.Tensor):
            t_np = t.detach().cpu().numpy().flatten()
            if self.boundary_interp_func is not None:
                temp_values = self.boundary_interp_func(t_np)
                return torch.tensor(temp_values / self.T_max, dtype=torch.float32, device=self.device).unsqueeze(1)
            else:
                temp_values = 4.03 + 16.11 * np.sin(2 * np.pi * t_np / 365 - 1.709)
                return torch.tensor(temp_values / self.T_max, dtype=torch.float32, device=self.device).unsqueeze(1)
        return torch.tensor([[4.03 + 16.11 * np.sin(2 * np.pi * t / 365 - 1.709)]], device=self.device) / self.T_max

    # ---- Loss components ----
    def _physics_loss(self, x, t):
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
        pde_res = C_eff * dT_dt - lambda_eff * d2T_dx2 - self.params["L"] * dtheta_dt
        return torch.mean(pde_res**2)

    def _data_loss(self, x_obs, t_obs, T_obs_true):
        T_pred = self.model(torch.cat([x_obs, t_obs], dim=1))
        return torch.mean((T_pred - T_obs_true)**2)

    def _boundary_loss(self, x, t):
        T_pred = self.model(torch.cat([x, t], dim=1))
        bc0_mask = (x == 0).squeeze()
        bcL_mask = (x == 5).squeeze()
        loss = 0.0
        if bc0_mask.any():
            bc0_pred = T_pred[bc0_mask]
            bc0_true = self._boundary_temp(t[bc0_mask])
            loss += torch.mean((bc0_pred - bc0_true)**2)
        if bcL_mask.any():
            bcL_pred = T_pred[bcL_mask]
            bcL_true = torch.ones_like(bcL_pred) / self.T_max
            loss += torch.mean((bcL_pred - bcL_true)**2)
        return loss

    def _apply_constraints(self):
        with torch.no_grad():
            for name, (lo, hi) in self.param_bounds.items():
                param = getattr(self, name)
                param.data = torch.clamp(param.data, lo, hi)

    def get_params(self):
        return {n: getattr(self, n).item() for n in self.param_names}

    def _record_epoch_metrics(self, total_loss, physics_loss, boundary_loss, data_loss) -> None:
        """Collect losses and parameter values for downstream visualisation."""
        self.loss_history.append(float(total_loss.item()))
        self.loss_components["physics"].append(float(physics_loss.item()))
        self.loss_components["boundary"].append(float(boundary_loss.item()))
        self.loss_components["data"].append(float(data_loss.item()))
        for name in self.param_names:
            value = getattr(self, name)
            self.param_history[name].append(float(value.detach().cpu().item()))

    @staticmethod
    def _downsample(seq: Sequence[float], max_points: int = 1000) -> List[float]:
        if len(seq) <= max_points:
            return list(seq)
        if max_points <= 1:
            return [float(seq[-1])]
        step = max((len(seq) - 1) / (max_points - 1), 1.0)
        indices = [min(int(round(i * step)), len(seq) - 1) for i in range(max_points)]
        return [float(seq[i]) for i in dict.fromkeys(indices)]

    @staticmethod
    def _select_indices(length: int, max_points: int) -> np.ndarray:
        if length <= max_points:
            return np.arange(length, dtype=int)
        return np.unique(np.linspace(0, length - 1, max_points, dtype=int))

    def build_training_summary(
        self,
        *,
        status: str,
        epochs: int,
        validation: Dict[str, object] | None = None,
        max_points: int = 1000,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "status": status,
            "epochs": int(epochs),
            "timestamp": datetime.utcnow().isoformat(),
            "loss": {
                "total": self._downsample(self.loss_history, max_points=max_points),
                "physics": self._downsample(self.loss_components["physics"], max_points=max_points),
                "boundary": self._downsample(self.loss_components["boundary"], max_points=max_points),
                "data": self._downsample(self.loss_components["data"], max_points=max_points),
            },
            "parameters": self.get_params(),
            "param_history": {
                name: self._downsample(history, max_points=max_points)
                for name, history in self.param_history.items()
            },
        }
        if validation is not None:
            summary["validation"] = validation
        return summary

    def save_training_summary(
        self,
        target_path: str | Path,
        *,
        status: str,
        epochs: int,
        validation: Dict[str, object] | None = None,
        summary: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if summary is None:
            summary = self.build_training_summary(status=status, epochs=epochs, validation=validation)
        target.write_text(json.dumps(summary, indent=2))
        self._last_summary = summary
        return summary

    def build_validation_summary(
        self,
        x_obs: torch.Tensor,
        t_obs: torch.Tensor,
        T_obs_true: torch.Tensor,
        *,
        max_points: int = 400,
    ) -> Dict[str, object]:
        with torch.no_grad():
            preds = self.predict(x_obs, t_obs).cpu().numpy().flatten()
        true = (T_obs_true * self.T_max).cpu().numpy().flatten()
        times = t_obs.cpu().numpy().flatten()
        depths = x_obs.cpu().numpy().flatten()

        abs_error = np.abs(preds - true)
        overall = {
            "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
            "max_abs_error": float(abs_error.max()) if abs_error.size else 0.0,
        }

        per_depth = []
        for depth in sorted(np.unique(depths)):
            mask = depths == depth
            depth_times = times[mask]
            depth_true = true[mask]
            depth_pred = preds[mask]
            depth_error = abs_error[mask]

            order = np.argsort(depth_times)
            depth_times = depth_times[order]
            depth_true = depth_true[order]
            depth_pred = depth_pred[order]
            depth_error = depth_error[order]

            indices = self._select_indices(len(depth_times), max_points)
            per_depth.append(
                {
                    "depth_m": float(depth),
                    "time_days": depth_times[indices].tolist(),
                    "true": depth_true[indices].tolist(),
                    "pred": depth_pred[indices].tolist(),
                    "abs_error": depth_error[indices].tolist(),
                    "stats": {
                        "mean_abs_error": float(depth_error.mean()) if depth_error.size else 0.0,
                        "max_abs_error": float(depth_error.max()) if depth_error.size else 0.0,
                    },
                }
            )

        return {"overall": overall, "per_depth": per_depth}

    # ---- Training ----
    def train(self, x_obs, t_obs, T_obs_true, x_domain, t_domain, epochs=20000):
        set_seed(42)
        self.log("Starting inversion training...")
        if self.boundary_data is not None:
            self._create_boundary_interp_func()
            self.T_max = self.boundary_data[[c for c in self.boundary_data.columns if "temperature" in c]].max().max()
        else:
            self.T_max = 20.0

        self.loss_history.clear()
        for key in self.loss_components:
            self.loss_components[key].clear()
        for name in self.param_history:
            self.param_history[name].clear()

        n_samples = 5000
        x = torch.rand(n_samples, 1, device=self.device) * (x_domain[1] - x_domain[0]) + x_domain[0]
        t = torch.rand(n_samples, 1, device=self.device) * (t_domain[1] - t_domain[0]) + t_domain[0]
        x.requires_grad_(True)
        t.requires_grad_(True)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            physics_loss = self._physics_loss(x, t)
            boundary_loss = self._boundary_loss(x, t)
            data_loss = self._data_loss(x_obs, t_obs, T_obs_true)
            total_loss = 1e2 * (physics_loss + boundary_loss) + 1e5 * data_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self._apply_constraints()

            if epoch % 500 == 0:
                self.log(f"[{epoch:05d}] Loss={total_loss.item():.3e}, Params={self.get_params()}")
            self._record_epoch_metrics(total_loss, physics_loss, boundary_loss, data_loss)

        self.log("Inversion complete!")
        self._last_summary = self.build_training_summary(status="trained", epochs=epochs)
        return self.get_params()

    def save(self, model_path: str) -> None:
        """Persist the inversion model weights and parameter estimates."""
        state = {
            "model": self.model.state_dict(),
            "lambda_f": self.lambda_f.detach().cpu(),
            "C_f": self.C_f.detach().cpu(),
            "eta": self.eta.detach().cpu(),
            "b": self.b.detach().cpu(),
            "T_nabla": self.T_nabla.detach().cpu(),
            "T_max": float(self.T_max),
        }
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        self.log(f"Inversion PINN saved to {path}")

    def load(self, model_path: str) -> None:
        """Load inversion model weights and parameter estimates."""
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.lambda_f.data = state["lambda_f"].to(self.device).clone().detach()
        self.C_f.data = state["C_f"].to(self.device).clone().detach()
        self.eta.data = state["eta"].to(self.device).clone().detach()
        self.b.data = state["b"].to(self.device).clone().detach()
        self.T_nabla.data = state["T_nabla"].to(self.device).clone().detach()
        self.T_max = state.get("T_max", 20.0)
        self.model.eval()
        self.log(f"Inversion PINN weights loaded from {model_path}")
