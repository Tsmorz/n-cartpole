"""Extended Kalman filter for the cart + n-link pendulum state, from encoder readings.

A real rig measures cart position and joint angles and has to *derive* the four to
eight velocities. Differencing successive readings amplifies encoder noise and
quantization by ``1 / dt``; low-pass filtering the difference trades that noise for
lag. A model-based observer avoids the trade: it *predicts* the velocities from the
equations of motion and the force it knows it commanded, and uses the encoder
readings only to correct the prediction, so the estimate is smooth *and* has almost no
lag. This is the estimator a firmware would run, so the simulator can put it in the
policy's observation path and the policy trains against exactly its errors.

State ``[x, x_dot, theta_0, theta_0_dot, ...]`` (the same layout as the physics
state, angles unwrapped). Prediction integrates ``dynamics._ode`` with a midpoint
step and a numerical Jacobian; the update uses the position readings only.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.env.dynamics import PhysicsParams, _model, _ode


class StateEKF:
    """EKF over the physics state, driven by the applied force and position readings."""

    def __init__(
        self,
        physics: PhysicsParams,
        n_links: int,
        pos_std: np.ndarray,
        force_std: float = 0.5,
        accel_std: float = 5.0,
    ) -> None:
        """Build the filter.

        Args:
            physics: the NOMINAL model (the filter does not know the true plant).
            n_links: number of pendulum links.
            pos_std: ``(n_links + 1,)`` std of each position reading
                ``[x, theta_0, ...]`` including quantization (``res / sqrt(12)``).
            force_std: unmodelled force on the cart (N): rail friction, force-gain
                error. Larger trusts the readings more over the model.
            accel_std: unmodelled angular acceleration of each link (rad/s^2).

        """
        self.n = n_links
        self.dim = 2 + 2 * n_links
        self.dt = physics.dt
        self.model = _model(physics, n_links)
        self.R = np.diag(np.asarray(pos_std, dtype=float) ** 2)
        # Continuous process-noise densities on the three/five acceleration states.
        sig = np.concatenate(([force_std / physics.M], np.full(n_links, accel_std)))
        self._q_vel = (sig**2) * self.dt  # discrete velocity-state variance per step
        self.H = np.zeros((n_links + 1, self.dim))
        self.H[0, 0] = 1.0
        for k in range(n_links):
            self.H[k + 1, 2 + 2 * k] = 1.0
        self._vel_idx = np.array([1] + [3 + 2 * k for k in range(n_links)])
        self.x = np.zeros(self.dim)
        self.P = np.eye(self.dim)

    def reset(self, pos: np.ndarray) -> np.ndarray:
        """Start from the first position reading with unknown (zero) velocities."""
        self.x = np.zeros(self.dim)
        self.x[0] = pos[0]
        self.x[2::2] = pos[1:]
        var = np.diag(self.R)
        self.P = np.diag(
            [var[0], 1.0] + [v for k in range(self.n) for v in (var[k + 1], 1.0)]
        )
        return self.x.copy()

    def _f(self, s: np.ndarray, u: float) -> np.ndarray:
        return _ode(0.0, s, u, self.model)

    def update(self, applied_force: float, pos: np.ndarray) -> np.ndarray:
        """Predict one step under ``applied_force``, correct with ``pos``; return the state."""
        dt, u, s = self.dt, applied_force, self.x
        # Midpoint prediction.
        k1 = self._f(s, u)
        s_pred = s + dt * self._f(s + 0.5 * dt * k1, u)
        # Numerical Jacobian of the continuous dynamics at the current estimate.
        A = np.empty((self.dim, self.dim))
        eps = 1e-6
        for i in range(self.dim):
            d = np.zeros(self.dim)
            d[i] = eps
            A[:, i] = (self._f(s + d, u) - self._f(s - d, u)) / (2 * eps)
        F = np.eye(self.dim) + dt * A + 0.5 * dt**2 * (A @ A)
        Q = np.zeros((self.dim, self.dim))
        Q[self._vel_idx, self._vel_idx] = self._q_vel
        # Positions inherit the velocity uncertainty over the step.
        P = F @ self.P @ F.T + Q
        # Position-reading update.
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        y = pos - self.H @ s_pred
        self.x = s_pred + K @ y
        I_KH = np.eye(self.dim) - K @ self.H
        self.P = I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form
        return self.x.copy()
