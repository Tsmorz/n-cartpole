"""Per-episode randomization of the *true* plant, for sim-to-real robustness.

The controller only ever sees the nominal model; a real rig differs from it by
measurement error, manufacturing tolerance, and wear. Sampling the plant that the
dynamics integrate each episode (while rewards, shaping and replay relabeling keep
using the nominal parameters) forces the policy to tolerate that gap without
needing any parameter inputs — so the deployed actor stays a plain kinematics →
force network.

Every perturbation is symmetric under left-right mirroring, so the symmetric data
augmentation used in training stays valid.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from n_cartpole.env.dynamics import PhysicsParams


@dataclass
class PlantRandomization:
    """Ranges for the per-episode plant perturbation.

    ``*_rel`` fields are symmetric relative ranges (``0.1`` → uniform in
    ``[0.9, 1.1]`` x nominal); ``*_scale`` fields are multiplicative ``(lo, hi)``
    ranges sampled log-uniformly (friction is only known to within a factor).
    Each link is perturbed independently.
    """

    cart_mass_rel: float = 0.10
    cart_friction_scale: tuple[float, float] = (0.5, 2.0)
    link_mass_rel: float = 0.10
    link_length_rel: float = 0.02
    link_com_rel: float = 0.05
    link_inertia_rel: float = 0.15
    joint_friction_scale: tuple[float, float] = (0.5, 2.0)
    # Actuator gain error: the delivered force is ``gain * command``.
    force_gain_rel: float = 0.10

    def sample(
        self, nominal: PhysicsParams, n: int, rng: np.random.Generator
    ) -> tuple[PhysicsParams, float]:
        """Draw a perturbed plant; return it with the actuator gain in use.

        The returned ``PhysicsParams`` has its ``force_max`` scaled by the gain
        (a drive with a gain error also saturates at ``gain * force_max``); the
        caller multiplies the commanded force by the returned gain.
        """

        def rel(span: float, size: int | None = None):
            return rng.uniform(1.0 - span, 1.0 + span, size)

        def scale(lo_hi: tuple[float, float], size: int | None = None):
            lo, hi = lo_hi
            return np.exp(rng.uniform(np.log(lo), np.log(hi), size))

        lengths = nominal.link_lengths(n) * rel(self.link_length_rel, n)
        gain = float(rel(self.force_gain_rel))
        plant = dataclasses.replace(
            nominal,
            M=nominal.M * float(rel(self.cart_mass_rel)),
            b=nominal.b * float(scale(self.cart_friction_scale)),
            force_max=nominal.force_max * gain,
            masses=tuple(nominal.link_masses(n) * rel(self.link_mass_rel, n)),
            lengths=tuple(lengths),
            com=tuple(
                np.minimum(
                    nominal.link_coms(n)
                    * rel(self.link_com_rel, n)
                    * lengths
                    / nominal.link_lengths(n),
                    lengths,  # the centre of mass stays within the link
                )
            ),
            inertia=tuple(nominal.link_inertias(n) * rel(self.link_inertia_rel, n)),
            joint_friction=tuple(
                nominal.joint_frictions(n) * scale(self.joint_friction_scale, n)
            ),
        )
        return plant, gain
