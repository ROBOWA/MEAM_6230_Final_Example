"""
Polishing Dynamical System.

Generates desired EE velocity by blending:
  - Reaching:  drives tool toward the sphere surface along the normal
  - Circular:  limit-cycle in the tangent plane around the attractor (top of sphere)

Blend weight sigma transitions from 0 (far above surface) to 1 (at/below contact)
via a smoothstep over [0, d_blend]:
  d >= d_blend  →  sigma = 0  (pure reaching)
  0 <= d < d_blend  →  smooth cubic transition
  d <= 0          →  sigma = 1  (full polishing)
"""

import numpy as np

class PolishingDS:
    sigma_close = 0.95
    def __init__(
        self,
        sphere,
        r_tool=0.009,
        v_target=0.05,
        omega=np.pi,
        r_circle=0.05,
        k_limit=2.0,
        d_blend=0.015,
    ):
        """
        Args:
            sphere:    SphereSurface instance
            r_tool:    tool sphere radius [m]
            v_target:  reaching speed toward surface [m/s]
            omega:     circular angular frequency [rad/s]
            r_circle:  desired polishing circle radius [m]
            k_limit:   radial attraction gain toward circle [1/s]
            d_blend:   distance scale for reaching↔circular blend [m]
        """
        self.sphere = sphere
        self.r_tool = r_tool
        self.v_target = v_target
        self.omega = omega
        self.r_circle = r_circle
        self.k_limit = k_limit
        self.d_blend = d_blend

        # Attractor: top of the offset sphere (tool center rides at R + r_tool)
        self.p_att = sphere.attractor_point(self.r_tool)

    def compute(self, ee_pos):
        """
        Returns:
            v_d   : desired EE linear velocity (3,) [m/s]
            n     : outward surface normal at ee_pos (3,)
            sigma : contact blend weight in [0,1]
        """
        ee_pos = np.asarray(ee_pos, dtype=float)

        d = self.sphere.signed_dist(ee_pos, self.r_tool)
        t1, t2, n = self.sphere.tangent_frame(ee_pos)

        # --- Reaching velocity: move toward the sphere along -n ---
        v_reach = -self.v_target * n

        # --- Circular limit-cycle in the tangent plane ---
        # e_tan: vector from attractor to ee, projected onto tangent plane
        e = ee_pos - self.p_att
        e_tan = e - np.dot(e, n) * n
        e_norm = np.linalg.norm(e_tan)

        if e_norm > 1e-4:
            e_unit = e_tan / e_norm
            perp = np.cross(n, e_unit)  # tangential direction for CCW rotation
            # Radial: attract toward radius r_circle; Tangential: constant angular speed
            v_circ = (
                -self.k_limit * (e_norm - self.r_circle) * e_unit
                + self.omega * self.r_circle * perp
            )
        else:
            # At the attractor center, kick outward along t1 to start the orbit
            v_circ = self.omega * self.r_circle * t1

        # --- Blend: sigma=0 far above surface, sigma=1 at/below contact (d=0) ---
        # Smoothstep over [0, d_blend]: full polishing at contact, full reaching far above.
        z = np.clip(1.0 - d / self.d_blend, 0.0, 1.0)
        sigma = z * z * (3.0 - 2.0 * z)
        
        if sigma < self.sigma_close:
            v_min_ratio = 0.2
            normal_scale = v_min_ratio + (1.0 - v_min_ratio) * (1.0 - sigma)

            v_normal = normal_scale * (-self.v_target * n)
            v_d = v_normal + sigma * v_circ
        else:  
            # v_d = np.max((1.0 - sigma), 0) * v_reach + sigma * v_circ
            v_d = (1.0 - sigma) * v_reach + sigma * v_circ
        return v_d, n, sigma
