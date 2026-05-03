"""
Polishing DS for cylindrical tool — point-contact (tool_center_site on sphere surface).

Key geometry:
  The tool face is a flat disk of radius r_cyl.  When the disk center sits on the
  sphere (d_center = 0), the disk rim is ABOVE the sphere surface by:
      h = R - sqrt(R² - r_cyl²)  ≈ r_cyl²/(2R)

  Physical contact (rim touches sphere) first occurs when:
      d_center = R - sqrt(R² - r_cyl²)  ≡  r_contact_offset  ≈ 1.2 mm

  So the DS uses an effective sphere radius:
      R_eff = sqrt(R² - r_cyl²)

  d = ||p - c|| - R_eff == 0   ↔   tool rim just touching sphere.
  sigma → 1 at true first contact, not 1.2 mm too late.
"""

import numpy as np


class PolishingDSTool:
    def __init__(
        self,
        sphere,
        r_cyl=0.0155,       # cylinder face radius [m] — from tool_collision size[0]
        v_target=0.05,
        omega=np.pi / 3,
        r_circle=0.03,
        k_limit=6.0,
        d_blend=0.03,
        sigma_close=0.98,
    ):
        self.sphere = sphere
        self.v_target = v_target
        self.omega = omega
        self.r_circle = r_circle
        self.k_limit = k_limit
        self.d_blend = d_blend
        self.sigma_close = sigma_close   # exposed so controller can read it

        # Effective sphere radius for the face center of a flat-ended cylinder.
        # R_eff < R; the difference (≈1.2 mm) is the contact offset.
        R = sphere.R
        self.R_eff = float(np.sqrt(max(R * R - r_cyl * r_cyl, 0.0)))
        self.r_tool = 0.0               # kept for API compatibility

        # Attractor: position of tool face center when rim touches sphere top.
        # z_att = sphere.c[2] + R_eff  (not R)
        self.p_att = sphere.c.copy()
        self.p_att[2] += self.R_eff

    def compute(self, ee_pos):
        """
        Args:
            ee_pos: world position of tool_center_site (3,)

        Returns:
            v_d   : desired EE velocity (3,) [m/s]
            n     : outward surface normal at ee_pos (3,)
            sigma : blend weight — 0 = far/reaching, 1 = rim touching sphere
        """
        ee_pos = np.asarray(ee_pos, dtype=float)

        # Signed distance from tool face center to effective sphere surface.
        v_from_c = ee_pos - self.sphere.c
        dist = np.linalg.norm(v_from_c)
        d = dist - self.R_eff             # 0 when rim first touches sphere
        n = v_from_c / dist               # outward unit normal

        # Tangent frame (t1 used as orbit kick at attractor)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(ref, n)
        t1 /= np.linalg.norm(t1)

        # ── Reaching: approach along -n ───────────────────────────────────
        v_reach = -self.v_target * n

        # ── Circular limit-cycle in the tangent plane ─────────────────────
        e = ee_pos - self.p_att
        e_tan = e - np.dot(e, n) * n
        e_norm = np.linalg.norm(e_tan)

        if e_norm > 1e-4:
            e_unit = e_tan / e_norm
            perp = np.cross(n, e_unit)       # CCW tangential direction
            v_circ = (
                -self.k_limit * (e_norm - self.r_circle) * e_unit
                + self.omega * self.r_circle * perp
            )
        else:
            v_circ = self.omega * self.r_circle * t1

        # ── Smoothstep sigma: 0 = far, 1 = rim touching sphere ────────────
        z = np.clip(1.0 - d / self.d_blend, 0.0, 1.0)
        sigma = z * z * (3.0 - 2.0 * z)

        # ── Velocity blend ────────────────────────────────────────────────
        if sigma < self.sigma_close:
            # Approaching: reduce normal component as sigma grows.
            normal_scale = 0.2 + 0.8 * (1.0 - sigma)
            v_d = normal_scale * v_reach + sigma * v_circ
        else:
            # Near / at contact: hand normal direction to controller entirely.
            v_d = sigma * v_circ

        return v_d, n, sigma
