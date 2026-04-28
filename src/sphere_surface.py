import numpy as np


class SphereSurface:
    """Analytic 3/8-sphere surface (polar angle 0..135 deg from top)."""

    def __init__(self, center, radius, max_polar_angle=3 * np.pi / 4):
        self.c = np.array(center, dtype=float)
        self.R = float(radius)
        self.max_polar = float(max_polar_angle)  # 135 deg = 3*pi/4

    # ------------------------------------------------------------------
    def normal(self, p):
        """Outward unit normal at p (pointing away from sphere center)."""
        v = np.asarray(p, dtype=float) - self.c
        return v / np.linalg.norm(v)

    def project(self, p):
        """Project p onto the sphere surface."""
        v = np.asarray(p, dtype=float) - self.c
        return self.c + self.R * v / np.linalg.norm(v)

    def signed_dist(self, p, r_tool=0.0):
        """Signed distance of tool center from contact surface (>0 = above)."""
        return np.linalg.norm(np.asarray(p) - self.c) - (self.R + r_tool)

    def tangent_frame(self, p):
        """Orthonormal frame (t1, t2, n) at p; n is the outward normal."""
        n = self.normal(p)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(ref, n)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return t1, t2, n

    def polar_angle(self, p):
        """Polar angle (radians) of p measured from the top of the sphere."""
        n = self.normal(p)
        return np.arccos(np.clip(np.dot(n, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))

    def in_region(self, p):
        """True if p projects into the 3/8-sphere polishing region."""
        return self.polar_angle(p) <= self.max_polar

    def attractor_point(self):
        """Contact-surface point at the top of the sphere (circle center)."""
        return self.c + np.array([0.0, 0.0, self.R])
