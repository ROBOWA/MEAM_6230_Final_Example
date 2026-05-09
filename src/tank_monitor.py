"""
Real-time energy-tank monitor for PolishingController.

Usage in the simulation loop
─────────────────────────────
    from src.tank_monitor import TankMonitor

    monitor = TankMonitor(tank_s_max=ctrl.tank_s_max, window_s=10.0)

    while running:
        v_d, n, sigma, contact_state, F_des, F_n = ctrl.step()
        env.step()
        monitor.update(sim_time, ctrl.last_tank_info)   # non-blocking

    monitor.freeze()        # stop live updates, keep the window open
    monitor.save("tank.png")
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


# ── colour palette ────────────────────────────────────────────────────
_C_TANK   = "#2196F3"   # tank level fill (blue)
_C_PD     = "#4CAF50"   # dissipated / charging (green)
_C_PR     = "#FF9800"   # orbit draw (orange)
_C_PN     = "#E91E63"   # force draw (pink/red)
_C_NET    = "#9C27B0"   # net power into tank (purple)
_C_BETA_R = "#FF9800"
_C_BETA_N = "#E91E63"
_C_SMAX   = "#F44336"   # s_max reference line


class TankMonitor:
    """Live matplotlib window for the Amanhoud energy-tank state.

    Parameters
    ----------
    tank_s_max : float
        Maximum tank level (same as controller parameter).
    window_s : float
        Width of the rolling time window shown in the plots [s].
    update_every : int
        Call matplotlib draw only every this many calls to update().
        At 1 kHz simulation, update_every=50 gives ~20 Hz refresh.
    figsize : tuple
        Matplotlib figure size (width, height) in inches.
    """

    def __init__(
        self,
        tank_s_max: float = 60.0,
        window_s: float = 10.0,
        update_every: int = 50,
        figsize: tuple = (13, 8),
    ):
        self.tank_s_max = tank_s_max
        self.window_s = window_s
        self.update_every = update_every
        self._call_count = 0
        self._frozen = False

        # Rolling data buffers
        maxlen = 100_000
        self._t       = deque(maxlen=maxlen)
        self._s       = deque(maxlen=maxlen)
        self._pd      = deque(maxlen=maxlen)
        self._pr      = deque(maxlen=maxlen)
        self._pn      = deque(maxlen=maxlen)
        self._net     = deque(maxlen=maxlen)
        self._beta_r  = deque(maxlen=maxlen)
        self._beta_n  = deque(maxlen=maxlen)
        self._alpha   = deque(maxlen=maxlen)

        # Build figure
        plt.ion()
        self._fig = plt.figure(figsize=figsize, num="Energy Tank Monitor")
        self._fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(
            3, 2,
            figure=self._fig,
            left=0.08, right=0.97,
            top=0.92, bottom=0.08,
            hspace=0.45, wspace=0.35,
        )

        _ax_kw = dict(facecolor="#16213e")

        # ── Row 0: tank level (wide) ──────────────────────────────────
        self._ax_tank = self._fig.add_subplot(gs[0, :], **_ax_kw)
        # ── Row 1 left: power flows ───────────────────────────────────
        self._ax_pwr  = self._fig.add_subplot(gs[1, 0], **_ax_kw)
        # ── Row 1 right: net ṡ  ────────────────────────────────────────
        self._ax_net  = self._fig.add_subplot(gs[1, 1], **_ax_kw)
        # ── Row 2 left: beta gates ────────────────────────────────────
        self._ax_beta = self._fig.add_subplot(gs[2, 0], **_ax_kw)
        # ── Row 2 right: alpha ────────────────────────────────────────
        self._ax_alph = self._fig.add_subplot(gs[2, 1], **_ax_kw)

        for ax in (self._ax_tank, self._ax_pwr, self._ax_net,
                   self._ax_beta, self._ax_alph):
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.grid(True, color="#2a2a4a", linewidth=0.5)

        # Create line / fill objects (initialized empty)
        t_dummy = np.array([0.0])
        z_dummy = np.array([0.0])

        # Tank level
        self._fill_tank = self._ax_tank.fill_between(
            t_dummy, z_dummy, color=_C_TANK, alpha=0.4, label="tank s")
        self._line_tank, = self._ax_tank.plot(
            [], [], color=_C_TANK, lw=1.5)
        self._line_smax = self._ax_tank.axhline(
            tank_s_max, color=_C_SMAX, ls="--", lw=1.2, label=f"s_max={tank_s_max:.0f}")
        self._line_szero = self._ax_tank.axhline(
            0.0, color="white", ls=":", lw=0.8)
        self._ax_tank.set_ylim(-tank_s_max * 0.05, tank_s_max * 1.15)
        self._ax_tank.set_ylabel("Tank energy  s  [J]", fontsize=9)
        self._ax_tank.set_title("Energy Tank Level", fontsize=10, fontweight="bold")
        self._ax_tank.legend(fontsize=8, loc="upper right",
                             facecolor="#1a1a2e", labelcolor="white")

        # Power flows
        self._line_pd, = self._ax_pwr.plot([], [], color=_C_PD, lw=1.2,
                                            label="p_d (dissipated → charges)")
        self._line_pr, = self._ax_pwr.plot([], [], color=_C_PR, lw=1.2,
                                            label="β_r·p_r (orbit draw)")
        self._line_pn, = self._ax_pwr.plot([], [], color=_C_PN, lw=1.2,
                                            label="β_n·p_n (force draw)")
        self._ax_pwr.axhline(0, color="white", ls=":", lw=0.6)
        self._ax_pwr.set_ylabel("Power  [W]", fontsize=9)
        self._ax_pwr.set_title("Power Flows", fontsize=10, fontweight="bold")
        self._ax_pwr.legend(fontsize=7, loc="upper right",
                             facecolor="#1a1a2e", labelcolor="white")

        # Net ṡ
        self._line_net, = self._ax_net.plot([], [], color=_C_NET, lw=1.2,
                                             label="ṡ = α·p_d − β_r·p_r − β_n·p_n")
        self._fill_net_pos = self._ax_net.fill_between(
            t_dummy, z_dummy, color=_C_PD, alpha=0.25)
        self._fill_net_neg = self._ax_net.fill_between(
            t_dummy, z_dummy, color=_C_PR, alpha=0.25)
        self._ax_net.axhline(0, color="white", ls=":", lw=0.6)
        self._ax_net.set_ylabel("ṡ  [J/s]", fontsize=9)
        self._ax_net.set_title("Net Tank Rate  ṡ", fontsize=10, fontweight="bold")
        self._ax_net.legend(fontsize=7, loc="upper right",
                             facecolor="#1a1a2e", labelcolor="white")

        # Beta gates
        self._line_br, = self._ax_beta.plot([], [], color=_C_BETA_R, lw=1.5,
                                             drawstyle="steps-post",
                                             label="β_r' (orbit gate)")
        self._line_bn, = self._ax_beta.plot([], [], color=_C_BETA_N, lw=1.5,
                                             drawstyle="steps-post",
                                             ls="--",
                                             label="β_n' (force gate)")
        self._ax_beta.set_ylim(-0.08, 1.18)
        self._ax_beta.set_yticks([0, 1])
        self._ax_beta.set_ylabel("β  [0/1]", fontsize=9)
        self._ax_beta.set_title("Beta Gates", fontsize=10, fontweight="bold")
        self._ax_beta.legend(fontsize=7, loc="upper right",
                              facecolor="#1a1a2e", labelcolor="white")

        # Alpha (tank-charging coefficient)
        self._line_alpha, = self._ax_alph.plot([], [], color="white", lw=1.2,
                                                label="α (charging coeff)")
        self._ax_alph.set_ylim(-0.08, 1.18)
        self._ax_alph.set_yticks([0.0, 0.5, 1.0])
        self._ax_alph.set_ylabel("α  [0–1]", fontsize=9)
        self._ax_alph.set_title("Charging Coefficient α", fontsize=10,
                                 fontweight="bold")
        self._ax_alph.legend(fontsize=7, loc="upper right",
                              facecolor="#1a1a2e", labelcolor="white")

        self._fig.suptitle("Energy Tank Monitor  —  live", fontsize=12,
                           fontweight="bold", color="white")
        plt.show(block=False)
        plt.pause(0.05)

    # ------------------------------------------------------------------
    def update(self, t: float, tank_info: dict) -> None:
        """Feed one timestep of data.  Call after every ctrl.step().

        Parameters
        ----------
        t         : current simulation time [s]
        tank_info : ctrl.last_tank_info dict (empty dict if tank is disabled)
        """
        if self._frozen or not tank_info:
            return

        s         = tank_info.get("tank_s", 0.0)
        pd        = tank_info.get("pd", 0.0)
        pr        = tank_info.get("pr", 0.0)
        pn        = tank_info.get("pn", 0.0)
        beta_r    = tank_info.get("beta_r", 1.0)
        beta_n    = tank_info.get("beta_n", 1.0)
        beta_rp   = tank_info.get("beta_r_prime", 1.0)
        beta_np   = tank_info.get("beta_n_prime", 1.0)
        alpha     = tank_info.get("alpha", 1.0)
        ds_tank   = tank_info.get("ds_tank", 0.0)

        # dt-scaled net rate  →  J/s
        net = ds_tank / max(self._get_dt(), 1e-9)

        self._t.append(t)
        self._s.append(s)
        self._pd.append(pd)
        self._pr.append(beta_r * pr)
        self._pn.append(beta_n * pn)
        self._net.append(net)
        self._beta_r.append(beta_rp)
        self._beta_n.append(beta_np)
        self._alpha.append(alpha)

        self._call_count += 1
        if self._call_count % self.update_every != 0:
            return

        self._redraw()

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Stop live updates and do a final full-history redraw."""
        self._frozen = True
        self._redraw(full=True)
        self._fig.suptitle("Energy Tank Monitor  —  final", fontsize=12,
                           fontweight="bold", color="white")
        self._fig.canvas.draw()

    # ------------------------------------------------------------------
    def save(self, path: str, dpi: int = 150) -> None:
        """Save the current figure to a file."""
        self._fig.savefig(path, dpi=dpi, facecolor=self._fig.get_facecolor())
        print(f"[TankMonitor] saved → {path}")

    # ------------------------------------------------------------------
    def _get_dt(self) -> float:
        if len(self._t) < 2:
            return 0.001
        return self._t[-1] - self._t[-2]

    # ------------------------------------------------------------------
    def _redraw(self, full: bool = False) -> None:
        t_arr    = np.array(self._t)
        s_arr    = np.array(self._s)
        pd_arr   = np.array(self._pd)
        pr_arr   = np.array(self._pr)
        pn_arr   = np.array(self._pn)
        net_arr  = np.array(self._net)
        br_arr   = np.array(self._beta_r)
        bn_arr   = np.array(self._beta_n)
        al_arr   = np.array(self._alpha)

        if len(t_arr) == 0:
            return

        # Rolling window mask
        t_now = t_arr[-1]
        if full:
            mask = np.ones(len(t_arr), dtype=bool)
        else:
            mask = t_arr >= (t_now - self.window_s)

        t = t_arr[mask]
        if len(t) == 0:
            return

        x0, x1 = t[0], max(t[-1], t[0] + 0.1)

        def _set_xlim(ax):
            ax.set_xlim(x0, x1)
            ax.set_xlabel("time  [s]", fontsize=9)

        # ── Tank level ──────────────────────────────────────────────
        s = s_arr[mask]
        self._line_tank.set_data(t, s)

        # Rebuild fill_between (must remove old and re-add)
        self._fill_tank.remove()
        self._fill_tank = self._ax_tank.fill_between(
            t, s, color=_C_TANK, alpha=0.35)

        # Colour the fill red when tank is low (< 10 % of s_max)
        low = s < 0.1 * self.tank_s_max
        if low.any():
            self._ax_tank.fill_between(
                t, np.where(low, s, 0.0), color=_C_SMAX, alpha=0.25)

        self._ax_tank.set_ylim(-self.tank_s_max * 0.05, self.tank_s_max * 1.15)
        _set_xlim(self._ax_tank)

        # ── Power flows ─────────────────────────────────────────────
        self._line_pd.set_data(t, pd_arr[mask])
        self._line_pr.set_data(t, pr_arr[mask])
        self._line_pn.set_data(t, pn_arr[mask])
        self._ax_pwr.relim()
        self._ax_pwr.autoscale_view(scalex=False)
        _set_xlim(self._ax_pwr)

        # ── Net ṡ ───────────────────────────────────────────────────
        net = net_arr[mask]
        self._line_net.set_data(t, net)

        self._fill_net_pos.remove()
        self._fill_net_neg.remove()
        self._fill_net_pos = self._ax_net.fill_between(
            t, np.maximum(net, 0), color=_C_PD, alpha=0.20)
        self._fill_net_neg = self._ax_net.fill_between(
            t, np.minimum(net, 0), color=_C_PR, alpha=0.20)

        self._ax_net.relim()
        self._ax_net.autoscale_view(scalex=False)
        _set_xlim(self._ax_net)

        # ── Beta gates ──────────────────────────────────────────────
        self._line_br.set_data(t, br_arr[mask])
        self._line_bn.set_data(t, bn_arr[mask])
        _set_xlim(self._ax_beta)

        # ── Alpha ───────────────────────────────────────────────────
        self._line_alpha.set_data(t, al_arr[mask])
        _set_xlim(self._ax_alph)

        # Ticker on the tank panel showing current value
        self._ax_tank.set_title(
            f"Energy Tank Level   s = {s[-1]:.2f} J  /  {self.tank_s_max:.0f} J"
            f"  ({100*s[-1]/self.tank_s_max:.1f} %)",
            fontsize=10, fontweight="bold", color="white",
        )

        plt.pause(0.001)
