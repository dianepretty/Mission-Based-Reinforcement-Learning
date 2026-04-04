"""
CivicRenderer — Pygame Dashboard for Rwanda Civic Issue RL
===========================================================
A visually rich, real-time dashboard showing:
  Left panel   : Report queue with urgency color bars and category icons
  Center panel : Institution map with animated dispatch lines and workload bars
  Right panel  : Live stats — trust meter, reward graph, step counter
  Bottom bar   : Scrolling event log with color-coded entries
"""

import pygame
import pygame.gfxdraw
import numpy as np
import math
import time
from typing import Optional

# ── Color palette ────────────────────────────────────────────────────────────

# Backgrounds
BG_DARK        = (15,  20,  30)
BG_PANEL       = (22,  30,  45)
BG_CARD        = (30,  40,  58)
BG_CARD_HOVER  = (38,  50,  72)

# Urgency colors (1→5)
URGENCY_COLORS = {
    0: (60,  70,  90),    # empty
    1: (46, 160,  67),    # low — green
    2: (120, 190,  50),   # medium-low
    3: (240, 180,  20),   # medium — amber
    4: (230, 110,  30),   # high — orange
    5: (220,  45,  45),   # critical — red
}

# Institution colors
INST_COLORS = [
    (80, 140, 220),   # NLA — blue
    (60, 190, 160),   # WASAC — teal
    (180, 100, 220),  # REG — purple
    (220, 140,  60),  # MININFRA — amber
]

# Category colors (matching institutions)
CATEGORY_COLORS = INST_COLORS

# Text
TEXT_PRIMARY   = (220, 225, 235)
TEXT_SECONDARY = (140, 150, 170)
TEXT_MUTED     = (80,  90, 110)

# Accent
ACCENT_GREEN   = (46, 200, 100)
ACCENT_RED     = (220,  60,  60)
ACCENT_BLUE    = (80, 160, 240)
ACCENT_GOLD    = (240, 190,  50)

# Category icons (text symbols)
CATEGORY_ICONS = ["⬡", "≋", "⚡", "⬟"]  # Land, Water, Elec, Roads
CATEGORY_NAMES = ["Land", "Water", "Electricity", "Roads"]
INST_NAMES     = ["NLA", "WASAC", "REG", "MININFRA"]


class CivicRenderer:
    """Full pygame dashboard for the CivicIssueEnv."""

    def __init__(self, width: int = 1100, height: int = 680):
        pygame.init()
        pygame.display.set_caption("Rwanda Civic Dispatch — RL Agent Dashboard")

        self.W, self.H = width, height
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_md  = pygame.font.SysFont("Arial", 15, bold=False)
        self.font_sm  = pygame.font.SysFont("Arial", 13, bold=False)
        self.font_xs  = pygame.font.SysFont("Arial", 11, bold=False)
        self.font_bold= pygame.font.SysFont("Arial", 14, bold=True)

        # Layout zones
        self.left_x   = 10
        self.left_w   = 280
        self.center_x = 300
        self.center_w = 460
        self.right_x  = 770
        self.right_w  = 320
        self.log_y    = self.H - 110
        self.log_h    = 100

        # Institution node positions (center panel)
        cx = self.center_x + self.center_w // 2
        cy = 340
        r  = 145
        self.inst_pos = [
            (cx - r,         cy - int(r * 0.7)),   # NLA — top left
            (cx + r,         cy - int(r * 0.7)),   # WASAC — top right
            (cx - r,         cy + int(r * 0.7)),   # REG — bottom left
            (cx + r,         cy + int(r * 0.7)),   # MININFRA — bottom right
        ]

        # Animation state
        self.dispatch_anims = []    # [{from_pos, to_pos, t, color}]
        self.pulse_timers   = [0.0] * 4   # per institution
        self.reward_history = []
        self.trust_history  = []
        self.frame          = 0
        self.episode_count  = 1        
        self.start_time     = time.time()
        self.last_info      = None

    # ── Main render entry point ───────────────────────────────────────────────

    def render(self, info: dict) -> Optional[np.ndarray]:
        """Called each env step. info is env._get_info()."""
        self.last_info = info
        self.frame += 1

        if info.get("step", 0) == 1:   # ← ADD THIS
            self.episode_count += 1    # ← AND THIS

        # Track histories
        self.reward_history.append(info.get("total_reward", 0))
        self.trust_history.append(info.get("trust", 70))
        if len(self.reward_history) > 200:
            self.reward_history.pop(0)
            self.trust_history.pop(0)

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        self.screen.fill(BG_DARK)

        # Draw panels
        self._draw_header(info)
        self._draw_report_queue(info)
        self._draw_institution_map(info)
        self._draw_stats_panel(info)
        self._draw_event_log(info)
        self._draw_dispatch_animations()

        pygame.display.flip()
        self.clock.tick(10)   # 30 fps cap

        return None

    # ── Header ───────────────────────────────────────────────────────────────

    def _draw_header(self, info: dict):
        # Background strip
        pygame.draw.rect(self.screen, BG_PANEL, (0, 0, self.W, 48))
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, 48), (self.W, 48), 1)

        title = self.font_lg.render("Rwanda Civic Dispatch  ·  RL Agent Dashboard", True, TEXT_PRIMARY)
        self.screen.blit(title, (16, 14))

        # Right side: step & episode info
        step  = info.get("step", 0)
        trust = info.get("trust", 70)
        total_r = info.get("total_reward", 0)
        elapsed = int(time.time() - self.start_time)

        right_txt = f"Episode {self.episode_count} | Step {step:03d}/200   |   Trust {trust:.0f}%   |   Score {total_r:+.1f}   |   {elapsed}s"
        rtxt = self.font_md.render(right_txt, True, TEXT_SECONDARY)
        self.screen.blit(rtxt, (self.W - rtxt.get_width() - 16, 16))

    # ── Left panel: report queue ──────────────────────────────────────────────

    def _draw_report_queue(self, info: dict):
        reports = info.get("reports", [])
        x, y = self.left_x, 58
        w    = self.left_w

        # Panel background
        pygame.draw.rect(self.screen, BG_PANEL, (x, y, w, self.log_y - y - 8), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x, y, w, self.log_y - y - 8), 1, border_radius=10)

        # Header
        hdr = self.font_bold.render("REPORT QUEUE", True, ACCENT_BLUE)
        self.screen.blit(hdr, (x + 12, y + 10))

        active = sum(1 for r in reports if r["active"])
        cnt = self.font_sm.render(f"{active}/10 active", True, TEXT_SECONDARY)
        self.screen.blit(cnt, (x + w - cnt.get_width() - 12, y + 12))

        # Separator
        pygame.draw.line(self.screen, TEXT_MUTED, (x + 10, y + 32), (x + w - 10, y + 32), 1)

        # Report cards
        card_h = 46
        for i, report in enumerate(reports):
            cy = y + 40 + i * (card_h + 4)
            self._draw_report_card(report, x + 8, cy, w - 16, card_h, i)

    def _draw_report_card(self, report: dict, x: int, y: int, w: int, h: int, idx: int):
        if not report["active"]:
            # Empty slot — dim placeholder
            pygame.draw.rect(self.screen, BG_DARK, (x, y, w, h), border_radius=6)
            empty = self.font_xs.render("— empty slot —", True, TEXT_MUTED)
            self.screen.blit(empty, (x + w // 2 - empty.get_width() // 2, y + h // 2 - 6))
            return

        urgency  = report["urgency"]
        category = report["category"]
        color    = URGENCY_COLORS.get(urgency, URGENCY_COLORS[1])
        cat_col  = CATEGORY_COLORS[category]

        # Card background — slightly brighter for high urgency
        bg = BG_CARD_HOVER if urgency >= 4 else BG_CARD
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=6)

        # Urgency stripe on left edge
        pygame.draw.rect(self.screen, color, (x, y, 4, h), border_radius=3)

        # Category color dot
        pygame.draw.circle(self.screen, cat_col, (x + 20, y + h // 2), 7)

        # Category name
        cat_txt = self.font_sm.render(CATEGORY_NAMES[category], True, TEXT_PRIMARY)
        self.screen.blit(cat_txt, (x + 34, y + 6))

        # Urgency label
        urg_txt = self.font_bold.render(f"U:{urgency}", True, color)
        self.screen.blit(urg_txt, (x + 34, y + 22))

        # Wait time
        wait_txt = self.font_xs.render(f"{report['days_waiting']}d wait", True, TEXT_SECONDARY)
        self.screen.blit(wait_txt, (x + 80, y + 24))

        # Evidence bar
        ev_w = int((w - 130) * report["evidence"])
        ev_bg = (40, 50, 70)
        pygame.draw.rect(self.screen, ev_bg, (x + 140, y + 28, w - 150, 8), border_radius=4)
        if ev_w > 0:
            pygame.draw.rect(self.screen, ACCENT_GREEN, (x + 140, y + 28, ev_w, 8), border_radius=4)

        ev_lbl = self.font_xs.render("evidence", True, TEXT_MUTED)
        self.screen.blit(ev_lbl, (x + 140, y + 8))

        # Slot number
        slot_txt = self.font_xs.render(f"#{idx}", True, TEXT_MUTED)
        self.screen.blit(slot_txt, (x + w - 22, y + 4))

        # Urgency flash for critical reports
        if urgency == 5:
            pulse = abs(math.sin(time.time() * 3)) * 60
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((220, 45, 45, int(pulse)))
            self.screen.blit(overlay, (x, y))

    # ── Center panel: institution map ─────────────────────────────────────────

    def _draw_institution_map(self, info: dict):
        institutions = info.get("institutions", [])
        x, y = self.center_x, 58
        w, h = self.center_w, self.log_y - y - 8

        # Panel background
        pygame.draw.rect(self.screen, BG_PANEL, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x, y, w, h), 1, border_radius=10)

        # Header
        hdr = self.font_bold.render("INSTITUTION MAP", True, ACCENT_BLUE)
        self.screen.blit(hdr, (x + 12, y + 10))

        # Last action label
        action_lbl = info.get("last_action", "—")
        if len(action_lbl) > 52:
            action_lbl = action_lbl[:49] + "..."
        act_surf = self.font_sm.render(action_lbl, True, ACCENT_GOLD)
        self.screen.blit(act_surf, (x + 12, y + 28))

        pygame.draw.line(self.screen, TEXT_MUTED, (x + 10, y + 46), (x + w - 10, y + 46), 1)

        # Connection lines between institutions (background mesh)
        for i in range(len(self.inst_pos)):
            for j in range(i + 1, len(self.inst_pos)):
                pygame.draw.line(self.screen, (35, 45, 65),
                                 self.inst_pos[i], self.inst_pos[j], 1)

        # Institution nodes
        for i, inst in enumerate(institutions):
            self._draw_institution_node(i, inst)

    def _draw_institution_node(self, idx: int, inst: dict):
        pos   = self.inst_pos[idx]
        color = INST_COLORS[idx]
        name  = INST_NAMES[idx]

        capacity   = inst.get("capacity", 1.0)
        workload   = inst.get("workload", 0)
        res_rate   = inst.get("resolution_rate", 0.8)

        # Outer glow ring (size reflects workload)
        glow_r = 36 + workload * 4
        pulse  = abs(math.sin(time.time() * 2 + idx)) * 0.4 + 0.6
        glow_color = tuple(int(c * 0.3 * pulse) for c in color)
        pygame.draw.circle(self.screen, glow_color, pos, glow_r)

        # Main circle
        node_r = 32
        pygame.draw.circle(self.screen, color, pos, node_r)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, pos, node_r, 2)

        # Overload indicator
        if capacity < 0.3:
            ov_r = node_r + 5
            ov_pulse = abs(math.sin(time.time() * 5)) * 150
            pygame.draw.circle(self.screen, (220, 60, 60, int(ov_pulse)), pos, ov_r, 3)

        # Institution name inside node
        name_surf = self.font_bold.render(name, True, (10, 10, 20))
        self.screen.blit(name_surf, (pos[0] - name_surf.get_width() // 2,
                                     pos[1] - name_surf.get_height() // 2))

        # Capacity bar below node
        bar_w, bar_h = 70, 8
        bx = pos[0] - bar_w // 2
        by = pos[1] + node_r + 6
        pygame.draw.rect(self.screen, (40, 50, 70), (bx, by, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * capacity)
        cap_color = ACCENT_GREEN if capacity > 0.5 else (240, 140, 40) if capacity > 0.25 else ACCENT_RED
        if fill_w > 0:
            pygame.draw.rect(self.screen, cap_color, (bx, by, fill_w, bar_h), border_radius=4)

        # Resolution rate label
        rr_txt = self.font_xs.render(f"{res_rate*100:.0f}% resolve", True, TEXT_SECONDARY)
        self.screen.blit(rr_txt, (pos[0] - rr_txt.get_width() // 2, by + 11))

        # Workload dots above node
        for d in range(int(workload)):
            dx = pos[0] - (workload * 8) // 2 + d * 9
            dy = pos[1] - node_r - 12
            pygame.draw.circle(self.screen, color, (dx, dy), 4)

    # ── Dispatch animation ────────────────────────────────────────────────────

    def add_dispatch_animation(self, from_pos, to_pos, color):
        self.dispatch_anims.append({
            "from": from_pos, "to": to_pos, "t": 0.0, "color": color
        })

    def _draw_dispatch_animations(self):
        finished = []
        for anim in self.dispatch_anims:
            anim["t"] += 0.04
            t = min(anim["t"], 1.0)

            # Interpolated position of the "packet"
            px = anim["from"][0] + (anim["to"][0] - anim["from"][0]) * t
            py = anim["from"][1] + (anim["to"][1] - anim["from"][1]) * t

            # Trail line
            pygame.draw.line(self.screen, anim["color"],
                             anim["from"], (int(px), int(py)), 2)

            # Packet dot
            pygame.draw.circle(self.screen, TEXT_PRIMARY, (int(px), int(py)), 6)
            pygame.draw.circle(self.screen, anim["color"],  (int(px), int(py)), 4)

            if anim["t"] >= 1.0:
                finished.append(anim)

        for f in finished:
            self.dispatch_anims.remove(f)

    # ── Right panel: stats ────────────────────────────────────────────────────

    def _draw_stats_panel(self, info: dict):
        x, y = self.right_x, 58
        w, h = self.right_w, self.log_y - y - 8

        pygame.draw.rect(self.screen, BG_PANEL, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x, y, w, h), 1, border_radius=10)

        hdr = self.font_bold.render("LIVE STATS", True, ACCENT_BLUE)
        self.screen.blit(hdr, (x + 12, y + 10))
        pygame.draw.line(self.screen, TEXT_MUTED, (x + 10, y + 30), (x + w - 10, y + 30), 1)

        cy = y + 40

        # Trust meter
        trust = info.get("trust", 70)
        cy = self._draw_trust_meter(x + 10, cy, w - 20, trust)
        cy += 14

        # Key numbers
        stats = [
            ("Resolved today",    str(info.get("resolved_today", 0)),   ACCENT_GREEN),
            ("Critical pending",  str(info.get("critical_pending", 0)), ACCENT_RED),
            ("Total score",       f"{info.get('total_reward', 0):+.1f}", ACCENT_GOLD),
            ("Step",              f"{info.get('step', 0)} / 200",        TEXT_SECONDARY),
        ]
        for label, value, color in stats:
            lbl_s = self.font_sm.render(label, True, TEXT_SECONDARY)
            val_s = self.font_bold.render(value, True, color)
            self.screen.blit(lbl_s, (x + 12, cy))
            self.screen.blit(val_s, (x + w - val_s.get_width() - 12, cy))
            pygame.draw.line(self.screen, BG_CARD, (x + 10, cy + 18), (x + w - 10, cy + 18), 1)
            cy += 22

        cy += 6

        # Reward graph
        cy = self._draw_sparkline(x + 10, cy, w - 20, 80,
                                  self.reward_history, "Cumulative reward", ACCENT_GOLD)
        cy += 14

        # Trust graph
        self._draw_sparkline(x + 10, cy, w - 20, 60,
                             self.trust_history, "Trust over time", ACCENT_BLUE)

    def _draw_trust_meter(self, x, y, w, trust) -> int:
        lbl = self.font_bold.render("CITIZEN TRUST", True, TEXT_PRIMARY)
        self.screen.blit(lbl, (x, y))

        pct_txt = self.font_bold.render(f"{trust:.0f}%", True,
                                        ACCENT_GREEN if trust > 50 else ACCENT_RED)
        self.screen.blit(pct_txt, (x + w - pct_txt.get_width(), y))

        y += 20
        bar_h = 18
        # Background
        pygame.draw.rect(self.screen, BG_DARK, (x, y, w, bar_h), border_radius=9)
        # Fill
        fill = int(w * trust / 100)
        if fill > 0:
            # Gradient color: green at high, red at low
            if trust > 60:
                tc = ACCENT_GREEN
            elif trust > 35:
                tc = (240, 180, 40)
            else:
                tc = ACCENT_RED
            pygame.draw.rect(self.screen, tc, (x, y, fill, bar_h), border_radius=9)

        # Collapse threshold marker
        marker_x = x + int(w * 0.20)
        pygame.draw.line(self.screen, ACCENT_RED, (marker_x, y - 3), (marker_x, y + bar_h + 3), 2)
        danger = self.font_xs.render("collapse", True, ACCENT_RED)
        self.screen.blit(danger, (marker_x - danger.get_width() // 2, y + bar_h + 4))

        # Border
        pygame.draw.rect(self.screen, TEXT_MUTED, (x, y, w, bar_h), 1, border_radius=9)

        return y + bar_h + 16

    def _draw_sparkline(self, x, y, w, h, data, label, color) -> int:
        lbl = self.font_sm.render(label, True, TEXT_SECONDARY)
        self.screen.blit(lbl, (x, y))
        y += 16

        # Background
        pygame.draw.rect(self.screen, BG_DARK, (x, y, w, h), border_radius=6)

        if len(data) > 1:
            mn, mx = min(data), max(data)
            span = max(mx - mn, 1)
            pts = []
            for i, v in enumerate(data[-w:]):
                px = x + int(i * w / max(len(data[-w:]) - 1, 1))
                py = y + h - int((v - mn) / span * h)
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 2)

            # Current value label
            cur = self.font_xs.render(f"{data[-1]:.1f}", True, color)
            self.screen.blit(cur, (x + w - cur.get_width() - 2, y + 2))

        pygame.draw.rect(self.screen, TEXT_MUTED, (x, y, w, h), 1, border_radius=6)
        return y + h

    # ── Bottom log ────────────────────────────────────────────────────────────

    def _draw_event_log(self, info: dict):
        x, y = 10, self.log_y
        w, h = self.W - 20, self.log_h

        pygame.draw.rect(self.screen, BG_PANEL, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x, y, w, h), 1, border_radius=8)

        hdr = self.font_bold.render("EVENT LOG", True, ACCENT_BLUE)
        self.screen.blit(hdr, (x + 12, y + 8))

        log = info.get("event_log", [])
        for i, entry in enumerate(reversed(log[-5:])):
            alpha = 255 - i * 40
            if "COLLAPSE" in entry or "✗" in entry:
                col = (220, 80, 80, alpha)
            elif "+" in entry or "Dispatched" in entry:
                col = (100, 200, 130, alpha)
            elif "↑" in entry:
                col = (240, 180, 50, alpha)
            else:
                col = (140, 155, 175, alpha)

            color_rgb = col[:3]
            surf = self.font_sm.render(entry, True, color_rgb)
            self.screen.blit(surf, (x + 12, y + 26 + i * 14))

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self):
        pygame.quit()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from environment.custom_env import CivicIssueEnv

    env      = CivicIssueEnv(render_mode="human")
    renderer = env._renderer  # will be created on first render call
    obs, info = env.reset(seed=0)

    print("Running random agent visualization — close the window to stop.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        if term or trunc:
            obs, info = env.reset()

        pygame.time.wait(400)   # slow enough to see decisions

    env.close()