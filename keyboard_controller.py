#!/usr/bin/env python3
import pygame
import numpy as np

class KeyboardController:
    def __init__(self, vx_scale=1.0, yaw_scale=1.5, smooth=0.2):
        pygame.init()

        self.vx_scale = vx_scale
        self.yaw_scale = yaw_scale
        self.smooth = smooth

        self.vx = 0.0
        self.yaw = 0.0
        self.l_target = 0.20
        self.l_min = 0.15
        self.l_max = 0.35
        self.l_step = 0.01

        self.screen = pygame.display.set_mode((450, 330))
        pygame.display.set_caption("Crazydog Keyboard Controller")

        self.font = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)

    def smooth_update(self, old, new):
        return old * (1 - self.smooth) + new * self.smooth

    def read(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        self.draw_ui()

        target_vx = 0.0
        target_yaw = 0.0

        if keys[pygame.K_w]:
            target_vx = 1.0
        elif keys[pygame.K_s]:
            target_vx = -1.0

        if keys[pygame.K_a]:
            target_yaw = 1.0
        elif keys[pygame.K_d]:
            target_yaw = -1.0

        if keys[pygame.K_SPACE]:
            self.vx = 0.0
            self.yaw = 0.0

        self.vx = self.smooth_update(self.vx, target_vx)
        self.yaw = self.smooth_update(self.yaw, target_yaw)

        return self.vx * self.vx_scale, self.yaw * self.yaw_scale

    def handle_height_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.l_target = min(self.l_target + self.l_step, self.l_max)
        elif keys[pygame.K_e]:
            self.l_target = max(self.l_target - self.l_step, self.l_min)

    def get_command(self):
        v_ref, yaw_rate_ref = self.read()
        self.handle_height_keys()
        return v_ref, yaw_rate_ref, self.l_target

    def draw_ui(self):
        # ---------- Colors ----------
        BG = (24, 26, 32)
        PANEL = (36, 40, 52)
        TEXT = (220, 220, 230)
        ACCENT = (100, 200, 255)
        GREEN = (120, 220, 160)
        ORANGE = (255, 200, 120)
        GRAY = (120, 120, 130)

        self.screen.fill(BG)

        # ---------- Header ----------
        pygame.draw.rect(self.screen, PANEL, (0, 0, 450, 50))
        title = self.font_large.render("Crazydog · Keyboard HUD", True, ACCENT)
        self.screen.blit(title, (20, 12))

        # ---------- Helper ----------
        def draw_bar(x, y, w, h, value, vmin, vmax, color):
            pygame.draw.rect(self.screen, (60, 60, 70), (x, y, w, h), border_radius=4)
            ratio = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
            pygame.draw.rect(
                self.screen,
                color,
                (x, y, int(w * ratio), h),
                border_radius=4
            )

        y0 = 70

        # ---------- Forward Velocity ----------
        label = self.font.render("FORWARD VELOCITY", True, TEXT)
        self.screen.blit(label, (30, y0))

        v = self.vx * self.vx_scale
        val_text = self.font_large.render(f"{v:+.2f}  m/s", True, GREEN)
        self.screen.blit(val_text, (30, y0 + 22))

        draw_bar(200, y0 + 28, 200, 10, v, -1.0, 1.0, GREEN)

        # ---------- Yaw Rate ----------
        y1 = y0 + 80
        label = self.font.render("YAW RATE", True, TEXT)
        self.screen.blit(label, (30, y1))

        yaw = self.yaw * self.yaw_scale
        val_text = self.font_large.render(f"{yaw:+.2f}  rad/s", True, ORANGE)
        self.screen.blit(val_text, (30, y1 + 22))

        draw_bar(200, y1 + 28, 200, 10, yaw, -1.5, 1.5, ORANGE)

        # ---------- Height ----------
        y2 = y1 + 80
        label = self.font.render("HEIGHT (l)", True, TEXT)
        self.screen.blit(label, (30, y2))

        h_text = self.font_large.render(f"{self.l_target:.3f}  m", True, ACCENT)
        self.screen.blit(h_text, (30, y2 + 22))

        # slider
        pygame.draw.line(self.screen, GRAY, (200, y2 + 32), (400, y2 + 32), 4)
        ratio = (self.l_target - self.l_min) / (self.l_max - self.l_min)
        knob_x = int(200 + ratio * 200)
        pygame.draw.circle(self.screen, ACCENT, (knob_x, y2 + 32), 7)

        # min / max
        min_t = self.font.render(f"{self.l_min:.2f}", True, GRAY)
        max_t = self.font.render(f"{self.l_max:.2f}", True, GRAY)
        self.screen.blit(min_t, (195, y2 + 45))
        self.screen.blit(max_t, (370, y2 + 45))

        pygame.display.flip()

