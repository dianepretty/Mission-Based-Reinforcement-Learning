import pygame
import numpy as np

class CivicRenderer:
    def __init__(self, institutions):
        pygame.init()
        self.width, self.height = 600, 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Rwanda Civic Redress RL Simulation")
        self.font = pygame.font.SysFont("Arial", 18)
        self.inst_names = institutions
        
        # Fixed positions for the 4 institutions
        self.positions = [(100, 100), (450, 100), (100, 300), (450, 300)]

    def render(self, state, last_action, reward):
        self.screen.fill((240, 240, 240)) # Light Gray BG
        
        agent_pos = int(state[-1])
        
        for i, pos in enumerate(self.positions):
            urgency = state[i*2]
            wait_time = state[i*2+1]
            
            # Color intensity based on urgency
            color = (255, 255 - (urgency * 50), 200 - (urgency * 40)) if urgency > 0 else (200, 200, 200)
            
            # Draw Institution Node
            pygame.draw.circle(self.screen, color, pos, 45)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 45, 2)
            
            # Draw Label
            txt = self.font.render(f"{self.inst_names[i]} (U:{int(urgency)})", True, (0,0,0))
            self.screen.blit(txt, (pos[0]-35, pos[1]-10))
            
            # Highlight Agent Location
            if i == agent_pos:
                pygame.draw.circle(self.screen, (0, 255, 0), pos, 50, 4)

        # Dashboard Text
        stats = self.font.render(f"Action: {last_action} | Total Reward: {reward:.2f}", True, (50, 50, 50))
        self.screen.blit(stats, (20, 370))
        
        pygame.display.flip()

    def close(self):
        pygame.quit()