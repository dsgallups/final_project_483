import pygame
import math
import random

class Ball:
    MAX_VEL = 5
    COLOR = (255,255,255)

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius

        angle = self._get_random_angle(-40, 40, [0])
        self.x_vel = math.cos(angle) * self.MAX_VEL
        self.y_vel = math.sin(angle) * self.MAX_VEL

    def _get_random_angle(self, min_angle, max_angle, excluded):
        #bad code universally
        angle = 0
        while angle in excluded:
            angle = math.radians(random.randrange(min_angle, max_angle))
        
        return angle

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)
    
    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


        angle = self._get_random_angle(-30, 30, [0])

        negative_multiplier = -1 if self.x_vel > 0 else 1
        self.x_vel = negative_multiplier * abs(math.cos(angle) * self.MAX_VEL)
        self.y_vel = math.sin(angle) * self.MAX_VEL