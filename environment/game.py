from .paddle import Paddle
from .ball import Ball
import pygame
import math
import random
pygame.init()

class GameInformation:
    def __init__(self, left_hits, right_hits, left_score, right_score):
        self.left_hits = left_hits
        self.right_hits = right_hits
        self.left_score = left_score
        self.right_score = right_score

class Game:
    SCORE_FONT = pygame.font.SysFont("serif", 50)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    RED = (255, 0, 0)

    def __init__(self, window, width, height):
        self.window = window
        self.width = width
        self.height = height
        self.left_paddle = Paddle(10, self.height//2 - Paddle.HEIGHT//2, Paddle.WIDTH, Paddle.HEIGHT)
        self.right_paddle = Paddle(self.width - 10 - Paddle.WIDTH, self.height//2 - Paddle.HEIGHT//2, Paddle.WIDTH, Paddle.HEIGHT)
        self.ball = Ball(self.width//2, self.height//2, 7)

        self.left_score = 0
        self.right_score = 0
        self.left_hits = 0
        self.right_hits = 0

    def _draw_score(self):
        left_score_text = self.SCORE_FONT.render(f"{self.left_score}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{self.right_score}", 1, self.WHITE)
        self.window.blit(left_score_text, (self.width//4 - left_score_text.get_width()//2, 20))
        self.window.blit(right_score_text, (self.width * (3/4) - right_score_text.get_width()//2, 20))

    def _draw_hits(self):
            hits_text = self.SCORE_FONT.render(
                f"{self.left_hits + self.right_hits}", 1, self.RED)
            self.window.blit(hits_text, (self.width //
                                        2 - hits_text.get_width()//2, 10))

    def _draw_divider(self):
        for i in range(0, self.height, self.height//20):
            if i % 2 == 0:
                pygame.draw.rect(self.window, self.WHITE, (self.width//2 - 1, i, 2, self.height//45))
    
    def _handle_collision(self):
        ball = self.ball
        left_paddle = self.left_paddle
        right_paddle = self.right_paddle

        #ceiling
        if ball.y + ball.radius >= self.height or ball.y - ball.radius <= 0:
            ball.y_vel *= -1
        
        #left paddle
        if ball.x_vel < 0:
            if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
                if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                    ball.x_vel *= -1
                    #handle change in y
                    middle_y = left_paddle.y + left_paddle.height / 2
                    difference_in_y = middle_y - ball.y
                    #adjusts speed according to distance from center hit
                    reduction_factor = (left_paddle.height /2 ) / ball.MAX_VEL
                    ball.y_vel= -1 * (difference_in_y / reduction_factor)

                    #give the y_velocity a little randomness
                    angle = math.radians(random.randrange(-5, 5))
                    ball.y_vel = math.sin(angle) + ball.y_vel

                    
                    self.left_hits += 1

        #right paddle
        else:
            if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:
                if ball.x + ball.radius >= right_paddle.x:
                    ball.x_vel *= -1
                    #handle change in y
                    middle_y = right_paddle.y + right_paddle.height / 2
                    difference_in_y = middle_y - ball.y
                    #adjusts speed according to distance from center hit
                    reduction_factor = (right_paddle.height /2 ) / ball.MAX_VEL
                    ball.y_vel= -1 * (difference_in_y / reduction_factor)


                    #give the y_velocity a little randomness
                    angle = math.radians(random.randrange(-5, 5))
                    ball.y_vel = math.sin(angle) + ball.y_vel
                

                    self.right_hits += 1
    
    # def draw(self, draw_score=True, draw_hits=False)
    def draw(self, draw_score=True, draw_hits=False):
        self.window.fill(self.BLACK)

        self._draw_divider()

        if draw_score:
            self._draw_score()
        
        if draw_hits:
            self._draw_hits()

        for paddle in [self.left_paddle, self.right_paddle]:
            paddle.draw(self.window)

        self.ball.draw(self.window)


    def move_paddle(self, left=True, up=True):
        
        if left:
            if up and self.left_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.left_paddle.y + Paddle.HEIGHT > self.height:
                return False
            self.left_paddle.move(up)
        else:
            if up and self.right_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.right_paddle.y + Paddle.HEIGHT > self.height:
                return False
            self.right_paddle.move(up)

        return True

    def loop(self):

        self.ball.move()
        self._handle_collision()

        if self.ball.x < self.left_paddle.x + self.left_paddle.width:
            self.right_score += 1
            self.ball.reset()

        elif self.ball.x > self.right_paddle.x:
            self.left_score += 1
            self.ball.reset()

        
        return GameInformation(self.left_hits, self.right_hits, self.left_score, self.right_score)

    def reset(self):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.left_hits = 0
        self.right_score = 0
        self.right_hits = 0