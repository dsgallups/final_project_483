import pygame
from environment import Game
import neat
import os

class PongGame:
    def __init__ (self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self):
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)

            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    p = PongGame(pygame.display.set_mode((800, 600)), 800, 600)
    p.test_ai()