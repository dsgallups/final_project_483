import pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

#window title
pygame.display.set_caption("Pong")

#max FPS
#can be changed when training models
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0,0,0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7
SCORE_FONT = pygame.font.SysFont("serif", 50)
WINNING_SCORE = 10

class Paddle:
    COLOR = WHITE
    VEL = 4

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

class Ball:
    MAX_VEL = 5
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)
    
    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.x_vel = -1 * self.x_vel
        self.y_vel = 0


def draw(win, paddles, ball, left_score, right_score):
    win.fill(BLACK)

    left_score_text = SCORE_FONT.render(f"{left_score}", 1, WHITE)
    right_score_text = SCORE_FONT.render(f"{right_score}", 1, WHITE)
    win.blit(left_score_text, (WIDTH//4 - left_score_text.get_width()//2, 20))
    win.blit(right_score_text, (WIDTH * (3/4) - right_score_text.get_width()//2, 20))



    for paddle in paddles:
        paddle.draw(WINDOW)

    # for that nice center line
    for i in range(0, HEIGHT, HEIGHT//20):
        if i % 2 == 0:
            pygame.draw.rect(win, WHITE, (WIDTH//2 - 1, i, 2, HEIGHT//45))

    ball.draw(WINDOW)

    # most intensive command
    pygame.display.update()


def handle_collision(ball, left_paddle, right_paddle):

    #ceiling
    if ball.y + ball.radius >= HEIGHT or ball.y - ball.radius <= 0:
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



def handle_paddle_movement(keys, left_paddle, right_paddle):
    if keys[pygame.K_w] and left_paddle.y - left_paddle.VEL >= 0:
        left_paddle.move(up=True)

    if keys[pygame.K_s] and left_paddle.y + left_paddle.height + left_paddle.VEL <= HEIGHT:
        left_paddle.move(up=False)

    if keys[pygame.K_i] and right_paddle.y - right_paddle.VEL >= 0:
        right_paddle.move(up=True)
    
    if keys[pygame.K_k] and right_paddle.y + right_paddle.height + right_paddle.VEL <= HEIGHT:
        right_paddle.move(up=False)

def game_over(ball, left_paddle, right_paddle, message):
            text = SCORE_FONT.render(message, 1, WHITE)
            WINDOW.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
            pygame.display.update()
            pygame.time.delay(5000)
            left_paddle.reset()
            right_paddle.reset()
            ball.reset()


class Game:
    def __init__(self, window, width, height):
        self.window = window
        self.width = width
        self.height = height
        self.left_paddle = Paddle()


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle



def main():
    run = True
    clock = pygame.time.Clock()

    left_paddle = Paddle(10, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = Ball(WIDTH//2, HEIGHT//2, BALL_RADIUS)

    left_score = 0
    right_score = 0

    while run:
        clock.tick(FPS)
        draw(WINDOW, [left_paddle, right_paddle], ball, left_score, right_score)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

                #breaks out of for loop
                break

        keys = pygame.key.get_pressed()
        handle_paddle_movement(keys, left_paddle, right_paddle)
        ball.move()
        handle_collision(ball, left_paddle, right_paddle)

        if ball.x < left_paddle.x + left_paddle.width:
            right_score += 1

            if right_score >= WINNING_SCORE:
                game_over(ball, left_paddle, right_paddle, "Right Player Won!")
                left_score = 0
                right_score = 0
            else:
                ball.reset()
        elif ball.x > right_paddle.x:
            left_score += 1

            if left_score >= WINNING_SCORE:
                game_over(ball, left_paddle, right_paddle, "Right Player Won!")
                left_score = 0
                right_score = 0
            else:
                ball.reset()
        
        

    pygame.quit()

# this will only run if you call this file, but not if it was called from another file.
if __name__ == "__main__":
	main()
