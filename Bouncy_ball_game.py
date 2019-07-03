import pygame
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

SCREENSIZE = 1000, 600


class Ball:
    def __init__(self, color, x, y, size=20, speed=0):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.speed = speed  # Balls vertical speed

    def apply_gravity(self):
        if self.speed <= 5:
            self.speed += 1

        self.y += self.speed

    def out_of_bounds(self):
        return 0 >= self.y or self.y + self.size >= SCREENSIZE[1]


class PipePair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = GREEN
        self.size = 70
        self.speed = 2  # Pipe pairs horizontal speed
        self.space = 150  # Space between pipe pairs

    def move(self):
        self.x -= self.speed


def display_score(current_score, screen):
    font = pygame.font.SysFont(None, 60)
    text = font.render("Score: {}".format(current_score), True, WHITE)
    screen.blit(text, [0, 0])


def collision(object_1, object_2):
    return 220 >= object_2.x >= 130 and (object_2.y + object_2.space <= object_1.y + 20 or object_1.y <= object_2.y)


def bouncy_ball():
    pygame.init()
    pygame.display.set_caption('Bouncy Ball')

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREENSIZE)
    score = 0

    ball = Ball(YELLOW, 200, 300)
    pipe_pairs = []

    exit_program = False

    while not exit_program:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                exit_program = True

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:  # Space pressed
                    ball.speed = -10

        # Logic section
        if len(pipe_pairs) == 0 or pipe_pairs[-1].x == 500:
            pipe_pairs.append(PipePair(SCREENSIZE[0], random.randint(50, 400)))

        if pipe_pairs[0].x == -70:
            pipe_pairs.pop(0)

        elif pipe_pairs[0].x == 150:
            score += 1

        if ball.out_of_bounds() or collision(ball, pipe_pairs[0]):
            ball.color = RED
            exit_program = True

        # Drawing section
        screen.fill(BLACK)

        for pipe_pair in pipe_pairs:
            pygame.draw.rect(screen, GREEN, [pipe_pair.x, 0, pipe_pair.size, pipe_pair.y])
            pygame.draw.rect(screen, GREEN, [pipe_pair.x, pipe_pair.y + pipe_pair.space, pipe_pair.size, SCREENSIZE[1]])
            pipe_pair.move()

        pygame.draw.circle(screen, ball.color, [ball.x, ball.y], ball.size)
        ball.apply_gravity()

        display_score(score, screen)

        pygame.display.flip()
        clock.tick(60)

    exit_window = False
    while not exit_window:  # Exit screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_window = True
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    exit_window = True
                    bouncy_ball()

    pygame.display.quit()


if __name__ == '__main__':
    bouncy_ball()
