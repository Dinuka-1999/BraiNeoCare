import pygame

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Floating Apple Game")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

class Apple(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("apple.png").convert_alpha()  # Load apple image
        self.image = pygame.transform.scale(self.image, (100, 100))  # Resize image
        self.rect = self.image.get_rect()
        self.rect.x = (SCREEN_WIDTH - self.rect.width) // 2
        self.rect.y = SCREEN_HEIGHT - self.rect.height

    def update(self, is_space_pressed):
        if is_space_pressed:
            self.rect.y -= 5
            if self.rect.y < 0:
                self.rect.y = 0
        else:
            self.rect.y += 5
            if self.rect.y > SCREEN_HEIGHT - self.rect.height:
                self.rect.y = SCREEN_HEIGHT - self.rect.height

all_sprites = pygame.sprite.Group()
apple = Apple()
all_sprites.add(apple)

# Game loop
running = True
is_space_pressed = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_space_pressed = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                is_space_pressed = False

    # Update
    all_sprites.update(is_space_pressed)

    # Draw
    screen.fill(WHITE)
    all_sprites.draw(screen)

    # Refresh screen
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
