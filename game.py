import pygame
from pygame.locals import *
import cv2
import numpy as np
import random
from color_detect import *


def initialize_game():
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Brick Racing Game")

    # Set up Pygame window and camera
    global screen_width, screen_height, road_w, roadmark_w, right_lane, left_lane, speed, screen, camera
    screen_width, screen_height = 800, 600
    road_w = int(screen_width / 2.2)
    roadmark_w = int(screen_width / 80)
    right_lane = screen_width / 2 + road_w / 4
    left_lane = screen_width / 2 - road_w / 4

    screen = pygame.display.set_mode((screen_width, screen_height))
    camera = cv2.VideoCapture(0)

    # Set up game variables
    global car, car_loc, car2, car2_loc, car_speed, counter, score, clock, FPS, scaling_factor
    car = pygame.image.load("Part2/self_car.png")
    car_loc = car.get_rect()
    car_loc.center = right_lane, screen_height * 0.8

    car2 = pygame.image.load("Part2/other_car.png")
    car2_loc = car2.get_rect()
    car2_loc.center = left_lane, screen_height * 0.2
    car_speed = 0
    counter = 0
    score = 0
    speed = 5
    scaling_factor = 0.1

    clock = pygame.time.Clock()
    FPS = 60

    # Set up color detection thresholds and road surface
    global lower_threshold, upper_threshold, road_surface
    #GREEN
    lower_threshold = np.array([40, 100, 100])
    upper_threshold = np.array([80, 255, 255])

    road_surface = pygame.Surface((road_w, screen_height), pygame.SRCALPHA)
    pygame.draw.rect(road_surface, (50, 50, 50, 128), (0, 0, road_w, screen_height))
    pygame.draw.line(road_surface, (0, 0, 0), (0, 0), (0, screen_height), 2)
    pygame.draw.line(road_surface, (0, 0, 0), (road_w - 2, 0), (road_w - 2, screen_height), 2)

    # Draw the roadmark rectangles on the road_surface
    pygame.draw.rect(road_surface, (255, 240, 60, 128), (road_w // 2 - roadmark_w // 2, 0, roadmark_w, screen_height))
    pygame.draw.rect(road_surface, (255, 255, 255, 128), (road_w // 2 - roadmark_w // 2 + road_w // 2.2, 0, roadmark_w, screen_height))
    pygame.draw.rect(road_surface, (255, 255, 255, 128), (road_w // 2 - roadmark_w // 2 - road_w // 2.2, 0, roadmark_w, screen_height))

def draw_score_speed():
    # Draw score and speed text on the Pygame window
    global score_text, speed_text
    font = pygame.font.Font(None, 36)
    score_text = font.render("Score: " + str(score), True, (255, 240, 60))
    speed_text = font.render("Speed: " + str(speed), True, (255, 240, 60))

def display_game_over():
    # Display "Game Over" message on the Pygame window
    game_over_font = pygame.font.Font(None, 72)
    game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))
    screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, screen_height // 2 - game_over_text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(3000)

def handle_input():
    # Handle user input for car movement
    global car_speed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        car_speed = -15  # Move left
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        car_speed = 15  # Move right
    else:
        car_speed = 0  # Stop moving when key is released

def move_cars():
    # Move the other car and update score based on time
    global counter, speed, car2_loc, score
    counter += 1
    if counter == 100:
        speed += 1
        counter = 0

    car2_loc[1] += speed
    if car2_loc[1] > screen_height:
        score += 1
        if random.randint(0, 1) == 0:
            car2_loc.center = right_lane, -200
        else:
            car2_loc.center = left_lane, -200

def check_collision():
    # Check for collision between the player's car and the other car
    global car_loc, car2_loc
    if car_loc.colliderect(car2_loc):
        display_game_over()
        return True
    return False

def adjust_car_position(color_position):
    # Adjust the player's car position based on color detection
    global car_loc, scaling_factor
    car_loc.centerx += -scaling_factor * (color_position - screen_width // 2)


def main():
    # Initialize the global variable 'running'
    global running
    running = True

    # Main game loop
    while running:


        # Handle user input
        handle_input()
        
        # Move cars and update game state
        move_cars()

        # Check for collision
        if check_collision():
            break

        # Ensure the player's car stays within the road boundaries
        if car_loc.left < screen_width // 2 - road_w // 2:
            car_loc.left = screen_width // 2 - road_w // 2
        elif car_loc.right > screen_width // 2 + road_w // 2:
            car_loc.right = screen_width // 2 + road_w // 2

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move the player's car based on its speed
        car_loc.move_ip([car_speed, 0])

        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error reading frame from camera.")
            break

        # Perform color detection on the frame
        success, color_position = detect_color(frame, lower_threshold, upper_threshold)
        if success:
            # Adjust the player's car position based on color detection
            adjust_car_position(color_position)

        # Convert the OpenCV image to Pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (screen_width, screen_height))

        # Draw the score and speed on the Pygame window
        draw_score_speed()

        # Blit (draw) the frame, road surface, cars, and text on the Pygame window
        screen.blit(frame, (0, 0))
        screen.blit(road_surface, (screen_width // 2 - road_w // 2, 0))
        screen.blit(car, car_loc)
        screen.blit(car2, car2_loc)
        screen.blit(score_text, (screen_width - score_text.get_width() - 20 , 10))
        screen.blit(speed_text, (screen_width - speed_text.get_width() - 20, 50))

        # Update the Pygame display
        pygame.display.flip()

        # Cap the frame rate using the clock
        clock.tick(FPS)

    # Release the camera and quit Pygame when the game loop ends
    camera.release()
    pygame.quit()

if __name__ == "__main__":
    # Initialize the game and start the main loop
    initialize_game()
    main()
