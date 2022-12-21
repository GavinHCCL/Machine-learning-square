import pygame
import numpy as np
import time

# Initialize pygame
pygame.init()

# Set the window size
window_size = (500, 500)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the title of the window
pygame.display.set_caption("Dot Window")

# Set the starting position of the black square
square_pos = (320, 300)

# Set the reward for reaching the green dot
reward = 1

# Set the punishment for moving away from the green dot
punishment = -0.1

# Set the learning rate
alpha = 0.1

# Initialize the Q-table
q_table = np.zeros((500, 500))

# Set the timer to 10 seconds
timer = 10

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Decrement the timer by 1 every iteration
    timer -= 1

    # Check if the timer has reached 0
    if timer == 0:
        # Reset the timer and the position of the black square
        timer = 10
        square_pos = (320, 300)

        # Apply the punishment
        q_table[square_pos[0], square_pos[1]] += punishment

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the green dot
    pygame.draw.circle(screen, (0, 255, 0), (180, 200), 5)

    # Draw the black square
    pygame.draw.rect(screen, (0, 0, 0), (square_pos[0], square_pos[1], 10, 10))

    # Check if the timer has reached 0
    if timer == 0:
        # Check if the black square has reached the green circle
        if square_pos != (180, 200):
            # Reset the timer and the position of the black square
            timer = 10
            square_pos = (320, 300)

            # Apply the punishment
            q_table[square_pos[0], square_pos[1]] += punishment

    # Calculate the distance between the black square and the green dot
    distance = np.sqrt((square_pos[0] - 180)**2 + (square_pos[1]))

    # Check if the black square has moved out of the window
    if square_pos[0] < 0 or square_pos[0] >= 500 or square_pos[1] < 0 or square_pos[1] >= 500:
        continue

    # Update the Q-value for the current state
    q_table[square_pos[0], square_pos[1]] = q_table[square_pos[0], square_pos[1]] + alpha * (reward + q_table[180, 200] - q_table[square_pos[0], square_pos[1]])

    # Choose the action with the highest Q-value
    action = np.argmax(q_table[square_pos[0], square_pos[1]])


    # Update the Q-values of the neighboring states
    for i, action in enumerate(range(4)):
        if action == 0:
            next_state = (square_pos[0] - 1, square_pos[1])
        elif action == 1:
            next_state = (square_pos[0] + 1, square_pos[1])
        elif action == 2:
            next_state = (square_pos[0], square_pos[1] - 1)
        elif action == 3:
            next_state = (square_pos[0], square_pos[1] + 1)

        # Check if the next state is within the bounds of the window
        if next_state[0] >= 0 and next_state[0] < 500 and next_state[1] >= 0 and next_state[1] < 500:
            q_table[next_state[0], next_state[1]] = q_table[next_state[0], next_state[1]] + alpha * (reward + q_table[180, 200] - q_table[next_state[0], next_state[1]])

            # Punish the black square for moving away from the green dot
    if distance < np.sqrt((square_pos[0] - 180)**2 + (square_pos[1] - 200)**2):
        q_table[square_pos[0], square_pos[1]] += punishment

    # Update the display
    pygame.display.update()

    # Sleep for 10 milliseconds
    time.sleep(0.01)

