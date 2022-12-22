import random

# Generate a random maze using depth-first search
def generate_maze(maze_size):
  maze = [[0 for _ in range(maze_size)] for _ in range(maze_size)]
  stack = []
  visited = set()
  current_cell = (0, 0)
  maze[current_cell[0]][current_cell[1]] = 1
  visited.add(current_cell)
  stack.append(current_cell)

  while stack:
    neighbors = []
    x, y = current_cell
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
      new_x, new_y = x + dx, y + dy
      if (0 <= new_x < maze_size) and (0 <= new_y < maze_size) and ((new_x, new_y) not in visited):
        neighbors.append((new_x, new_y))
    if neighbors:
      new_cell = random.choice(neighbors)
      maze[new_cell[0]][new_cell[1]] = 1
      visited.add(new_cell)
      stack.append(new_cell)
      current_cell = new_cell
    else:
      current_cell = stack.pop()
  return maze

# Initialize Q-learning model
def initialize_q_learning(maze_size):
  q_values = {}
  for i in range(maze_size):
    for j in range(maze_size):
      q_values[(i, j)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
  return q_values

# Select the next action using the Q-learning model
def select_action(q_values, current_state, epsilon):
  if random.random() < epsilon:
    return random.choice(['up', 'down', 'left', 'right'])
  else:
    return max(q_values[current_state], key=q_values[current_state].get)

# Update the Q-learning model based on the current state and action
def update_q_values(q_values, current_state, action, reward, next_state, alpha, gamma):
  q_values[current_state][action] = (1 - alpha) * q_values[current_state][action] + alpha * (reward + gamma * max(q_values[next_state].values()))

# Main function to navigate the maze
def navigate_maze(maze, q_values, alpha, gamma, epsilon, max_steps):
  current_state = (0, 0) # start at the top-left corner
  steps = 0
  while current_state != (len(maze) - 1, len(maze) - 1): # stop when we reach the bottom-right corner
    action = select_action(q_values, current_state, epsilon)
    x, y = current_state
    if action == 'up':
      next_state = (x - 1, y)
      reward = -1 # penalize movement
    elif action == 'down':
      next_state = (x + 1, y)
      reward = -1
    elif action == 'left':
      next_state = (x, y - 1)
      reward = -1
    elif action == 'right':
      next_state = (x, y + 1)
      reward = -1
    if next_state[0] < 0 or next_state[0] >= len(maze) or next_state[1] < 0 or next_state[1] >= len(maze) or maze[next_state[0]][next_state[1]] == 0:
      reward = -10 # penalize attempts to move out of the maze
      next_state = current_state
    update_q_values(q_values, current_state, action, reward, next_state, alpha, gamma)
    current_state = next_state
    steps += 1
    if steps >= max_steps:
      break
  return steps

# Test the maze navigation
maze_size = 10
maze = generate_maze(maze_size)
q_values = initialize_q_learning(maze_size)
alpha = 0.1
gamma = 0.9
epsilon = 0.1
max_steps = 1000

steps = navigate_maze(maze, q_values, alpha, gamma, epsilon, max_steps)
print(f'Number of steps taken to reach the finish: {steps}')
