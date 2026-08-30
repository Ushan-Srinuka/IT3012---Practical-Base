import random

class SimpleReflexAgent:
    """Step 1.2: Simple Reflex Agent (No Memory)"""
    def sense_and_act(self, percept: dict) -> str:
        # Condition-Action Rules mapping directly from current percept to action
        if percept['food_here']:
            return 'Stay' 
        elif percept['wall_ahead']:
            # Agent only knows a wall is ahead, so it blindly turns
            return random.choice(['Left', 'Right', 'Down'])
        else:
            return 'Up' 

class ModelBasedAgent:
    """Step 1.3: Model-Based Agent (Maintains Internal State)"""
    def __init__(self):
        # Internal memory state tracking relative movements
        self.visited_states = set()
        self.relative_x = 0
        self.relative_y = 0
        self.last_action = 'Up'

    def sense_and_act(self, percept: dict) -> str:
        # 1. Update State (Transition/Sensor Model) based on last action taken
        if self.last_action == 'Up':
            self.relative_y += 1
        elif self.last_action == 'Down':
            self.relative_y -= 1
        elif self.last_action == 'Left':
            self.relative_x -= 1
        elif self.last_action == 'Right':
            self.relative_x += 1
            
        current_state = (self.relative_x, self.relative_y)
        self.visited_states.add(current_state)

        # 2. Condition-Action Rules querying the internal memory
        if percept['food_here']:
            action = 'Stay'
        elif percept['wall_ahead']:
            # Calculate coordinates of surrounding tiles
            possible_moves = {
                'Left': (self.relative_x - 1, self.relative_y), 
                'Right': (self.relative_x + 1, self.relative_y), 
                'Down': (self.relative_x, self.relative_y - 1)
            }
            
            # Filter the moves to prioritize unvisited tiles
            unvisited = [move for move, pos in possible_moves.items() if pos not in self.visited_states]
            
            if unvisited:
                action = random.choice(unvisited)
            else:
                action = random.choice(list(possible_moves.keys()))
        else:
            action = 'Up'
            
        self.last_action = action
        return action