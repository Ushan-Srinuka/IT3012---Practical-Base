import random
from collections import deque
import heapq

class SimpleReflexAgent:
    """Step 1.2 (Lab 2): Simple Reflex Agent (No Memory)"""
    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Stay' 
        elif percept['wall_ahead']:
            return random.choice(['Left', 'Right', 'Down'])
        else:
            return 'Up' 

class ModelBasedAgent:
    """Step 1.3 (Lab 2): Model-Based Agent (Maintains Internal State)"""
    def __init__(self):
        self.visited_states = set()
        self.relative_x = 0
        self.relative_y = 0
        self.last_action = 'Up'

    def sense_and_act(self, percept: dict) -> str:
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

        if percept['food_here']:
            action = 'Stay'
        elif percept['wall_ahead']:
            possible_moves = {
                'Left': (self.relative_x - 1, self.relative_y), 
                'Right': (self.relative_x + 1, self.relative_y), 
                'Down': (self.relative_x, self.relative_y - 1)
            }
            unvisited = [move for move, pos in possible_moves.items() if pos not in self.visited_states]
            
            if unvisited:
                action = random.choice(unvisited)
            else:
                action = random.choice(list(possible_moves.keys()))
        else:
            action = 'Up'
            
        self.last_action = action
        return action

class SearchAgent:
    """Step 1.2 & 1.3 (Lab 3): Uninformed Search Agent[cite: 8]"""
    def __init__(self):
        self.plan = []
        # Observation Task: Change this to 'DFS' or 'UCS' to see different paths![cite: 8]
        self.active_algo = 'BFS' 

    def _get_successors(self, state, grid_size, walls):
        """Helper function to find valid adjacent moves."""
        x, y = state
        w, h = grid_size
        successors = []
        if y + 1 < h and (x, y + 1) not in walls: successors.append(('Up', (x, y + 1)))
        if y - 1 >= 0 and (x, y - 1) not in walls: successors.append(('Down', (x, y - 1)))
        if x - 1 >= 0 and (x - 1, y) not in walls: successors.append(('Left', (x - 1, y)))
        if x + 1 < w and (x + 1, y) not in walls: successors.append(('Right', (x + 1, y)))
        return successors

    def bfs_search(self, start, goal, walls, grid_size):
        """BFS uses a FIFO queue (deque.popleft())[cite: 8]"""
        frontier = deque([(start, [])])
        reached = set([start]) # Reached set prevents infinite loops[cite: 8]
        
        while frontier:
            current_state, path = frontier.popleft()
            
            if current_state == goal:
                return path
                
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                if child_state not in reached:
                    reached.add(child_state)
                    frontier.append((child_state, path + [action]))
        return []

    def dfs_search(self, start, goal, walls, grid_size):
        """DFS uses a LIFO stack (list.pop())[cite: 8]"""
        frontier = [(start, [])] 
        reached = set([start])
        
        while frontier:
            current_state, path = frontier.pop()
            
            if current_state == goal:
                return path
                
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                if child_state not in reached:
                    reached.add(child_state)
                    frontier.append((child_state, path + [action]))
        return []

    def ucs_search(self, start, goal, walls, grid_size):
        """UCS uses a Priority Queue (heapq.heappop()) ordered by path cost[cite: 8]"""
        frontier = []
        heapq.heappush(frontier, (0, start, [])) 
        reached = {start: 0} 
        
        while frontier:
            cost, current_state, path = heapq.heappop(frontier)
            
            if current_state == goal:
                return path
                
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                new_cost = cost + 1
                if child_state not in reached or new_cost < reached[child_state]:
                    reached[child_state] = new_cost
                    heapq.heappush(frontier, (new_cost, child_state, path + [action]))
        return []

    def sense_and_act(self, percept: dict) -> str:
        """Step 1.3: Formulating the Plan[cite: 8]"""
        # Check if self.plan is empty[cite: 8]
        if not self.plan:
            if not percept['all_food']:
                return 'Stay'
            
            start_pos = tuple(percept['agent_pos'])
            
            # Find the closest food pellet from percept['all_food'][cite: 8]
            closest_food = min(percept['all_food'], key=lambda f: abs(f[0] - start_pos[0]) + abs(f[1] - start_pos[1]))
            goal_pos = tuple(closest_food)
            
            walls = set(tuple(w) for w in percept['walls'])
            grid_size = percept['grid_size']
            
            # Execute the search method matching self.active_algo and store in self.plan[cite: 8]
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start_pos, goal_pos, walls, grid_size)
                
        # Return the first action from the plan using return self.plan.pop(0)[cite: 8]
        if self.plan:
            return self.plan.pop(0)
        else:
            return 'Stay'