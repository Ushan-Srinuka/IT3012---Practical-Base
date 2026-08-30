import random
import math
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
    """Step 1.2 & 1.3 (Lab 3 & 4): Informed/Uninformed Search Agent"""
    def __init__(self):
        self.plan = []
        # Changed to AStar for Lab 04
        self.active_algo = 'AStar' 

    # --- Lab 04: Heuristic Functions[cite: 9] ---
    def manhattan_distance(self, pos, goal):
        """h(n) = |x_1 - x_2| + |y_1 - y_2|"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x_1-x_2)^2 + (y_1-y_2)^2)"""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def _get_successors(self, state, grid_size, walls):
        x, y = state
        w, h = grid_size
        successors = []
        if y + 1 < h and (x, y + 1) not in walls: successors.append(('Up', (x, y + 1)))
        if y - 1 >= 0 and (x, y - 1) not in walls: successors.append(('Down', (x, y - 1)))
        if x - 1 >= 0 and (x - 1, y) not in walls: successors.append(('Left', (x - 1, y)))
        if x + 1 < w and (x + 1, y) not in walls: successors.append(('Right', (x + 1, y)))
        return successors

    # --- Lab 04: A* Search[cite: 9] ---
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        frontier = []
        reached_states = set()
        
        g_cost = 0
        if heuristic_type == 'manhattan':
            h_cost = self.manhattan_distance(start_pos, goal_pos)
        else:
            h_cost = self.euclidean_distance(start_pos, goal_pos)
            
        f_cost = g_cost + h_cost
        
        # Priority queue tuple: (f_cost, g_cost, current_pos, path_taken)[cite: 9]
        heapq.heappush(frontier, (f_cost, g_cost, start_pos, []))
        
        while frontier:
            current_f, current_g, current_pos, path_taken = heapq.heappop(frontier)
            
            if current_pos == goal_pos:
                return path_taken
                
            if current_pos in reached_states:
                continue
                
            reached_states.add(current_pos)
            
            for action, child_state in self._get_successors(current_pos, grid_size, walls):
                if child_state not in reached_states:
                    new_g = current_g + 1
                    
                    if heuristic_type == 'manhattan':
                        new_h = self.manhattan_distance(child_state, goal_pos)
                    else:
                        new_h = self.euclidean_distance(child_state, goal_pos)
                        
                    new_f = new_g + new_h
                    heapq.heappush(frontier, (new_f, new_g, child_state, path_taken + [action]))
                    
        return []

    # --- Lab 03 Algorithms ---
    def bfs_search(self, start, goal, walls, grid_size):
        frontier = deque([(start, [])])
        reached = set([start]) 
        
        while frontier:
            current_state, path = frontier.popleft()
            if current_state == goal: return path
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                if child_state not in reached:
                    reached.add(child_state)
                    frontier.append((child_state, path + [action]))
        return []

    def dfs_search(self, start, goal, walls, grid_size):
        frontier = [(start, [])] 
        reached = set([start])
        
        while frontier:
            current_state, path = frontier.pop()
            if current_state == goal: return path
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                if child_state not in reached:
                    reached.add(child_state)
                    frontier.append((child_state, path + [action]))
        return []

    def ucs_search(self, start, goal, walls, grid_size):
        frontier = []
        heapq.heappush(frontier, (0, start, [])) 
        reached = {start: 0} 
        
        while frontier:
            cost, current_state, path = heapq.heappop(frontier)
            if current_state == goal: return path
            for action, child_state in self._get_successors(current_state, grid_size, walls):
                new_cost = cost + 1
                if child_state not in reached or new_cost < reached[child_state]:
                    reached[child_state] = new_cost
                    heapq.heappush(frontier, (new_cost, child_state, path + [action]))
        return []

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            if not percept['all_food']:
                return 'Stay'
            
            start_pos = tuple(percept['agent_pos'])
            
            # Lab 04: Use Manhattan distance to find the closest food[cite: 9]
            closest_food = min(percept['all_food'], key=lambda f: self.manhattan_distance(start_pos, f))
            goal_pos = tuple(closest_food)
            
            walls = set(tuple(w) for w in percept['walls'])
            grid_size = percept['grid_size']
            
            # Executing algorithms based on config[cite: 9]
            if self.active_algo == 'AStar':
                self.plan = self.astar_search(start_pos, goal_pos, walls, grid_size, 'manhattan')
            elif self.active_algo == 'BFS':
                self.plan = self.bfs_search(start_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start_pos, goal_pos, walls, grid_size)
                
        if self.plan:
            return self.plan.pop(0)
        else:
            return 'Stay'