import random
from typing import List, Tuple, Dict

# -----------------------------
# 1. Game State Definitions
# -----------------------------

class Tower:
    def __init__(self, tower_id: int, x: int, y: int, tower_type: str = "Cannon",
                 level: int = 1, range_: int = 2, damage: int = 10, cost: int = 100):
        self.tower_id = tower_id
        self.x = x
        self.y = y
        self.tower_type = tower_type
        self.level = level
        self.range_ = range_
        self.damage = damage
        self.cost = cost

    def upgrade(self):
        # Example simplistic upgrade logic
        self.level += 1
        self.damage += 5
        self.range_ += 1
        self.cost += 50

class Enemy:
    def __init__(self, enemy_type: str, health: int, speed: int, reward: int):
        self.enemy_type = enemy_type
        self.health = health
        self.speed = speed
        self.reward = reward
        self.path_position = 0  # index along the path

class GameState:
    def __init__(self):
        # Example 5x5 map
        # 'P' = path, '.' = buildable land, 'X' = blocked
        self.map_layout = [
            ['P','P','P','P','P'],
            ['.','.','.','.','P'],
            ['X','X','X','.','P'],
            ['.','.','.','.','P'],
            ['P','P','P','P','P'],
        ]
        # Predefined path from top-left corner (0,0) to bottom-left corner (4,0)?
        # For simplicity, let's store a list of coordinates that define the path
        self.path = [(0,0), (0,1), (0,2), (0,3), (0,4),
                     (1,4), (2,4), (3,4), (4,4), (4,3), (4,2), (4,1), (4,0)]
        
        self.towers: Dict[int, Tower] = {}
        self.next_tower_id = 1

        self.gold = 300
        self.health = 20
        self.wave_number = 0
        
        # Sample wave definitions (each wave is a list of enemies)
        self.waves = [
            [("Basic", 10, 1, 5) for _ in range(5)],  # 5 Basic enemies
            [("Basic", 10, 1, 5) for _ in range(5)] + [("Fast", 8, 2, 5) for _ in range(3)],  # Mix
            [("Tank", 25, 1, 10) for _ in range(2)] + [("Basic", 10, 1, 5) for _ in range(3)],
        ]

    def print_map(self):
        """Optional: visualize map and towers (text-based)."""
        map_copy = [row[:] for row in self.map_layout]
        # Mark towers on map
        for _, tower in self.towers.items():
            map_copy[tower.x][tower.y] = 'T'
        
        print("Map Layout (P=Path, .=Empty, X=Blocked, T=Tower):")
        for row in map_copy:
            print(" ".join(row))
        print()

# -----------------------------
# 2. Running a Single Wave
# -----------------------------

def run_wave(state: GameState, wave_enemies: List[Tuple[str, int, int, int]]):
    """Simulate the wave from start to finish."""
    # Create Enemy objects
    enemies = [Enemy(enemy_type=e[0], health=e[1], speed=e[2], reward=e[3]) for e in wave_enemies]

    wave_ongoing = True
    while wave_ongoing:
        # Move enemies
        for enemy in enemies:
            # Move forward 'speed' steps along the path
            enemy.path_position += enemy.speed
            if enemy.path_position >= len(state.path):
                # Enemy reached end
                state.health -= 1  # or more damage if desired
                # Mark the enemy as "dead" by setting health <= 0
                enemy.health = 0
        
        # Towers shoot
        for _, tower in state.towers.items():
            for enemy in enemies:
                if enemy.health > 0:
                    # Check distance from tower to enemy
                    enemy_coord = state.path[enemy.path_position] if enemy.path_position < len(state.path) else None
                    if enemy_coord is not None:
                        dist = abs(tower.x - enemy_coord[0]) + abs(tower.y - enemy_coord[1])
                        if dist <= tower.range_:
                            # Shoot enemy
                            enemy.health -= tower.damage
        
        # Remove dead enemies, grant reward
        alive_enemies = []
        for enemy in enemies:
            if enemy.health > 0:
                alive_enemies.append(enemy)
            else:
                state.gold += enemy.reward
        enemies = alive_enemies
        
        # Check end condition
        if state.health <= 0:
            # Player lost
            print("Player health reached 0. Game Over.")
            return
        if len(enemies) == 0:
            # Wave cleared
            wave_ongoing = False

# -----------------------------
# 3. Interacting with the LLM
# -----------------------------

def get_llm_decision(state: GameState) -> str:
    """
    Here, you'd make a single call to the LLM, providing a concise summary:
    - wave_number
    - gold
    - health
    - existing towers (positions, levels, etc.)
    - upcoming wave info (enemy types, counts, etc.)
    
    For the sake of example, we'll just return a random action or a do-nothing action.
    """
    
    # In a real setup, you'd do something like:
    # prompt = create_prompt(state)
    # action = call_llm_api(prompt)
    # return action
    
    # For demonstration, let's randomly decide:
    possible_actions = ["DO_NOTHING", "BUILD", "UPGRADE"]
    chosen = random.choice(possible_actions)
    return chosen

def parse_llm_action(state: GameState, action: str):
    """
    Interpret the LLM's response. For example, if LLM says:
    "BUILD Cannon (3,2)" -> we parse tower type "Cannon", position (3,2), cost check, etc.
    "UPGRADE Tower 1" -> find Tower #1 and upgrade it if gold is sufficient.
    "DO_NOTHING" -> skip.
    
    For the sake of the example, we will do a simplified version:
    - If "BUILD", pick a random buildable spot and build a tower.
    - If "UPGRADE", pick a random existing tower and upgrade.
    """
    
    if action == "DO_NOTHING":
        return
    
    elif action == "BUILD":
        # find a random buildable spot
        buildable_spots = []
        for i in range(len(state.map_layout)):
            for j in range(len(state.map_layout[0])):
                if state.map_layout[i][j] == '.':  # buildable
                    # also check we don't already have a tower here
                    if not any(tower.x == i and tower.y == j for tower in state.towers.values()):
                        buildable_spots.append((i, j))
        
        if buildable_spots and state.gold >= 100:
            bx, by = random.choice(buildable_spots)
            new_tower = Tower(
                tower_id=state.next_tower_id,
                x=bx,
                y=by,
                tower_type="Cannon",
                level=1,
                range_=2,
                damage=10,
                cost=100
            )
            state.towers[state.next_tower_id] = new_tower
            state.next_tower_id += 1
            state.gold -= 100
            print(f"Built a tower at {(bx,by)}. Gold left: {state.gold}")
        else:
            print("LLM chose BUILD, but no available spots or insufficient gold.")
    
    elif action == "UPGRADE":
        if state.towers:
            tower_id_to_upgrade = random.choice(list(state.towers.keys()))
            tower_to_upgrade = state.towers[tower_id_to_upgrade]
            upgrade_cost = 50  # simplified
            if state.gold >= upgrade_cost:
                tower_to_upgrade.upgrade()
                state.gold -= upgrade_cost
                print(f"Upgraded Tower #{tower_id_to_upgrade} to level {tower_to_upgrade.level}. Gold left: {state.gold}")
            else:
                print("LLM chose UPGRADE, but insufficient gold.")


# -----------------------------
# 4. Main Game Loop
# -----------------------------

def main():
    state = GameState()
    
    print("Starting Tower Defense Benchmark!")
    state.print_map()
    
    while state.wave_number < len(state.waves) and state.health > 0:
        state.wave_number += 1
        current_wave = state.waves[state.wave_number - 1]
        
        # -- Build/Upgrade Phase (LLM Action) --
        print(f"\n== Wave #{state.wave_number} (Preparation) ==")
        llm_action = get_llm_decision(state)
        print(f"LLM decided: {llm_action}")
        parse_llm_action(state, llm_action)
        
        # -- Wave Simulation --
        print(f"\n== Wave #{state.wave_number} (Combat) ==")
        run_wave(state, current_wave)
        
        # If health <= 0, game over
        if state.health <= 0:
            break
    
    if state.health > 0:
        print(f"All waves cleared! Final health: {state.health}, Final gold: {state.gold}")
    else:
        print(f"You survived until wave {state.wave_number}, but then lost.")

if __name__ == "__main__":
    main()
