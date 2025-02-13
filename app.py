import random
from typing import List, Tuple, Dict
import openai

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

import json
import logging

def get_llm_decision(state: GameState, next_wave: List[Tuple[str, int, int, int]]) -> str:
    """
    Calls the OpenAI Chat API to decide whether to BUILD, UPGRADE, or DO_NOTHING,
    based on the current GameState and the next wave's enemy composition.
    
    Expects the LLM to return a JSON object in one of these forms:
    
    {
      "action": "DO_NOTHING"
    }
    or
    {
      "action": "BUILD",
      "tower_type": "<some tower type>",
      "position": [row, col]
    }
    or
    {
      "action": "UPGRADE",
      "tower_id": <tower_id_integer>
    }
    """
    
    logging.info("[get_llm_decision] init")
    
    # --- 1. Build the system message (instructions for the assistant) ---
    system_prompt = (
        "You are a Tower Defense decision-making AI. "
        "You receive the current game state and must decide ONE of three actions each wave: "
        "BUILD a new tower, UPGRADE an existing tower, or DO_NOTHING. "
        "Your response MUST be valid JSON with the structure described. "
        "Do not include any extra keys or commentary."
    )
    
    # --- 2. Summarize the relevant state for the LLM ---
    # Here, we provide the next wave info, the player's gold, health, tower info, etc.
    
    # Convert wave enemy data into a short summary (e.g. count how many of each type)
    enemy_summary = {}
    for e_type, e_health, e_speed, e_reward in next_wave:
        enemy_summary[e_type] = enemy_summary.get(e_type, 0) + 1
    
    towers_info = [
        {
            "tower_id": t.tower_id,
            "type": t.tower_type,
            "level": t.level,
            "range": t.range_,
            "damage": t.damage,
            "position": [t.x, t.y]
        }
        for t in state.towers.values()
    ]
    
    user_prompt = (
        f"Current wave number: {state.wave_number}\n"
        f"Health: {state.health}\n"
        f"Gold: {state.gold}\n"
        f"Towers: {towers_info}\n"
        f"Upcoming wave enemy counts: {enemy_summary}\n\n"
        "Decide ONE action:\n"
        "- DO_NOTHING\n"
        "- BUILD <tower_type> at [row,col]\n"
        "- UPGRADE <tower_id>\n\n"
        "Response format must be strictly JSON, with one of these forms:\n\n"
        "{{\"action\": \"DO_NOTHING\"}}\n\n"
        "{{\"action\": \"BUILD\", \"tower_type\": \"Cannon\", \"position\": [3,2]}}\n\n"
        "{{\"action\": \"UPGRADE\", \"tower_id\": 2}}\n"
    )
    
    # --- 3. Call the OpenAI Chat API ---
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # or whichever model is suitable
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        llm_output = response.choices[0].message.content.strip()
        logging.info(f"[get_llm_decision] Raw LLM output: {llm_output}")
        
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        # Fallback: do nothing if there's an API error
        return "DO_NOTHING"
    
    # --- 4. Parse the LLM's JSON response ---
    try:
        data = json.loads(llm_output)
        action = data.get("action", "DO_NOTHING")
        
        # Validate the action field
        if action not in ["DO_NOTHING", "BUILD", "UPGRADE"]:
            logging.warning(f"LLM returned unrecognized action: {action}. Defaulting to DO_NOTHING.")
            return "DO_NOTHING"
        
        # Return the entire JSON string so we can handle it outside
        # or just return the `data` dict. We'll return it as a string
        # so parse_llm_action can interpret it.
        return llm_output
    
    except json.JSONDecodeError as decode_err:
        logging.error(f"Failed to decode LLM JSON response: {decode_err}")
        # If JSON is invalid, we default to doing nothing.
        return "DO_NOTHING"


def parse_llm_action(state: GameState, llm_raw_json: str):
    """
    Interpret the JSON output from the LLM, which might look like:
      {"action": "DO_NOTHING"}
      {"action": "BUILD", "tower_type": "Cannon", "position": [3,2]}
      {"action": "UPGRADE", "tower_id": 2}
    """
    try:
        data = json.loads(llm_raw_json)
    except json.JSONDecodeError:
        # If there's a parsing error, do nothing
        print("Failed to parse LLM JSON. Doing nothing.")
        return
    
    action = data.get("action", "DO_NOTHING")
    
    if action == "DO_NOTHING":
        print("LLM chose DO_NOTHING.")
        return
    
    elif action == "BUILD":
        tower_type = data.get("tower_type", "Cannon")
        position = data.get("position", [0, 0])  # default fallback
        if len(position) == 2:
            bx, by = position
            # Check gold and if buildable, etc.
            # We can re-use the earlier logic or refine it
            if state.gold >= 100 and is_buildable(state, bx, by):
                new_tower = Tower(
                    tower_id=state.next_tower_id,
                    x=bx,
                    y=by,
                    tower_type=tower_type,  # use the LLM-provided type or "Cannon" by default
                    level=1,
                    range_=2,
                    damage=10,
                    cost=100
                )
                state.towers[state.next_tower_id] = new_tower
                state.next_tower_id += 1
                state.gold -= 100
                print(f"Built a {tower_type} tower at {(bx,by)}. Gold left: {state.gold}")
            else:
                print("LLM chose BUILD, but not feasible (insufficient gold or invalid position).")
        else:
            print("LLM provided invalid build position. Doing nothing.")
    
    elif action == "UPGRADE":
        tower_id = data.get("tower_id")
        if not tower_id or tower_id not in state.towers:
            print("LLM chose UPGRADE, but provided invalid tower_id.")
            return
        tower_to_upgrade = state.towers[tower_id]
        upgrade_cost = 50  # simplified
        if state.gold >= upgrade_cost:
            tower_to_upgrade.upgrade()
            state.gold -= upgrade_cost
            print(f"Upgraded Tower #{tower_id} to level {tower_to_upgrade.level}. Gold left: {state.gold}")
        else:
            print("LLM chose UPGRADE, but insufficient gold.")

def is_buildable(state: GameState, x: int, y: int) -> bool:
    """
    Returns True if (x,y) is within bounds, not a path, not blocked,
    and doesn't already have a tower.
    """
    if not (0 <= x < len(state.map_layout) and 0 <= y < len(state.map_layout[0])):
        return False
    if state.map_layout[x][y] != '.':  # Must be buildable spot
        return False
    # Check no existing tower
    for t in state.towers.values():
        if t.x == x and t.y == y:
            return False
    return True


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
        llm_json_response = get_llm_decision(state, current_wave)
        parse_llm_action(state, llm_json_response)
        
        # -- Wave Simulation --
        print(f"\n== Wave #{state.wave_number} (Combat) ==")
        run_wave(state, current_wave)
        
        if state.health <= 0:
            break
    
    if state.health > 0:
        print(f"All waves cleared! Final health: {state.health}, Final gold: {state.gold}")
    else:
        print(f"You survived until wave {state.wave_number}, but then lost.")


if __name__ == "__main__":
    main()
