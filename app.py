import os
import openai
import time
import json
import logging
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------
#  Logging Setup
# ---------------------------------------------------------------------
log_directory = "./logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d')
log_filename = f"{log_directory}/log_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename, 
    filemode='a'  # 'a' for append, 'w' for overwrite
)

# ---------------------------------------------------------------------
#  OpenAI Setup
#  (Uses older `openai.chat.completions.create` as requested)
# ---------------------------------------------------------------------
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_org = os.getenv('OPENAI_ORG')

openai.organization = openai_org
openai.api_key = openai_api_key

# ---------------------------------------------------------------------
#  Tower & Enemy Metadata
# ---------------------------------------------------------------------
TOWER_TYPES = {
    "Cannon": {
        "cost": 100,
        "range": 2,
        "base_damage": 10,
        # Multipliers vs each enemy type: 
        # e.g. Cannon does half damage vs Armored, can't hit Flying
        "damage_mods": {
            "Basic": 1.0,
            "Fast": 1.0,
            "Tank": 0.8,
            "Armored": 0.5,
            "Flying": 0.0  # can't hit flying
        }
    },
    "Archer": {
        "cost": 75,
        "range": 3,
        "base_damage": 7,
        # Archer can hit flying, does normal damage to armor:
        "damage_mods": {
            "Basic": 1.0,
            "Fast": 1.0,
            "Tank": 0.6,
            "Armored": 1.0,
            "Flying": 1.0
        }
    },
    "Laser": {
        "cost": 150,
        "range": 2,
        "base_damage": 15,
        # Laser is very effective vs Armored, but cannot target Flying:
        "damage_mods": {
            "Basic": 1.0,
            "Fast": 1.0,
            "Tank": 1.5,
            "Armored": 2.0,
            "Flying": 0.0
        }
    }
    # You can add more tower types here if desired.
}

# You can add or modify enemy types as you like:
# (enemy_type, health, speed, reward)
# We'll dynamically generate waves, but these are reference stats.
ENEMY_TYPE_STATS = {
    "Basic":   {"health": 15, "speed": 1, "reward": 5},
    "Fast":    {"health": 12, "speed": 2, "reward": 6},
    "Tank":    {"health": 40, "speed": 1, "reward": 10},
    "Armored": {"health": 25, "speed": 1, "reward": 7},
    "Flying":  {"health": 10, "speed": 3, "reward": 8}
}

# ---------------------------------------------------------------------
#  Game Classes
# ---------------------------------------------------------------------
class Tower:
    def __init__(self, tower_id: int, x: int, y: int, tower_type: str):
        self.tower_id = tower_id
        self.x = x
        self.y = y
        self.tower_type = tower_type
        self.level = 1

        # Pull stats from TOWER_TYPES:
        tower_data = TOWER_TYPES[tower_type]
        self.base_cost = tower_data["cost"]
        self.range_ = tower_data["range"]
        self.base_damage = tower_data["base_damage"]
        self.damage_mods = tower_data["damage_mods"]

        # Actual cost so that an upgrade can modify:
        self.current_cost = self.base_cost

    def upgrade(self):
        self.level += 1
        # Example upgrade approach:
        # - Increase base damage a bit each time
        # - Increase cost for next upgrade
        # - Possibly also extend range slightly
        self.base_damage += 3
        self.range_ += 0 if self.level % 2 == 0 else 1  # +1 range every other level
        self.current_cost += 50

    def get_damage_against(self, enemy_type: str) -> float:
        """
        Calculate how much damage this tower does against the given enemy type,
        applying the appropriate multipliers and level scaling.
        """
        multiplier = self.damage_mods.get(enemy_type, 1.0)
        # e.g. each level adds +3 to base, so total is base_damage * multiplier
        return self.base_damage * multiplier


class Enemy:
    def __init__(self, enemy_type: str, health: int, speed: int, reward: int):
        self.enemy_type = enemy_type
        self.health = health
        self.speed = speed
        self.reward = reward
        self.path_position = 0  # index along the path


class GameState:
    def __init__(self):
        # Sample 5x5 map
        self.map_layout = [
            ['P','P','P','P','P'],
            ['.','.','.','.','P'],
            ['X','X','X','.','P'],
            ['.','.','.','.','P'],
            ['P','P','P','P','P'],
        ]
        # Path from top-left (0,0) to bottom-left (4,0) in a snake-like route:
        self.path = [
            (0,0), (0,1), (0,2), (0,3), (0,4),
            (1,4), (2,4), (3,4), (4,4),
            (4,3), (4,2), (4,1), (4,0)
        ]
        
        self.towers: Dict[int, Tower] = {}
        self.next_tower_id = 1

        self.gold = 300
        self.health = 20
        self.wave_number = 0
        
        # Wave data assigned later by generate_waves()
        self.waves: List[List[Enemy]] = []

        # This dictionary will track how many enemies of each type got through
        # in the *previous* wave. Use this to inform the LLM about weaknesses.
        self.last_wave_leaks: Dict[str, int] = {}

    def print_map(self):
        """Simple console print to see towers / layout."""
        map_copy = [row[:] for row in self.map_layout]
        for _, tower in self.towers.items():
            map_copy[tower.x][tower.y] = 'T'
        
        print("Map Layout (P=Path, .=Empty, X=Blocked, T=Tower):")
        for row in map_copy:
            print(" ".join(row))
        print()

def print_and_log_map(state: GameState):
    """Print and log the current map state."""
    map_copy = [row[:] for row in state.map_layout]
    for _, tower in state.towers.items():
        map_copy[tower.x][tower.y] = 'T'
    
    map_str = "\nMap Layout (P=Path, .=Empty, X=Blocked, T=Tower):\n"
    for row in map_copy:
        row_str = " ".join(row)
        map_str += row_str + "\n"
    
    print(map_str)
    logging.info(map_str)

# ---------------------------------------------------------------------
#  Steep Wave Generation Example
# ---------------------------------------------------------------------
def generate_waves(num_waves: int) -> List[List[Tuple[str, int, int, int]]]:
    """
    Generate a list of waves with various enemy types,
    each wave is a list of (enemy_type, health, speed, reward).
    We'll escalate difficulty more quickly.
    """
    all_waves = []
    for i in range(num_waves):
        wave_enemies = []
        
        # Basic enemies scale
        basic_count = 3 + (i * 2)
        basic_health = ENEMY_TYPE_STATS["Basic"]["health"] + i * 4
        basic_speed  = ENEMY_TYPE_STATS["Basic"]["speed"] + (i // 3)
        basic_reward = ENEMY_TYPE_STATS["Basic"]["reward"] + (i // 3)
        for _ in range(basic_count):
            wave_enemies.append(("Basic", basic_health, basic_speed, basic_reward))
        
        # Fast enemies (start from wave 1)
        if i >= 1:
            fast_count  = 1 + (i // 2)
            fast_health = ENEMY_TYPE_STATS["Fast"]["health"] + i * 2
            fast_speed  = ENEMY_TYPE_STATS["Fast"]["speed"] + (i // 4)
            fast_reward = ENEMY_TYPE_STATS["Fast"]["reward"] + (i // 2)
            for _ in range(fast_count):
                wave_enemies.append(("Fast", fast_health, fast_speed, fast_reward))

        # Tank enemies (start from wave 2)
        if i >= 2:
            tank_count  = 1 + (i // 3)
            tank_health = ENEMY_TYPE_STATS["Tank"]["health"] + i * 8
            tank_speed  = ENEMY_TYPE_STATS["Tank"]["speed"] + (i // 5)
            tank_reward = ENEMY_TYPE_STATS["Tank"]["reward"] + (i // 3)
            for _ in range(tank_count):
                wave_enemies.append(("Tank", tank_health, tank_speed, tank_reward))

        # Armored enemies (start from wave 3)
        if i >= 3:
            armored_count  = 1 + (i // 3)
            armored_health = ENEMY_TYPE_STATS["Armored"]["health"] + i * 5
            armored_speed  = ENEMY_TYPE_STATS["Armored"]["speed"] + (i // 6)
            armored_reward = ENEMY_TYPE_STATS["Armored"]["reward"] + (i // 3)
            for _ in range(armored_count):
                wave_enemies.append(("Armored", armored_health, armored_speed, armored_reward))

        # Flying enemies (start from wave 4)
        if i >= 4:
            flying_count  = 1 + (i // 4)
            flying_health = ENEMY_TYPE_STATS["Flying"]["health"] + i * 2
            flying_speed  = ENEMY_TYPE_STATS["Flying"]["speed"] + (i // 6)
            flying_reward = ENEMY_TYPE_STATS["Flying"]["reward"] + (i // 3)
            for _ in range(flying_count):
                wave_enemies.append(("Flying", flying_health, flying_speed, flying_reward))

        all_waves.append(wave_enemies)

    return all_waves


# ---------------------------------------------------------------------
#  Core Wave Execution
# ---------------------------------------------------------------------
def run_wave(state: GameState, wave_enemies: List[Tuple[str, int, int, int]]):
    """Simulate the wave. Also track how many of each type made it through."""
    logging.info(f"\n{'='*50}\n[Wave {state.wave_number}] Combat Phase\n{'='*50}")

    enemies = [Enemy(e[0], e[1], e[2], e[3]) for e in wave_enemies]

    initial_enemy_count = len(enemies)
    logging.info(f"Wave {state.wave_number} starting with {initial_enemy_count} enemies")
    logging.info(f"Player state - Health: {state.health}, Gold: {state.gold}")

    # Track how many of each type leak through
    leaks_this_wave: Dict[str, int] = {}

    wave_ongoing = True
    while wave_ongoing:
        # Move enemies
        for enemy in enemies:
            enemy.path_position += enemy.speed
            if enemy.path_position >= len(state.path):
                state.health -= 1
                # Count this leak
                leaks_this_wave[enemy.enemy_type] = leaks_this_wave.get(enemy.enemy_type, 0) + 1
                enemy.health = 0  # mark as "dead"

        # Towers shoot
        for _, tower in state.towers.items():
            for enemy in enemies:
                if enemy.health > 0 and enemy.path_position < len(state.path):
                    dist = abs(tower.x - state.path[enemy.path_position][0]) + \
                           abs(tower.y - state.path[enemy.path_position][1])
                    if dist <= tower.range_:
                        # Tower damage depends on enemy type
                        damage = tower.get_damage_against(enemy.enemy_type)
                        enemy.health -= damage
        
        # Remove dead enemies, add rewards
        alive = []
        for enemy in enemies:
            if enemy.health > 0:
                alive.append(enemy)
            else:
                # If not counted as leak, we get a reward
                if enemy.path_position < len(state.path):
                    state.gold += enemy.reward
        enemies = alive

        if state.health <= 0:
            print("Player health reached 0. Game Over!")
            break
        if not enemies:
            wave_ongoing = False

    logging.info(f"Wave {state.wave_number} completed")
    logging.info(f"Enemies that leaked through: {leaks_this_wave}")
    logging.info(f"Final state - Health: {state.health}, Gold: {state.gold}")
    
    # Log the map state after the wave
    print_and_log_map(state)

    # Store the leak data in the state so the LLM can see it next wave
    state.last_wave_leaks = leaks_this_wave


def get_valid_build_positions(state: GameState) -> List[Tuple[int, int]]:
    """
    Return a list of (row, col) coordinates where the map cell is '.' 
    and there is no tower on that cell.
    """
    valid_positions = []
    rows = len(state.map_layout)
    cols = len(state.map_layout[0])

    for r in range(rows):
        for c in range(cols):
            if state.map_layout[r][c] == '.':
                # Ensure no existing tower occupies this cell
                if not any(t.x == r and t.y == c for t in state.towers.values()):
                    valid_positions.append((r, c))
    return valid_positions

# ---------------------------------------------------------------------
#  LLM Decision (keeping older style usage)
# ---------------------------------------------------------------------
def get_llm_decision(state: GameState,
                     next_wave: List[Tuple[str, int, int, int]],
                     model_name: str) -> str:
    """
    Calls the OpenAI Chat API to decide:
       - DO_NOTHING
       - BUILD tower (Cannon/Archer/Laser) at [row,col]
       - UPGRADE tower_id
    We provide the full state, including:
      - last wave leaks by enemy type
      - multiple tower types
    """
    logging.info(f"\n{'='*50}\n[Wave {state.wave_number}] LLM Decision Request\n{'='*50}")
    
    system_prompt = (
        "You are a Tower Defense decision-making AI. "
        "Your goal is to make an optimal tower-building or upgrading decision. "
        "Return only valid JSON (no markdown fences). "
        "The JSON must match exactly one of these forms:\n\n"
        "1) {\"action\": \"DO_NOTHING\"}\n"
        "2) {\"action\": \"BUILD\", \"tower_type\": \"Cannon\", \"position\": [row,col]}\n"
        "3) {\"action\": \"UPGRADE\", \"tower_id\": 2}\n"
        "\nDo not include extra keys or commentary."
    )

    # Summarize next wave
    enemy_summary = {}
    for (etype, ehealth, espeed, ereward) in next_wave:
        enemy_summary[etype] = enemy_summary.get(etype, 0) + 1

    # Tower info
    towers_info = []
    for t in state.towers.values():
        towers_info.append({
            "tower_id": t.tower_id,
            "type": t.tower_type,
            "level": t.level,
            "range": t.range_,
            "base_damage": t.base_damage,
            "position": [t.x, t.y]
        })

    # Buildable positions
    valid_positions = get_valid_build_positions(state)

    # Map layout as strings
    rows = len(state.map_layout)
    cols = len(state.map_layout[0])
    textual_map = [
        "".join(state.map_layout[r][c] for c in range(cols)) 
        for r in range(rows)
    ]

    # If we had leaks from last wave, show them
    leak_info = state.last_wave_leaks if state.last_wave_leaks else {}

    # Also let the LLM know about available tower types (with cost, range, base_damage)
    tower_type_options = {}
    for ttype, data in TOWER_TYPES.items():
        tower_type_options[ttype] = {
            "cost": data["cost"],
            "range": data["range"],
            "base_damage": data["base_damage"]
        }

    user_prompt = (
        f"Current wave number: {state.wave_number}\n"
        f"Health: {state.health}\n"
        f"Gold: {state.gold}\n"
        f"Towers: {towers_info}\n\n"
        f"Available tower types (cost, range, base_damage): {tower_type_options}\n\n"
        f"Enemies that leaked last wave: {leak_info}\n"
        f"(These are the counts of each enemy type that made it through.)\n\n"
        f"Upcoming wave enemy counts: {enemy_summary}\n\n"
        f"Map layout rows (P=path, .=empty, X=blocked):\n{textual_map}\n\n"
        f"Valid build positions (row,col): {valid_positions}\n\n"
        "Decide ONE action:\n"
        "- DO_NOTHING\n"
        "- BUILD <tower_type> at [row,col]\n"
        "- UPGRADE <tower_id>\n\n"
        "Return your decision ONLY as valid JSON."
    )

    logging.info(f"System Prompt:\n{system_prompt}\n")
    logging.info(f"User Prompt:\n{user_prompt}\n")

    # Pause to avoid rate-limit issues
    time.sleep(1)
    
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        llm_output = response.choices[0].message.content.strip()
        logging.info(f"Parsed LLM Output:\n{llm_output}\n")

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        llm_output = '{"action": "DO_NOTHING"}'
    
    return llm_output


def parse_llm_action(state: GameState, llm_raw_json: str):
    """
    Parses the LLM's JSON response for the action.
    Expecting:
      - {"action": "DO_NOTHING"}
      - {"action": "BUILD", "tower_type": "Cannon", "position": [3,2]}
      - {"action": "UPGRADE", "tower_id": 2}
    """
    try:
        data = json.loads(llm_raw_json)
    except json.JSONDecodeError:
        print("Failed to parse LLM JSON. Doing nothing.")
        return
    
    action = data.get("action", "DO_NOTHING")
    
    if action == "DO_NOTHING":
        print("LLM chose DO_NOTHING.")
        return
    
    elif action == "BUILD":
        tower_type = data.get("tower_type", "Cannon")
        pos = data.get("position", [0, 0])
        if (tower_type not in TOWER_TYPES):
            print(f"LLM tried BUILD with unknown tower type '{tower_type}'. Ignoring.")
            return
        if len(pos) == 2:
            bx, by = pos
            tower_cost = TOWER_TYPES[tower_type]["cost"]
            if is_buildable(state, bx, by) and state.gold >= tower_cost:
                new_tower = Tower(
                    tower_id=state.next_tower_id,
                    x=bx,
                    y=by,
                    tower_type=tower_type
                )
                state.towers[state.next_tower_id] = new_tower
                state.next_tower_id += 1
                state.gold -= tower_cost
                print(f"Built a {tower_type} tower at {pos}. Gold left: {state.gold}")
            else:
                print("LLM tried BUILD, but invalid position or insufficient gold.")
        else:
            print("LLM gave invalid 'position' format. Doing nothing.")
    
    elif action == "UPGRADE":
        tower_id = data.get("tower_id")
        if not tower_id or (tower_id not in state.towers):
            print("LLM tried UPGRADE with invalid tower_id.")
            return
        # Let's say each upgrade costs 50 gold:
        upgrade_cost = 50
        if state.gold >= upgrade_cost:
            t = state.towers[tower_id]
            t.upgrade()
            state.gold -= upgrade_cost
            print(f"Upgraded Tower #{tower_id} (Type: {t.tower_type}) to Level {t.level}. Gold left: {state.gold}")
        else:
            print("LLM chose UPGRADE, but insufficient gold.")


def is_buildable(state: GameState, x: int, y: int) -> bool:
    """Check if (x,y) is within bounds, '.' on the map, and no tower there."""
    rows = len(state.map_layout)
    cols = len(state.map_layout[0])
    if not (0 <= x < rows and 0 <= y < cols):
        return False
    if state.map_layout[x][y] != '.':
        return False
    for t in state.towers.values():
        if t.x == x and t.y == y:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                        help="Which OpenAI model to use (e.g. gpt-3.5-turbo, gpt-4, etc.)")
    parser.add_argument("--waves", type=int, default=2,
                        help="Number of waves to run.")
    args = parser.parse_args()

    model_name = args.model
    num_waves = args.waves

    # Initialize game state
    state = GameState()

    # Create wave definitions (steeper difficulty)
    wave_data = generate_waves(num_waves)
    state.waves = wave_data

    logging.info(f"\n{'='*50}\nStarting Tower Defense Test\n{'='*50}")
    logging.info(f"Configuration - Waves: {num_waves}, Model: {model_name}")
    
    print(f"Starting Tower Defense Benchmark ({num_waves} Wave(s))!")
    state.print_map()

    # Run waves
    while state.wave_number < len(state.waves) and state.health > 0:
        state.wave_number += 1
        current_wave = state.waves[state.wave_number - 1]
        
        # Build/Upgrade Phase
        print(f"== Wave #{state.wave_number} (Preparation) ==")
        llm_json = get_llm_decision(state, current_wave, model_name)
        parse_llm_action(state, llm_json)
        
        # Combat Phase
        print(f"\n== Wave #{state.wave_number} (Combat) ==")
        run_wave(state, current_wave)
        
        if state.health <= 0:
            break

    logging.info(f"\n{'='*50}\nGame Complete\n{'='*50}")

    if state.health > 0:
        print(f"All waves cleared! Final Health: {state.health}, Final Gold: {state.gold}")
        logging.info(f"All waves cleared! Health={state.health}, Gold={state.gold}")
    else:
        print(f"You survived until wave {state.wave_number}, but then lost. Health={state.health}")
        logging.info(f"Game Over at wave {state.wave_number}. Health={state.health}")


if __name__ == "__main__":
    main()
