import os
import openai
import time
import json
import logging
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
#  OpenAI Setup (Make sure your ENV vars are set)
# ---------------------------------------------------------------------
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_org = os.getenv('OPENAI_ORG')

openai.organization = openai_org
openai.api_key = openai_api_key

# ---------------------------------------------------------------------
#  Game Classes and Functions
# ---------------------------------------------------------------------
class Tower:
    def __init__(
        self,
        tower_id: int,
        x: int,
        y: int,
        tower_type: str = "Cannon",
        level: int = 1,
        range_: int = 2,
        damage: int = 10,
        cost: int = 100
    ):
        self.tower_id = tower_id
        self.x = x
        self.y = y
        self.tower_type = tower_type
        self.level = level
        self.range_ = range_
        self.damage = damage
        self.cost = cost

    def upgrade(self):
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
        # Sample 5x5 map
        # Using row-major indexing: map_layout[row][col]
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
        
        # Only 2 waves for this test
        self.waves = [
            [("Basic", 10, 1, 5) for _ in range(3)],  # wave 1
            [("Basic", 10, 1, 5) for _ in range(2)] + [("Fast", 8, 2, 5) for _ in range(2)],  # wave 2
        ]

    def print_map(self):
        """Simple console print to see towers / layout."""
        map_copy = [row[:] for row in self.map_layout]
        for _, tower in self.towers.items():
            map_copy[tower.x][tower.y] = 'T'
        
        print("Map Layout (P=Path, .=Empty, X=Blocked, T=Tower):")
        for row in map_copy:
            print(" ".join(row))
        print()


def run_wave(state: GameState, wave_enemies: List[Tuple[str, int, int, int]]):
    """Simulate the wave until all enemies are dead or reach the end."""
    enemies = [Enemy(e[0], e[1], e[2], e[3]) for e in wave_enemies]

    wave_ongoing = True
    while wave_ongoing:
        # Move enemies
        for enemy in enemies:
            enemy.path_position += enemy.speed
            if enemy.path_position >= len(state.path):
                state.health -= 1
                enemy.health = 0  # mark as "dead"

        # Towers shoot
        for _, tower in state.towers.items():
            for enemy in enemies:
                if enemy.health > 0:
                    # Check range (Manhattan distance)
                    if enemy.path_position < len(state.path):
                        ex, ey = state.path[enemy.path_position]
                        dist = abs(tower.x - ex) + abs(tower.y - ey)
                        if dist <= tower.range_:
                            enemy.health -= tower.damage
        
        # Remove dead enemies, add rewards
        alive = []
        for enemy in enemies:
            if enemy.health > 0:
                alive.append(enemy)
            else:
                state.gold += enemy.reward
        enemies = alive

        # Check conditions
        if state.health <= 0:
            print("Player health reached 0. Game Over!")
            break
        if not enemies:
            wave_ongoing = False


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
#  LLM Decision
# ---------------------------------------------------------------------
def get_llm_decision(state: GameState, next_wave: List[Tuple[str, int, int, int]]) -> str:
    """
    Calls the OpenAI Chat API to decide whether to:
     - BUILD
     - UPGRADE
     - DO_NOTHING
    Returns a JSON string that will be parsed by parse_llm_action.
    """
    logging.info("[get_llm_decision] init")
    
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

    # Summarize the wave
    enemy_summary = {}
    for (etype, ehealth, espeed, ereward) in next_wave:
        enemy_summary[etype] = enemy_summary.get(etype, 0) + 1

    # Tower info to show the LLM
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

    # Buildable positions
    valid_positions = get_valid_build_positions(state)

    # Provide a textual map layout (row by row), so the LLM sees what's blocked/path.
    # Also providing the valid build positions as an explicit list.
    rows = len(state.map_layout)
    cols = len(state.map_layout[0])
    textual_map = [
        "".join(state.map_layout[r][c] for c in range(cols)) 
        for r in range(rows)
    ]

    user_prompt = (
        f"Current wave number: {state.wave_number}\n"
        f"Health: {state.health}\n"
        f"Gold: {state.gold}\n"
        f"Towers: {towers_info}\n"
        f"Upcoming wave enemy counts: {enemy_summary}\n\n"
        f"Map layout (each string is a row, with 'P' for path, '.' for empty, 'X' for blocked):\n"
        f"{textual_map}\n\n"
        f"Valid build positions (row,col): {valid_positions}\n\n"
        "Decide ONE action:\n"
        "- DO_NOTHING\n"
        "- BUILD <tower_type> at [row,col] (must be in valid build positions)\n"
        "- UPGRADE <tower_id>\n\n"
        "Return your decision ONLY as valid JSON."
    )

    # A short pause to reduce rate-limit issues.
    time.sleep(1)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        llm_output = response.choices[0].message.content.strip()
        logging.info(f"[get_llm_decision] Raw LLM output: {llm_output}")
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        # Return a fallback JSON
        llm_output = '{"action": "DO_NOTHING"}'
    
    return llm_output


def parse_llm_action(state: GameState, llm_raw_json: str):
    """
    Parses the LLM's JSON response for the action.
    It should be one of:
        {"action": "DO_NOTHING"}
        {"action": "BUILD", "tower_type": "Cannon", "position": [3,2]}
        {"action": "UPGRADE", "tower_id": 2}
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
        if len(pos) == 2:
            bx, by = pos
            if is_buildable(state, bx, by) and state.gold >= 100:
                new_tower = Tower(
                    tower_id=state.next_tower_id,
                    x=bx,
                    y=by,
                    tower_type=tower_type,
                    level=1,
                    range_=2,
                    damage=10,
                    cost=100
                )
                state.towers[state.next_tower_id] = new_tower
                state.next_tower_id += 1
                state.gold -= 100
                print(f"Built a {tower_type} tower at {pos}. Gold left: {state.gold}")
            else:
                print("LLM tried BUILD, but invalid position or insufficient gold.")
        else:
            print("LLM gave invalid 'position' format. Doing nothing.")
    
    elif action == "UPGRADE":
        tower_id = data.get("tower_id")
        if not tower_id or tower_id not in state.towers:
            print("LLM tried UPGRADE with invalid tower_id.")
            return
        upgrade_cost = 50
        if state.gold >= upgrade_cost:
            t = state.towers[tower_id]
            t.upgrade()
            state.gold -= upgrade_cost
            print(f"Upgraded Tower #{tower_id} to level {t.level}. Gold left: {state.gold}")
        else:
            print("LLM chose UPGRADE, but insufficient gold.")


def is_buildable(state: GameState, x: int, y: int) -> bool:
    """Check if (x,y) is within bounds, is '.' (empty) on the map, and has no existing tower."""
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
    # Create the game and log some info
    state = GameState()
    logging.info("Starting Tower Defense Test for 2 waves.")
    
    print("Starting Tower Defense Benchmark (2 Waves)!")
    state.print_map()
    
    while state.wave_number < len(state.waves) and state.health > 0:
        state.wave_number += 1
        current_wave = state.waves[state.wave_number - 1]
        
        # -- Build/Upgrade Phase --
        print(f"== Wave #{state.wave_number} (Preparation) ==")
        llm_json = get_llm_decision(state, current_wave)
        parse_llm_action(state, llm_json)
        
        # -- Combat Phase --
        print(f"\n== Wave #{state.wave_number} (Combat) ==")
        run_wave(state, current_wave)
        
        if state.health <= 0:
            break
    
    # Final outcome
    if state.health > 0:
        print(f"All waves cleared! Final Health: {state.health}, Final Gold: {state.gold}")
        logging.info(f"All waves cleared! Health={state.health}, Gold={state.gold}")
    else:
        print(f"You survived until wave {state.wave_number}, but then lost. Health={state.health}")
        logging.info(f"Game Over at wave {state.wave_number}. Health={state.health}")

if __name__ == "__main__":
    main()
