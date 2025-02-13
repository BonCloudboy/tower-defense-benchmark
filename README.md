# LLM Tower Defense Benchmark

A tower defense game simulation designed to benchmark LLM (Large Language Model) decision-making capabilities. The project uses OpenAI's GPT models to make strategic decisions about tower placement and upgrades in a classic tower defense game setting.

## 🎮 Features

- **Dynamic Wave Generation**: Increasingly challenging waves of different enemy types
- **Multiple Tower Types**: 
  - Cannon: High damage, effective against basic units
  - Archer: Long range, can target flying units
  - Laser: Specialized against armored units
- **Enemy Variety**:
  - Basic: Standard units
  - Fast: Quick but fragile
  - Tank: High health, slow movement
  - Armored: Resistant to normal damage
  - Flying: Can only be hit by certain towers
- **Detailed Logging**: Comprehensive logging of game state, LLM decisions, and outcomes
- **Customizable Parameters**: Adjustable wave count and choice of OpenAI models

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages:
  ```
  openai
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BonCloudboy/tower-defense-benchmark.git
   cd tower-defense-benchmark
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   export OPENAI_ORG='your-org-id-here'  # if applicable
   ```

### Usage

Run the benchmark with default settings:
```bash
python tower_defense.py
```

Customize the run with command line arguments:
```bash
python tower_defense.py --model gpt-4o --waves 15
python tower_defense.py --model gpt-3.5-turbo --waves 15

```

Available arguments:
- `--model`: OpenAI model to use (default: gpt-3.5-turbo)
- `--waves`: Number of waves to run (default: 2)

## 🎯 Game Mechanics

### Map Layout
The game uses a 5x5 grid with the following elements:
- `P`: Path tiles that enemies follow
- `.`: Empty tiles where towers can be built
- `X`: Blocked tiles
- `T`: Placed towers

### Tower Types

| Tower Type | Cost | Range | Base Damage | Special Properties |
|------------|------|--------|-------------|-------------------|
| Cannon | 100 | 2 | 10 | Strong vs basic units, weak vs armored |
| Archer | 75 | 3 | 7 | Can target flying units |
| Laser | 150 | 2 | 15 | Very effective vs armored units |

### Enemy Types

| Enemy Type | Health | Speed | Reward | Special Properties |
|------------|--------|--------|---------|-------------------|
| Basic | 15 | 1 | 5 | Standard unit |
| Fast | 12 | 2 | 6 | Moves twice as fast |
| Tank | 40 | 1 | 10 | High health |
| Armored | 25 | 1 | 7 | Resistant to normal damage |
| Flying | 10 | 3 | 8 | Can only be hit by certain towers |

## 📊 Logging

The benchmark provides detailed logging of:
- LLM prompts and responses
- Game state after each wave
- Tower placement and upgrade decisions
- Enemy leaks and wave outcomes
- Final game results

Logs are stored in the `./logs` directory with timestamps.

## 🤖 LLM Decision Making

The LLM is provided with:
- Current game state (health, gold, wave number)
- Existing tower positions and levels
- Upcoming wave composition
- Previous wave performance
- Valid build positions

It must return one of three actions:
1. DO_NOTHING
2. BUILD a new tower
3. UPGRADE an existing tower

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔍 Future Improvements

- Add more tower types and enemy variants
- Implement different map layouts
- Add support for alternative LLM providers
- Create visualization tools for game playback
- Add multiplayer benchmark capabilities
- Implement different difficulty levels
