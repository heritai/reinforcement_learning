# RL Project: Real-Time Strategy

## Project Overview

This project focuses on applying Reinforcement Learning techniques to a Real-Time Strategy (RTS) game, similar to Warcraft or Starcraft. The chosen environment for this project is DeepRTS, a stable but still evolving module with limited documentation.

## 1. Deep-RTS Module

To clone the necessary GitHub repository, use the following commands:

```bash
git clone https://github.com/UIA-CAIR/DeepRTS.git --recurse-submodules
```

Important: Omitting `--recurse-submodules` will cause the module to fail.

### 1.1 Game Description

The module offers a simplified RTS environment with the following key components:

*   **Resources:** Two resource types:
    *   `gold`:  Gold.
    *   `lumber`: Lumber (wood).

    These resources are located in specific tiles on the game map.

*   **Terrain:**
    *   `grass`: Passable terrain.
    *   `wall`: Impassable terrain.
    *   `water`: Impassable, but units can shoot over it.

*   **Units and Buildings:**
    *   `townhall`: The base building that produces `peasant` units.
    *   `peasant`: A worker unit capable of:
        *   `harvesting`: Collecting resources (gold or lumber). Moves to a resource tile, spends time collecting, and returns to the `townhall` to deposit the resources.
        *   `combat`: Attacking enemy units or structures.
        *   `building`: Constructing buildings.
        *   `walking`: Moving to a specific location.

    *   `footman`: Melee infantry unit.
    *   `archer`: Ranged unit (fragile).
    *   `farm`: A building that provides 4 units of `food`.
    *   `barrack`: A building that can produce `footman` and `archer` units.

Each unit has a production cost in `gold` and `lumber`, and consumes `food` while alive.

The core gameplay loop in DeepRTS involves:

*   Producing `peasant` units to harvest resources.
*   Constructing `farm` buildings to increase the food cap (unit limit).
*   Constructing `barrack` buildings to produce combat units.
*   Attacking the opponent while maintaining a healthy economy.

The game ends when one player loses all their units and cannot produce any more.

Here's a table describing the attributes of different units and buildings (in order: health, armor, damage, range, speed, lumber cost, gold cost, food cost, food production):

| Unit/Building | Health | Armor | Damage | Range | Speed | Lumber | Gold | Food | Food Production |
|---|---|---|---|---|---|---|---|---|---|
| Peasant   | 30   | 0   | 2-9 (2) | 1  | 10   | 0   | 400   | -1  |   |
| Footman   | 60   | 4   | 2-9 (4) | 1  | 30   | 0   | 600   | -1  |   |
| Archer    | 40   | 0   | 3-9 (2) | 4  | 10   | 50  | 500   | -1  |   |
| TownHall  | 1200 | 0   |   |   |   | 250  | 500   | +1  |   |
| Farm      | 400  | 20   |   |   |   | 250  | 500   |   | +4 |
| Barrack   | 800  | 20   |   |   |   | 450  | 700   |   |   |

### 1.2 Module Usage

The `Engine.Config()` object allows you to configure the game: enable/disable units, set auto-harvesting for peasants, adjust starting resources, etc.  Here's a commented example:

```python
from DeepRTS import Engine

engine_config = Engine.Config()  # Create the configuration
engine_config.set_archer(True)    # Allow archers
engine_config.set_barracks(True)  # Allow barracks
engine_config.set_farm(True)      # Allow farms
engine_config.set_footman(True)   # Allow footmen
engine_config.set_auto_attack(False) # Disable auto-attack
engine_config.set_food_limit(1000)   # Set food limit
engine_config.set_harvest_forever(False) # Disable auto-harvest
engine_config.set_instant_building(False)  # Disable building latency
engine_config.set_pomdp(False)   # Disable fog of war
engine_config.set_console_caption_enabled(False) # Hide console info
engine_config.set_start_lumber(500)  # Set starting lumber
engine_config.set_start_gold(500)    # Set starting gold
engine_config.set_instant_town_hall(False) # Disable town hall latency
engine_config.set_terminal_signal(True) # Trigger end-of-game signal
```

The project encourages using *exactly* this configuration. Specifically, peasants should *not* auto-harvest, and building and unit production should be *instant*.

The graphical interface can also be configured:

```python
from DeepRTS import python

gui_config = python.Config(
    render=True,         # Enable the GUI
    view=True,
    inputs=True,        # Allow human player interaction
    caption=True,
    unit_health=True,
    unit_outline=True,
    unit_animation=True,
    audio=False
)
```

When `inputs` is set to `True` (note the `input` attribute in `gui_config`), you can interact with the game using the mouse: left-click to select a unit, right-click to designate a destination (attack if enemy, harvest if resource and peasant). With a building selected, the `1` key produces the first unit type (peasant for a `townhall`, `footman` for a `barrack`), the `2` key produces the second unit type (`archer` for a `barrack`). If a peasant is selected, `1` builds a `townhall`, `2` a `farm`, and `3` a `barrack`.

The `python.Config.Map` object contains available maps. The following code initializes the game and starts a match:

```python
MAP = python.Config.Map.TWENTYONE
game = python.Game(MAP, n_players=2, engine_config=engine_config, gui_config=gui_config)
game.set_max_fps(250)  # Increase FPS (if not visualizing)
game.set_max_ups(250)

game.reset()
while not game.is_terminal():
    game.update()
```

### 1.3 Interfacing

The `game` object offers several methods for interacting with the game.

*   **Game State:**  Use `game.get_state()` to obtain the game state.  This returns a tensor of size `X x Y x 10`, where `X` and `Y` are the map dimensions. The last dimension provides the following information, indexed from 0:

    *   `0`: Tile type: `2`: grass, `3`: wall, `4`: lumber, `5`: water, `6`: gold.
    *   `1`: Player ID of the unit/building on the tile.
    *   `2`:  1 if a building is present.
    *   `3`:  1 if a unit is present.
    *   `4`: Unit/building type: `1`: peasant, `3`: townhall, `4`: barracks, `5`: footman, `6`: farm, `7`: archer.
    *   `5`: Percentage of health remaining.
    *   `6`: Unit state: `1`: spawn, `2`: walk, `3`: despawn, `4`: harvesting, `5`: building, `6`: combat, `7`: dead, `8`: idle.
    *   `7`: Resource carried by a peasant (if applicable).
    *   `8`: Attack score for the unit.
    *   `9`: Defense score for the unit.

    *Note:* The player ID (index 1) is tricky. `0` indicates both player 0 and an empty tile. Consider using `state[:,:,1]+state[:,:,2]+state[:,:,3]` to get `1` for player 0, `2` for player 1, and `0` if the tile is empty.

*   **Action Transmission:** The list of players is in `game.players`. Thus, `game.players[0]` gives you the first player. There are two ways to pass an action to the game:

    *   `p.do_action(idAction)`:  `idAction` is an integer from 1 to 16 (inclusive), corresponding to the action list below:

        *   `1`: PreviousUnit (select previous unit)
        *   `2`: NextUnit (select next unit)
        *   `3`: MoveLeft
        *   `4`: MoveRight
        *   `5`: MoveUp
        *   `6`: MoveDown
        *   `7`: MoveUpLeft
        *   `8`: MoveUpRight
        *   `9`: MoveDownLeft
        *   `10`: MoveDownRight
        *   `11`: Attack
        *   `12`: Harvest
        *   `13`: Build0 (construct/produce from key 1)
        *   `14`: Build1 (construct/produce from key 2)
        *   `15`: Build2 (construct/produce from key 3)
        *   `16`: NoAction

        Commands 1 and 2 change the selected unit. Commands 3-15 apply to the selected unit.

    *   `p.do_manual_action(idAction, x, y)`: Mimics a human player using the mouse. `idAction` can be:
        *   `0`: NoAction
        *   `1`: Left click on tile (x, y)
        *   `2`: Right click on tile (x, y)

*   **Player Attributes:** The `Player` object contains:

    *   `food`: Food available.
    *   `food_consumption`: Food being consumed.
    *   `gold`: Gold available.
    *   `lumber`: Lumber available.
    *   `num_archer`, `num_barrack`, `num_farm`, `num_footman`, `num_peasant`, `num_town_hall`: Counts of each unit/building type.
    *   `statistic_damage_done`, `statistic_damage_taken`, `statistic_gathered_gold`, `statistic_gathered_lumber`, `statistic_units_created`: Statistics since the start of the game.
    *   `get_targeted_unit()`: Returns the selected unit.
    *   `set_targeted_unit_id(id)`: Sets the selected unit.

*   **Game Attributes:** The `Game` object contains:

    *   `get_height()`: Returns map height.
    *   `get_width()`: Returns map width.
    *   `units`: List of units in the game.
    *   `tilemap`: List of tiles (less useful, most info is in `get_state()`).

*   **Unit Attributes:** The `Unit` object contains:

    *   `type`: Unit type.
    *   `get_player()`: Returns the unit's player.
    *   `gold_carry`, `lumber_carry`: Resources carried by a peasant.
    *   `health`: Unit's health points.

More attributes and methods exist but are less crucial. Constants used in the game are defined in `src/Constants.h`, and unit stats are in `src/unit/UnitManager.cpp`.

## 2. Project Tasks

You are expected to define one or more tasks within the game and solve them using RL algorithms covered during the semester. Some example tasks include:

*   Harvesting resources as quickly as possible within a time limit.
*   Producing a certain number of military units as quickly as possible.
*   Reaching a target population within a limited time.

You must also define the reward function according to the task. Auxiliary functions and higher-level actions beyond the base game actions are encouraged.

Consider whether a global AI or an AI for each unit is more appropriate for the chosen task.

Your grade will be based on the relevance of the chosen solution and the appropriateness of the RL algorithms used for the defined tasks.

**Deliverables:**

*   A report detailing the tasks addressed, your algorithm's architecture, and the experiments conducted.

You are free to explore different approaches.

By disabling the GUI and increasing `fps` (frames per second) and `ups` (updates per second), you can accelerate the game when not visualizing it (using `game.set_max_fps(fps)` and `game.set_max_ups(ups)`).
