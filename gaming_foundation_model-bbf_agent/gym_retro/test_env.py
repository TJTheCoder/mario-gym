# import retro

from amago.envs.builtin.ale_retro import RetroArcade

if __name__ == "__main__":
    import json

    with open("game_splits/s2/s2_train.json", "r") as f:
        gs_dict = json.load(f)
    env = RetroArcade(game_start_dict=gs_dict, use_discrete_actions=False)
    for _ in range(1000):
        env.reset()
        print(env.rom_name)
