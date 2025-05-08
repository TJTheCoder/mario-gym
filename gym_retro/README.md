We have a version of gym retro that (at least at one time) supported playing the 1k "stable" games (that have human-designed reward functions) and like ~14k extra games that do not.

I need to revisit the reward-func free setup. But we have more recent setups for train/test split over the 1k stable games. 

We play Retro across multiple consoles with the amago `RetroArcade` wrapper. It takes a `.json` of "game starts", which in Retro are {game : [levels]}. We made some train/test split jsons in
`game_splits/`.

You should be able to install the version of retro with all these roms with:
```bash
pip install -e . /mnt/nfs_client/jake/stable-retro
```

Then test that it worked with:

```bash
xvfb-run python test_env.py
```

I've also dumped some code used by a member of the last project for playing the Mario game with `dopamine`. 
