# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xclayy/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xclayy/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): a program that reads Exophase's recorded trophy earn rates for Grand Theft Auto VI on PlayStation 5 up to a checkpoint and forecasts every trophy's earn rate at the target time, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## How it works

`main.py` (Python 3.13, standard library only, no network) works in logit space and splits each trophy's rate movement into two forces:

- **Dilution** — when Exophase refreshes its page, the tracked-player base jumps and the newcomers hold far fewer trophies, so almost every rate falls. A pooled dilution elasticity is estimated from the reads where the player count jumps, and the player count at the target is extrapolated.
- **Progression** — between refreshes the tracked players keep playing and earning. Each trophy gets its own recency-weighted, robust least-squares line of (dilution-corrected) logit rate against log time since the release day began, shrunk toward its tier's mean slope.

One regime needs a prior instead of a fit: a page that has never refreshed. On launch night Exophase's page sat frozen for hours and then refreshed in one jump (638 to 3,868 tracked players) while the newcomers had earned almost nothing beyond the trophy everybody gets, so ordinary rates fell to roughly a fifth. For a history with no player-count change at all, the forecast therefore uses a refresh mixture — most likely a refresh lands before the target and cuts non-saturated rates to about a fifth (logit shift about −1.5), with the remaining probability that the page stays as it is. A saturated trophy (essentially everybody holds it) is forecast with a tight band at the top.

The spread combines the fit's residual, slope uncertainty, dilution uncertainty, and a random-walk term that grows with the horizon. The program is deterministic: two runs on the same case write byte-identical output.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `/output/<id>.out` per case (default `/output`), each `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Practice results

Scored locally against simulated answers built from the kit's own published refresh totals (638 → 3,868 players, 943 → 4,276 trophies earned), across nine refresh scenarios (refresh factor 3×–10×, new-joiner earn rates 0.03–0.15 of incumbents, rarity gradients, uniform new-joiner rates):

- this program: pinball loss **0.16–0.26** (central scenario 0.16)
- house persistence baseline: **0.86–1.72** (central scenario 1.39; kit reports 1.51)

On a synthetic final-like case (308 reads over two days, player base growing ~3× to the target, per-trophy progression plus dilution): **0.17** vs baseline 1.47. Deterministic across runs; outputs validate (55 rows, unique ids, non-decreasing quantiles, ≤ 65,536 bytes) and run in ~0.1 s per case.

## License

MIT — see [LICENSE](LICENSE).
