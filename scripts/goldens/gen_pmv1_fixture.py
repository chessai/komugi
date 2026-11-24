"""KOM-002 stage B (#3, #17): capture the pre-KOM-002 encoder's planes and the win_state
sequence by REPLAYING each golden game's chosen moves at main@29213f2.
Run inside a worktree checked out at 29213f2 (118-plane engine), default devshell:
  PYTHONPATH=$PWD/src python gen_pmv1_fixture.py <goldens_dir> <out_dir>
Per golden file -> <out_dir>/pmv1_<mode>.npz with, per game, per ply BEFORE the move:
  digests[g][p]  sha256 of observation().astype(float32).tobytes()  (exact, 32 B)
  win_state[g][p]  uint8[3] AFTER the move
  sample planes: full (118,9,9) float32 for ply 0 and the last ply of each game.
"""
import hashlib, json, os, sys, subprocess, numpy as np
from komugi.engine.api import Gungi
from komugi.engine.action_encoding import encode_move
from komugi.engine.game import Game

gold, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = {"generated_at_commit": sha, "command": " ".join(sys.argv), "files": {}}
for mode in ("intro", "beginner", "intermediate", "advanced"):
    data = json.load(open(os.path.join(gold, f"playouts_{mode}.json")))
    Game.set_start_mode(mode)
    digests, wins, samples, sample_keys, plies_per_game = [], [], [], [], []
    for gi, game in enumerate(data["games"]):
        g = Game(); gd, gw = [], []
        for pi, ply in enumerate(game):
            obs = np.ascontiguousarray(g.observation(), dtype=np.float32)
            assert obs.shape == (118, 9, 9), obs.shape
            gd.append(hashlib.sha256(obs.tobytes()).hexdigest())
            if pi == 0 or pi == len(game) - 1:
                samples.append(obs); sample_keys.append((gi, pi))
            legal = Gungi(ply["fen_before"]).moves({"verbose": True})
            san = ply["chosen"]; sn = san.rstrip("=#")
            found = [m for m in legal if m["san"] == sn or m["san"] == san]
            assert found, (mode, gi, pi, san)
            g.play_action(encode_move(found[0]))
            gw.append(np.asarray(g.win_state(), dtype=np.uint8))
        digests.append(gd); wins.append(np.stack(gw)); plies_per_game.append(len(game))
    flat_d = np.array([d for gd in digests for d in gd], dtype="U64")
    flat_w = np.concatenate(wins)
    np.savez_compressed(os.path.join(out, f"pmv1_{mode}.npz"), digests=flat_d, win_state=flat_w,
                        plies_per_game=np.array(plies_per_game), sample_planes=np.stack(samples),
                        sample_keys=np.array(sample_keys))
    manifest["files"][f"pmv1_{mode}.npz"] = {"games": len(digests), "plies": int(flat_d.size), "samples": len(samples)}
    print(mode, len(digests), "games", flat_d.size, "plies", flush=True)
json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=2)
print("PMV1_DONE", json.dumps(manifest["files"]))
