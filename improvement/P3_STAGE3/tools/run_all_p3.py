"""Run the stage-3 experiment locally (needs a CUDA GPU for practical speed; ~1.5-2 min per run on a T4 when run one at
a time). Completed runs (metrics.json present) are skipped, interrupted runs resume from their per-epoch checkpoint.
Usage: python tools/run_all_p3.py --seeds 0 1 2 [--arms fix aug fix_aug] [--targets 0 1 2 3 4] [--anchor]
--anchor first runs the control arm (ie_off) for target 0 seed 0, which must reproduce the stage-1 T4 result
(selected epoch 99, accuracy 74.06340057636888) exactly on a Tesla T4 with torch 2.14.1+cu130."""
import argparse, json, os, subprocess, sys, time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUN = os.path.join(PKG, 'code', 'stage3', 'run_stage3.py')
p = argparse.ArgumentParser()
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--arms', nargs='+', default=['fix', 'aug', 'fix_aug'])
p.add_argument('--targets', nargs='+', default=['0', '1', '2', '3', '4'])
p.add_argument('--anchor', action='store_true'); p.add_argument('--out_root', default='runs_p3')
a = p.parse_args()
jobs = ([('ie_off', '0', 0)] if a.anchor else []) + [(arm, t, s) for s in a.seeds for t in a.targets for arm in a.arms]
for arm, t, s in jobs:
    t0 = time.time()
    r = subprocess.run([sys.executable, RUN, '--arm', arm, '--target', t, '--seed', str(s), '--out_root', a.out_root,
                        '--no_last_ckpt'], capture_output=True, text=True)
    done = [l for l in r.stdout.splitlines() if l.startswith(('DONE', 'SKIP'))]
    print(f'{arm:8s} target {t} seed {s} exit {r.returncode} {time.time() - t0:5.0f}s | {done[-1].split(" -> ")[0] if done else r.stderr.strip()[-300:]}', flush=True)
if a.anchor:
    m = json.load(open(os.path.join(PKG, 'code', 'stage3', a.out_root, 'ucihar', 'ie_off', 'target0', 'seed0', 'metrics.json')))
    got = (m['selection']['selected_epoch'], m['target_at_selected_epoch']['accuracy'])
    print('ANCHOR', got, 'EXACT MATCH with stage 1 (T4):', got == (99, 74.06340057636888))
