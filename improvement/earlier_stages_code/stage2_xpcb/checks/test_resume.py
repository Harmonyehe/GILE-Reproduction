"""Transfer-package resume tests (fake data, CPU or GPU, a few minutes). Writes only to stage2_xpc/runs_resume_tests/.
  R1  package runner == original stage-2 runner (when the original is available next to the package) - optional
  R2  for ie_off, ie_off_bal, xpc_bal: run interrupted after epoch 2 and resumed  ==  uninterrupted run (history + metrics)
  R3  a checkpoint from a different environment is NOT resumed: it is kept (renamed) and the run restarts from epoch 1,
      giving the same result as an uninterrupted run
  R4  a completed run is skipped and its files are not modified
  R5  a checkpoint whose configuration differs is refused (non-zero exit) and left untouched
Usage: python stage2_xpc/checks/test_resume.py [--original <path to original improvement-code/stage2_xpc/run_stage2.py>]"""
import argparse, csv, glob, hashlib, json, os, shutil, subprocess, sys
import torch

S2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RUNNER = os.path.join(S2, 'run_stage2.py')
OUT = os.path.join(S2, 'runs_resume_tests')
ap = argparse.ArgumentParser(); ap.add_argument('--original', default=None); a = ap.parse_args()
shutil.rmtree(OUT, ignore_errors=True)            # test scratch folder only
ok = True
def check(name, cond, info=''):
    global ok
    ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)

EP = '4'
def run(variant, root, extra=(), runner=RUNNER):
    r = subprocess.run([sys.executable, runner, '--variant', variant, '--target', '1', '--seed', '2', '--epochs', EP, '--fake',
                        '--out_root', root, '--lam', '1000', '--no_last_ckpt', *extra], capture_output=True, text=True)
    return r
def d(root, v): return os.path.join(root, 'ucihar', v, 'target1', 'seed2')
def hist(root, v): return list(csv.DictReader(open(os.path.join(d(root, v), 'history.csv'))))
def met(root, v): return json.load(open(os.path.join(d(root, v), 'metrics.json')))
SKIPCOLS = {'epoch_seconds'}
def same_hist(h1, h2): return len(h1) == len(h2) and all({k: x[k] for k in x if k not in SKIPCOLS} == {k: y[k] for k in y if k not in SKIPCOLS} for x, y in zip(h1, h2))
def same_met(m1, m2):
    keys = ('selection', 'target_at_selected_epoch', 'target_at_last_epoch', 'probes_selected_epoch')
    return all(m1[k] == m2[k] for k in keys)

A = os.path.join(OUT, 'uninterrupted'); B = os.path.join(OUT, 'interrupted'); C = os.path.join(OUT, 'crossenv')
for v in ('ie_off', 'ie_off_bal', 'xpc_bal'):
    r1 = run(v, A)
    r2 = run(v, B, ('--stop_after_epoch', '2'))
    ck = os.path.join(d(B, v), 'resume_state.pt')
    stopped = r2.returncode == 3 and os.path.exists(ck) and not os.path.exists(os.path.join(d(B, v), 'metrics.json'))
    r3 = run(v, B)
    resumed = 'RESUMED after epoch 2' in r3.stdout
    good = r1.returncode == 0 and r3.returncode == 0 and stopped and resumed
    eq = good and same_hist(hist(A, v), hist(B, v)) and same_met(met(A, v), met(B, v))
    check(f'R2 {v}: stop after epoch 2 + resume == uninterrupted (history and metrics)', eq,
          f'(stopped={stopped}, resumed={resumed}, rc={r1.returncode},{r2.returncode},{r3.returncode}; checkpoint removed after completion: {not os.path.exists(ck)})')
    if not good: print(r1.stderr[-1500:], r2.stderr[-1500:], r3.stderr[-1500:])
    if v == 'xpc_bal':
        check('R2b metrics.json records the resume', met(B, v)['resume']['resumed_after_epoch'] == 2, str(met(B, v)['resume']))

# R3 cross-environment checkpoint is not resumed
v = 'xpc_bal'
run(v, C, ('--stop_after_epoch', '2'))
ck = os.path.join(d(C, v), 'resume_state.pt'); st = torch.load(ck, map_location='cpu', weights_only=False)
st['env'] = {'device': 'other', 'gpu': 'Imaginary GPU', 'torch': '0.0'}; torch.save(st, ck)
r = run(v, C)
kept = glob.glob(os.path.join(d(C, v), 'resume_state.REJECTED_*.pt'))
check('R3 different-environment checkpoint not resumed, kept as REJECTED, run restarted and equals uninterrupted',
      r.returncode == 0 and 'NOT RESUMED' in r.stdout and len(kept) == 1 and same_hist(hist(A, v), hist(C, v)) and same_met(met(A, v), met(C, v)))

# R4 completed run is skipped, files untouched
def digest(p): return {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob(os.path.join(p, '*')))}
before = digest(d(A, 'xpc_bal')); r = run('xpc_bal', A); after = digest(d(A, 'xpc_bal'))
check('R4 completed run skipped, no file modified', r.returncode == 0 and 'SKIP' in r.stdout and before == after)

# R5 config mismatch refused, checkpoint untouched
D = os.path.join(OUT, 'mismatch'); run('xpc_bal', D, ('--stop_after_epoch', '2'))
ck = os.path.join(d(D, 'xpc_bal'), 'resume_state.pt'); h0 = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
r = subprocess.run([sys.executable, RUNNER, '--variant', 'xpc_bal', '--target', '1', '--seed', '2', '--epochs', EP, '--fake',
                    '--out_root', D, '--lam', '999', '--no_last_ckpt'], capture_output=True, text=True)
check('R5 checkpoint with different config refused and left untouched',
      r.returncode != 0 and os.path.exists(ck) and hashlib.sha256(open(ck, 'rb').read()).hexdigest() == h0)

# R1 package runner == original stage-2 runner (optional; needs the original code tree)
if a.original and os.path.exists(a.original):
    O = os.path.join(OUT, 'original')
    for v in ('ie_off', 'xpc_bal'):
        ro = run(v, O, runner=a.original)
        check(f'R1 {v}: package runner == original improvement-code runner (uninterrupted)',
              ro.returncode == 0 and same_hist(hist(A, v), hist(O, v)) and same_met(met(A, v), met(O, v)))
else:
    print('SKIP R1 (original runner not available here - expected on Colab)')
print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
