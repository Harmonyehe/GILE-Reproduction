"""Build P3_STAGE3_COLAB.ipynb (relative paths only). Embeds code/ (shared modules, stage3/, stage2_ref/) and verifies each
file by SHA-256 on Colab. Usage: python tools/build_p3_notebook.py"""
import base64, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
SKIP = {'__pycache__', 'runs_tests', 'runs_tests_resume', 'runs_p3', 'runs_smoke', 'data'}
files = []
for dp, dns, fns in os.walk(os.path.join(PKG, 'code')):
    dns[:] = sorted(x for x in dns if x not in SKIP)
    files += [os.path.relpath(os.path.join(dp, f), PKG).replace('\\', '/') for f in sorted(fns) if f.endswith(('.py', '.json'))]
payload = {f: {'sha256': hashlib.sha256(b).hexdigest(), 'b64': base64.b64encode(b).decode()}
           for f in files for b in [open(os.path.join(PKG, f), 'rb').read()]}
cfg = json.load(open(os.path.join(PKG, 'config', 'xpcb_experiment_config_COPY.json')))
ANCHOR, UCI = cfg['design']['anchor']['expected_T4'], cfg['dataset']

def code(src): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}
def md(src): return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
cells = [
md("""# Person 3 stage 3: GILE input-layout fix (C) and person-style augmentation (B) — UCI HAR, T4
Plan and success rule fixed in PREDECLARED_PLAN.md before any result. **Run cells one at a time.** After each seed cell,
wait until its zip has downloaded before starting the next cell. No Google Drive, no account data."""),
code("""# 1. GPU check
import subprocess, torch, sys
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"""),
code("""# 2. Write embedded files, verify SHA-256
import base64, hashlib, json, os
PKG = '/content/p3'
PAYLOAD = json.loads(%r)
for name, item in PAYLOAD.items():
    data = base64.b64decode(item['b64']); assert hashlib.sha256(data).hexdigest() == item['sha256'], name
    path = os.path.join(PKG, name); os.makedirs(os.path.dirname(path), exist_ok=True); open(path, 'wb').write(data)
print(len(PAYLOAD), 'files written and verified')
CODE = os.path.join(PKG, 'code'); S3 = os.path.join(CODE, 'stage3'); OUT = 'runs_p3'; ROOT = os.path.join(S3, OUT); os.chdir(CODE)""" % json.dumps(payload)),
code("""# 3. Download UCI HAR, verify SHA-256, arrange layout
import urllib.request, zipfile, shutil, io
URL, SHA = %r, %r
raw = urllib.request.urlopen(URL, timeout=300).read(); h = hashlib.sha256(raw).hexdigest(); print('sha256 ok:', h == SHA); assert h == SHA
inner = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw)).read('UCI HAR Dataset.zip'))); inner.extractall('/content/uci_raw')
src = '/content/uci_raw/UCI HAR Dataset'
for s in ('train', 'test'):
    dst = os.path.join(CODE, 'data', 'ucihar', s, 'InertialSignals'); os.makedirs(dst, exist_ok=True)
    for f in os.listdir(os.path.join(src, s, 'Inertial Signals')): shutil.copy(os.path.join(src, s, 'Inertial Signals', f), dst)
    for f in ('y_%%s.txt' %% s, 'subject_%%s.txt' %% s): shutil.copy(os.path.join(src, s, f), os.path.join(CODE, 'data', 'ucihar', s))
n = sum(len(fs) for _, _, fs in os.walk(os.path.join(CODE, 'data', 'ucihar'))); print(n, 'files (expect 22)'); assert n == 22""" % (UCI['url'], UCI['sha256'])),
code("""# 4. Tests (must end with ALL PASSED)
r = subprocess.run([sys.executable, 'stage3/checks/test_stage3.py'], capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-1500:]); assert 'ALL PASSED' in r.stdout, 'TESTS FAILED - stop here'"""),
code("""# 5. Helpers: 3 runs in parallel on the GPU (each run is deterministic on its own); skip completed runs
import time, glob
from concurrent.futures import ThreadPoolExecutor
from google.colab import files
ANCHOR = %r
ARMS = ('fix', 'aug', 'fix_aug'); TARGETS = ('0', '1', '2', '3', '4')
def run_one(a, t, seed):
    t0 = time.time()
    r = subprocess.run([sys.executable, 'stage3/run_stage3.py', '--arm', a, '--target', t, '--seed', str(seed),
                        '--out_root', OUT, '--no_last_ckpt'], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith(('DONE', 'SKIP', 'RESUMED', 'NOT RESUMED'))]
    print(f'seed {seed} target {t} {a:8s} exit {r.returncode} {time.time()-t0:5.0f}s | ' + ' || '.join(l.split(' -> ')[0] for l in lines), flush=True)
    if r.returncode != 0:
        d = os.path.join(ROOT, 'ucihar', a, f'target{t}', f'seed{seed}'); os.makedirs(d, exist_ok=True)
        stamp = time.strftime('%%Y%%m%%d_%%H%%M%%S')
        open(os.path.join(d, f'FAILED_{stamp}_stderr.txt'), 'w').write(r.stderr); open(os.path.join(d, f'FAILED_{stamp}_stdout.txt'), 'w').write(r.stdout)
        print('   STDERR tail:', r.stderr.strip().splitlines()[-3:], flush=True)
    return (a, t, r.returncode)
def run_seed(seed, workers=3):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda at: run_one(at[0], at[1], seed), [(a, t) for t in TARGETS for a in ARMS]))
    fails = [x for x in res if x[2] != 0]; print('FAILED RUNS:', fails if fails else 'none', f'| seed wall time {time.time()-t0:.0f}s')
def package_seed(seed, extra=()):
    dirs = [os.path.join(ROOT, 'ucihar', a, f'target{t}', f'seed{seed}') for a in ARMS for t in TARGETS] + list(extra)
    done = [d for d in dirs if os.path.exists(os.path.join(d, 'metrics.json'))]
    z = f'/content/p3_t4_seed{seed}.zip'; man = f'MANIFEST_p3_seed{seed}.sha256'; lines = []
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as Z:
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, '*'))):
                if f.endswith(('resume_state.pt', '.tmp')): continue
                arc = os.path.relpath(f, S3).replace(os.sep, '/')
                lines.append(f'{hashlib.sha256(open(f, "rb").read()).hexdigest()}  {arc}'); Z.write(f, arc)
        Z.writestr(man, '\\n'.join(lines) + '\\n')
    Z = zipfile.ZipFile(z); listed = dict(l.split('  ', 1)[::-1] for l in Z.read(man).decode().splitlines() if l.strip())
    bad = [a for a in Z.namelist() if a != man and hashlib.sha256(Z.read(a)).hexdigest() != listed.get(a)]
    print(z, len(listed), 'files', round(os.path.getsize(z)/2**20, 1), 'MB | verified', 'OK' if not bad else f'BAD {bad}', f'| complete runs {len(done)}/{len(dirs)}')
    return z""" % ANCHOR),
code("""# 6. Reproducibility anchor: control arm (ie_off) target 0 seed 0 must equal the stage-1 T4 value exactly
r = run_one('ie_off', '0', 0)
m = json.load(open(os.path.join(ROOT, 'ucihar', 'ie_off', 'target0', 'seed0', 'metrics.json')))
got = {'selected_epoch': m['selection']['selected_epoch'], 'accuracy': m['target_at_selected_epoch']['accuracy'], 'weighted_f1': m['target_at_selected_epoch']['weighted_f1']}
print('ANCHOR expected', ANCHOR); print('ANCHOR obtained', got); print('ANCHOR EXACT MATCH:', got == ANCHOR)"""),
] + [code(f"""# {7 + s}. SEED {s}: 15 runs (3 arms x 5 targets, 3 in parallel), package + verify + download. WAIT for the download.
run_seed({s})
z = package_seed({s}{", extra=[os.path.join(ROOT, 'ucihar', 'ie_off', 'target0', 'seed0')]" if s == 0 else ''})
files.download(z)""") for s in range(5)]
nb = {'nbformat': 4, 'nbformat_minor': 0, 'cells': cells,
      'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}}}
outp = os.path.join(PKG, 'P3_STAGE3_COLAB.ipynb')
json.dump(nb, open(outp, 'w', encoding='utf-8'), indent=1)
print('wrote', outp, round(os.path.getsize(outp) / 1024), 'KB,', len(payload), 'embedded files:', ', '.join(files))
