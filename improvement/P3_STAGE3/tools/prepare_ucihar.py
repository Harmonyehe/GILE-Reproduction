"""Download (or use a local copy of) the public UCI HAR archive, verify its SHA-256, and arrange the files where the
released loader expects them: code/data/ucihar/{train,test}/InertialSignals/*.txt + y_*.txt + subject_*.txt (22 files).
Same steps as cell 3 of P3_STAGE3_COLAB.ipynb.
Usage:  python tools/prepare_ucihar.py                 (downloads from the UCI repository, ~60 MB)
        python tools/prepare_ucihar.py --zip <path>    (uses an archive you already downloaded)"""
import argparse, hashlib, io, os, shutil, tempfile, urllib.request, zipfile

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
SHA = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CODE = os.path.join(PKG, 'code')
p = argparse.ArgumentParser(); p.add_argument('--zip'); a = p.parse_args()
raw = open(a.zip, 'rb').read() if a.zip else urllib.request.urlopen(URL, timeout=300).read()
h = hashlib.sha256(raw).hexdigest(); names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
tmp = tempfile.mkdtemp()
if 'UCI HAR Dataset.zip' in names:                     # the official outer archive
    assert h == SHA, f'SHA-256 mismatch: {h}'
    print('archive SHA-256 verified')
    zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw)).read('UCI HAR Dataset.zip'))).extractall(tmp)
else:                                                  # the inner 'UCI HAR Dataset.zip' given directly
    assert any(n.startswith('UCI HAR Dataset/') for n in names), 'not a UCI HAR archive'
    print(f'WARNING: inner archive given (sha256 {h[:16]}); the official SHA-256 applies to the outer archive only')
    zipfile.ZipFile(io.BytesIO(raw)).extractall(tmp)
src = os.path.join(tmp, 'UCI HAR Dataset')
for s in ('train', 'test'):
    dst = os.path.join(CODE, 'data', 'ucihar', s, 'InertialSignals'); os.makedirs(dst, exist_ok=True)
    for f in os.listdir(os.path.join(src, s, 'Inertial Signals')): shutil.copy(os.path.join(src, s, 'Inertial Signals', f), dst)
    for f in (f'y_{s}.txt', f'subject_{s}.txt'): shutil.copy(os.path.join(src, s, f), os.path.join(CODE, 'data', 'ucihar', s))
shutil.rmtree(tmp, ignore_errors=True)
n = sum(len(fs) for _, _, fs in os.walk(os.path.join(CODE, 'data', 'ucihar')) if True)
print(n, 'files under code/data/ucihar (expect 22; more if the loader has already cached .npz files)')
