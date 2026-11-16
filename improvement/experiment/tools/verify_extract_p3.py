"""Verify a returned p3_t4_seed<k>.zip against its internal SHA-256 manifest, copy it to returned_results/, and extract
it into code/stage3/ WITHOUT overwriting (existing identical files are skipped; differing files abort).
Usage: python tools/verify_extract_p3.py <zip> [<zip> ...]"""
import hashlib, os, shutil, sys, zipfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S3 = os.path.join(PKG, 'code', 'stage3')
RET = os.path.join(PKG, 'returned_results'); os.makedirs(RET, exist_ok=True)
for zp in sys.argv[1:]:
    Z = zipfile.ZipFile(zp)
    mans = [n for n in Z.namelist() if n.startswith('MANIFEST_') and n.endswith('.sha256')]
    assert len(mans) == 1, 'no manifest'
    rows = dict(l.split('  ', 1)[::-1] for l in Z.read(mans[0]).decode().splitlines() if l.strip())
    members = [n for n in Z.namelist() if n != mans[0]]
    bad = [n for n in members if hashlib.sha256(Z.read(n)).hexdigest() != rows.get(n)] + [n for n in rows if n not in members]
    assert not bad, f'hash mismatch: {bad}'
    for n in members:
        assert not n.startswith(('/', '\\')) and '..' not in n.split('/'), f'unsafe path {n}'
    new = same = 0
    for n in members + mans:
        dst = os.path.join(S3, 'manifests', n) if n in mans else os.path.join(S3, *n.split('/'))
        data = Z.read(n)
        if os.path.exists(dst):
            assert open(dst, 'rb').read() == data, f'existing file differs, not overwritten: {dst}'
            same += 1; continue
        os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'wb').write(data); new += 1
    keep = os.path.join(RET, os.path.basename(zp).replace(' (1)', ''))
    if not os.path.exists(keep): shutil.copy2(zp, keep)
    n_runs = sum(1 for n in members if n.endswith('metrics.json'))
    print(f'{os.path.basename(zp)}: {len(rows)} files verified by SHA-256 | {n_runs} complete runs | extracted {new}, '
          f'already present+identical {same} | zip sha256 {hashlib.sha256(open(zp, "rb").read()).hexdigest()[:16]}')
