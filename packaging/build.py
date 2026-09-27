"""Build on the target OS, smoke-test the binary, then archive it."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
target = os.environ.get('BUILD_TARGET', f'{sys.platform}-{platform.machine()}'.lower())
if not target or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in target):
    raise ValueError('Invalid BUILD_TARGET')
# Catch dependency import failures before spending time on PyInstaller.
env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
subprocess.run([sys.executable, 'packaging/entry.py', '--smoke-test'], check=True, timeout=120, env=env)
command = [
    sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
    '--onedir', '--windowed', '--name', 'NgpCraftPixel', '--paths', str(ROOT),
    '--exclude-module', 'PyQt5', '--exclude-module', 'PyQt6',
    '--exclude-module', 'PySide2',
    # pymatting reads its distribution version at import time. Without the
    # metadata rembg raises PackageNotFoundError (an ImportError subclass).
    '--copy-metadata', 'pymatting',
]
for package in ('mediapipe', 'rembg', 'onnxruntime'):
    command.extend(['--collect-all', package])
command.append('packaging/entry.py')
subprocess.run(command, check=True)
bundle = ROOT / 'dist' / 'NgpCraftPixel'
if sys.platform == 'darwin':
    bundle = bundle.with_suffix('.app')
    executable = bundle / 'Contents' / 'MacOS' / 'NgpCraftPixel'
else:
    executable = bundle / ('NgpCraftPixel.exe' if sys.platform == 'win32' else 'NgpCraftPixel')
report = ROOT / 'build' / 'smoke-test-error.txt'
report.unlink(missing_ok=True)
env['NGPCRAFT_SMOKE_REPORT'] = str(report)
try:
    subprocess.run([str(executable), '--smoke-test'], check=True, timeout=120, env=env)
finally:
    if report.exists():
        print(report.read_text(encoding='utf-8'), flush=True)
release = ROOT / 'release'
release.mkdir(exist_ok=True)
archive_base = release / f'NgpCraftPixel-{target}'
if sys.platform == 'darwin':
    # ditto preserves the symlinks and executable permissions in an app bundle.
    archive = Path(str(archive_base) + '.zip')
    subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(bundle), str(archive)], check=True)
else:
    archive = Path(shutil.make_archive(str(archive_base), 'zip' if sys.platform == 'win32' else 'gztar', root_dir=bundle.parent, base_dir=bundle.name))
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
archive.with_name(archive.name + '.sha256').write_text(f'{digest.hexdigest()}  {archive.name}\n', encoding='ascii')
print(f'Built and tested: {archive.name}')
