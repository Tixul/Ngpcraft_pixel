"""Run pip check, allowing only the verified MediaPipe 0.10.21 ARM tag defect."""
import importlib
import importlib.metadata
import platform
import subprocess
import sys


def known_mediapipe_tag_error(result, system, machine, version):
    return (
        result.returncode == 1
        and system == 'Darwin'
        and machine == 'arm64'
        and version == '0.10.21'
        and not result.stderr.strip()
        and result.stdout.strip() == 'mediapipe 0.10.21 is not supported on this platform'
    )


def main():
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True
    )
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode == 0:
        return 0
    try:
        version = importlib.metadata.version('mediapipe')
    except importlib.metadata.PackageNotFoundError:
        return result.returncode
    if not known_mediapipe_tag_error(result, platform.system(), platform.machine(), version):
        return result.returncode

    # The universal2 wheel incorrectly declares x86_64 only in WHEEL.
    # Verify the native libraries actually load in this ARM interpreter.
    # Do not accept missing dependencies, other versions, or any other pip error.
    # Upstream report: https://github.com/google-ai-edge/mediapipe/issues/5843
    for module in (
        'mediapipe.python._framework_bindings',
        'mediapipe.tasks.python.metadata.flatbuffers_lib._pywrap_flatbuffers',
        'mediapipe.tasks.cc.metadata.python._pywrap_metadata_version',
    ):
        importlib.import_module(module)
    print('Verified MediaPipe native imports on ARM64; ignoring its incorrect wheel platform tag only.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
