"""Frozen application entry point, including the CI smoke test."""
import os
from pathlib import Path
import sys
import traceback
import faulthandler
import time

_smoke_log = None
_started = time.monotonic()


def progress(message):
    line = f'[smoke +{time.monotonic() - _started:.1f}s] {message}'
    if sys.stdout is not None:
        print(line, flush=True)
    if _smoke_log is not None:
        _smoke_log.write(line + '\n')
        _smoke_log.flush()


def run():
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if '--smoke-test' in sys.argv:
        # Import directly so the original exception is not hidden by pipeline's
        # optional-feature fallback. Also exercise the symbols the app uses.
        progress('Importing OpenCV')
        import cv2
        progress('Importing MediaPipe')
        import mediapipe
        progress('Importing ONNX Runtime')
        import onnxruntime
        progress('Importing rembg / Numba')
        from rembg import remove, new_session

        progress('Importing application and Qt')
        import main
        import numpy as np
        from PIL import Image
        from PySide6.QtWidgets import QApplication
        import pipeline

        # Check optional libraries were actually collected, without downloading models.
        assert pipeline._HAS_CV2, 'OpenCV unavailable'
        assert pipeline._HAS_MEDIAPIPE, 'MediaPipe unavailable'
        assert pipeline._HAS_REMBG, 'rembg unavailable'
        assert pipeline._HAS_ONNXRUNTIME, 'ONNX Runtime unavailable'
        progress('Creating Qt application')
        app = QApplication([])
        progress('Creating main window')
        window = main.MainWindow()
        progress('Converting synthetic image')
        source = Image.new('RGBA', (32, 32), (220, 70, 30, 255))
        result = pipeline.process(source, target_w=16, target_h=16, remove_bg=False)
        assert result.size == (16, 16)
        assert np.all(np.asarray(result)[:, :, :3] % 17 == 0)
        progress('Closing main window')
        window.close()
        app.processEvents()
        progress('PASS')
    else:
        import main
        main.main()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    report = os.environ.get('NGPCRAFT_SMOKE_REPORT')
    if '--smoke-test' in sys.argv and report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        _smoke_log = open(report, 'a', encoding='utf-8')
        faulthandler.dump_traceback_later(60, repeat=True, file=_smoke_log)
    try:
        run()
    except BaseException:
        # Windowed Windows/macOS binaries have no stderr. Save the traceback
        # for build.py to display in Actions, without opening a blocking dialog.
        if '--smoke-test' in sys.argv and report:
            _smoke_log.write(traceback.format_exc())
            _smoke_log.flush()
            sys.exit(1)
        raise
    finally:
        if _smoke_log is not None:
            faulthandler.cancel_dump_traceback_later()
            _smoke_log.close()
