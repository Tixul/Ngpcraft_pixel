"""Frozen application entry point, including the CI smoke test."""
import os
from pathlib import Path
import sys
import traceback


def run():
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if '--smoke-test' in sys.argv:
        # Import directly so the original exception is not hidden by pipeline's
        # optional-feature fallback. Also exercise the symbols the app uses.
        import cv2
        import mediapipe
        import onnxruntime
        from rembg import remove, new_session

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
        app = QApplication([])
        window = main.MainWindow()
        source = Image.new('RGBA', (32, 32), (220, 70, 30, 255))
        result = pipeline.process(source, target_w=16, target_h=16, remove_bg=False)
        assert result.size == (16, 16)
        assert np.all(np.asarray(result)[:, :, :3] % 17 == 0)
        window.close()
        app.processEvents()
    else:
        import main
        main.main()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    try:
        run()
    except BaseException:
        # Windowed Windows/macOS binaries have no stderr. Save the traceback
        # for build.py to display in Actions, without opening a blocking dialog.
        report = os.environ.get('NGPCRAFT_SMOKE_REPORT')
        if '--smoke-test' in sys.argv and report:
            Path(report).write_text(traceback.format_exc(), encoding='utf-8')
            sys.exit(1)
        raise
