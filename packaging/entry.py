"""Frozen application entry point, including the CI smoke test."""
import sys

import main

if __name__ == '__main__':
    if '--smoke-test' in sys.argv:
        import numpy as np
        from PIL import Image
        from PySide6.QtWidgets import QApplication
        import pipeline

        # Check optional libraries were actually collected, without downloading models.
        assert pipeline._HAS_CV2
        assert pipeline._HAS_MEDIAPIPE
        assert pipeline._HAS_REMBG
        assert pipeline._HAS_ONNXRUNTIME
        app = QApplication([])
        window = main.MainWindow()
        source = Image.new('RGBA', (32, 32), (220, 70, 30, 255))
        result = pipeline.process(source, target_w=16, target_h=16, remove_bg=False)
        assert result.size == (16, 16)
        assert np.all(np.asarray(result)[:, :, :3] % 17 == 0)
        window.close()
        app.processEvents()
    else:
        main.main()
