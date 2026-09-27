@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo  NgpCraft Pixel - DIAGNOSTIC du venv
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] Pas de venv trouve dans .venv\
    echo     Relance run.bat pour le creer.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo [Python]
where python
python --version
echo.

echo [pip list - packages cles]
python -m pip list 2>nul | findstr /I "opencv mediapipe rembg onnxruntime pyside6 pillow numpy"
echo.

echo [Import core (doit tout passer)]
python -c "import PySide6; print('  PySide6', PySide6.__version__)" 2>&1
python -c "import PIL; print('  PIL', PIL.__version__)" 2>&1
python -c "import numpy; print('  numpy', numpy.__version__)" 2>&1
python -c "import cv2; print('  cv2', cv2.__version__)" 2>&1
echo.

echo [Import ML (optionnel)]
python -c "import mediapipe; print('  mediapipe OK')" 2>&1
python -c "import rembg; print('  rembg OK')" 2>&1
python -c "import onnxruntime; print('  onnxruntime OK')" 2>&1
echo.

echo [Test lancement main.py (dry-run imports seulement)]
python -c "import main" 2>&1
echo.

echo ================================================================
echo  Si main.py renvoie une erreur ci-dessus, copie-la et partage-la.
echo  Sinon, lance repair.bat pour reparer le venv.
echo ================================================================
echo.
pause
