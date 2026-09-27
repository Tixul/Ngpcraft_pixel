@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo  NgpCraft Pixel - Installation des dependances ML
echo ================================================================
echo.
echo Ce script installe :
echo   - opencv-contrib-python (superset, requis par mediapipe)
echo   - mediapipe (face detection + face mesh)
echo   - rembg + onnxruntime (matting ML + super-resolution)
echo.
echo ATTENTION : ferme NgpCraft Pixel AVANT de lancer ce script.
echo (cv2.pyd est verrouille tant que l'app tourne)
echo.
pause

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ML setup] Pas de venv trouve. Lance d'abord run.bat une fois pour le creer.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo [ML setup] Python utilise :
where python
echo.

echo [ML setup] Etape 1/3 : retrait de opencv-python s'il est present
echo (necessaire : conflit avec opencv-contrib-python requis par mediapipe)
python -m pip uninstall -y opencv-python
echo.

echo [ML setup] Etape 2/3 : installation de opencv-contrib-python
python -m pip install --upgrade opencv-contrib-python
if errorlevel 1 (
    echo.
    echo [ML setup] ERREUR install opencv-contrib-python. Arret.
    pause
    exit /b 1
)
echo.

echo [ML setup] Etape 3/3 : installation des libs ML
python -m pip install --upgrade mediapipe rembg onnxruntime
if errorlevel 1 (
    echo.
    echo [ML setup] ERREUR install deps ML. Verifie le message ci-dessus.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  [ML setup] Installation terminee avec succes.
echo.
echo  Tu peux maintenant relancer run.bat.
echo  Les features ML seront dispo dans le panneau de reglages :
echo    - Auto-crop sur visage (MediaPipe)
echo    - Preserver les yeux (MediaPipe Face Mesh)
echo    - rembg cutout (matting SOTA)
echo    - Super-resolution Real-ESRGAN
echo ================================================================
echo.
pause
