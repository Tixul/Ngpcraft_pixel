@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo  NgpCraft Pixel - REPAIR du venv
echo ================================================================
echo.
echo Ce script :
echo  - Vire les deux variantes d'opencv (clean state)
echo  - Reinstalle opencv-contrib-python et les core deps
echo  - Verifie que les imports de base marchent
echo.
echo IMPORTANT : ferme NgpCraft Pixel AVANT de continuer.
echo.
pause

if not exist ".venv\Scripts\python.exe" (
    echo [!] Pas de venv trouve. Relance run.bat pour le creer.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo [1/5] Uninstall opencv-python + opencv-contrib-python (nettoyage)
python -m pip uninstall -y opencv-python opencv-contrib-python
echo.

echo [2/5] Install core deps (PySide6 Pillow numpy)
python -m pip install --upgrade PySide6 Pillow numpy
if errorlevel 1 goto :err_core
echo.

echo [3/5] Install opencv-contrib-python (version propre)
python -m pip install --upgrade --force-reinstall opencv-contrib-python
if errorlevel 1 goto :err_cv2
echo.

echo [4/5] Verif imports core
python -c "import PySide6; import PIL; import numpy; import cv2; print('Core OK, cv2 version:', cv2.__version__)"
if errorlevel 1 goto :err_import
echo.

echo [5/5] Verif main.py import
python -c "import main; print('main OK')" 2>&1
if errorlevel 1 goto :err_main
echo.

echo ================================================================
echo  REPAIR OK. Relance run.bat pour ouvrir l'app.
echo.
echo  Pour installer les libs ML (optionnel) :
echo   - Option A : re-lance ce script avec flag ML :  repair.bat ml
echo   - Option B : lance install_ml.bat
echo   - Option C : depuis l'app, ouvre Gestionnaire de modeles ML
echo ================================================================
echo.
if "%1"=="ml" goto :install_ml
pause
exit /b 0

:install_ml
echo.
echo [BONUS] Install libs ML (mediapipe rembg onnxruntime)
python -m pip install --upgrade mediapipe rembg onnxruntime
if errorlevel 1 (
    echo Install ML echouee. Core est OK, relance l'app et utilise le Gestionnaire.
    pause
    exit /b 0
)
echo.
echo ML libs installees. Relance run.bat.
pause
exit /b 0

:err_core
echo.
echo ERREUR installation des core deps. Verifie ta connexion / droits.
pause
exit /b 1

:err_cv2
echo.
echo ERREUR install opencv-contrib-python. Peut-etre que cv2.pyd est encore
echo verrouille par un process. Ferme toute instance NgpCraft Pixel et re-lance.
pause
exit /b 1

:err_import
echo.
echo Les imports core echouent. Signale l'erreur vue ci-dessus.
pause
exit /b 1

:err_main
echo.
echo main.py ne s'importe pas. L'erreur exacte est au-dessus.
pause
exit /b 1
