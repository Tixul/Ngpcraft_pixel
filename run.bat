@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [NgpCraft Pixel] Creation du venv...
    python -m venv .venv || goto :err
    call ".venv\Scripts\activate.bat"
    echo [NgpCraft Pixel] Installation des dependances...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt || goto :err
) else (
    call ".venv\Scripts\activate.bat"
)

python main.py
goto :eof

:err
echo [NgpCraft Pixel] Erreur setup. Appuyez sur une touche...
pause
exit /b 1
