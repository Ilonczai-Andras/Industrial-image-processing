@echo off
echo ============================================================
echo  Virtualis kornyezet letrehozasa es csomagok telepitese
echo ============================================================

REM Virtuális környezet létrehozása
py -m venv .venv
if errorlevel 1 (
    echo HIBA: A virtualis kornyezet letrehozasa sikertelen.
    echo Gyozodj meg rola, hogy a Python telepitve van es elerheto PATH-ban.
    pause
    exit /b 1
)

REM Aktiválás
call .venv\Scripts\activate.bat

REM pip frissítése
py -m pip install --upgrade pip

REM Csomagok telepítése
echo.
echo Csomagok telepitese
pip install open3d numpy scipy matplotlib pyvista

echo.
echo ============================================================
echo  Telepites kesz!
echo  Futtatas: python registration_pipeline.py
echo ============================================================
pause
