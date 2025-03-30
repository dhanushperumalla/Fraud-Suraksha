@echo off
echo Fraud Suraksha - Startup Utility
echo ==============================
echo.

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

:menu
echo.
echo Choose an option:
echo 1. Launch Fraud Suraksha app
echo 2. Reset database (use if you get connection errors)
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto launch
if "%choice%"=="2" goto reset
if "%choice%"=="3" goto end

echo Invalid choice. Please try again.
goto menu

:reset
echo.
echo Resetting database...
python reset_db.py
echo.
echo Press any key to return to the menu...
pause > nul
goto menu

:launch
echo.
echo Launching Fraud Suraksha...
echo.
streamlit run app.py
goto end

:end
echo.
echo Thank you for using Fraud Suraksha!
pause 