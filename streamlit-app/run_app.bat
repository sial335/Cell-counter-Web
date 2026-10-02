@echo off
cd /d "%~dp0"

if not exist venv (
    echo First run: creating the environment. This needs internet ONCE.
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

echo Starting the Cell Counter...
streamlit run app.py
pause
