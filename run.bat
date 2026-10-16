@echo off
cd /d "%~dp0"
echo Starting Corpus Inquiry Lab...
.\.venv\Scripts\python.exe -m streamlit run src\app.py
pause
