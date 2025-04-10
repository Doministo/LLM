@echo off
title LLM Chat - Streamlit + Ollama
cd /d "%~dp0"

:: Aktivuj virtuální prostředí (pokud existuje)
IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Spusť Streamlit app
echo Spoustim aplikaci...
streamlit run Python\chat_llm_1v0.py

pause
