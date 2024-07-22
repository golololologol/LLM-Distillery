@echo off
:: Open a new console window and run the commands inside it
start cmd /k "python -m venv .venv && call .\.venv\Scripts\activate.bat"
