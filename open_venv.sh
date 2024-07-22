#!/bin/bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Start a new shell session to keep the terminal open and interact with the venv
exec bash --rcfile <(echo '. .venv/bin/activate')
