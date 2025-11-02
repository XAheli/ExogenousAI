#!/bin/bash
# Quick Start Script for ExogenousAI
# This script sets up the environment and runs the complete pipeline

echo "=========================================="
echo "  ExogenousAI Quick Start Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run the full analysis pipeline:"
echo "   python src/run_pipeline.py"
echo ""
echo "2. Launch the interactive dashboard:"
echo "   streamlit run src/dashboard.py"
echo ""
echo "3. Run tests:"
echo "   python -m pytest tests/"
echo ""
echo "=========================================="
