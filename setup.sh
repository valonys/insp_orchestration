#!/bin/bash
echo "🔧 Upgrading pip and installing matplotlib..."
python -m pip install --upgrade pip
python -m pip install --upgrade matplotlib

echo "📦 Installing remaining requirements..."
pip install -r requirements.txt
