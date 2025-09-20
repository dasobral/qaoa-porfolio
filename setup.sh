cargo init --name qaoa_portfolio --lib

pip install --upgrade pip setuptools wheel
pip install pennylane numpy pandas matplotlib scipy
pip install yfinance alpha-vantage quandl requests
pip install pytest black flake8 jupyter ipykernel

mkdir -p core/src/{portfolio,optimization,math,utils}
mkdir -p python/{qaoa_portfolio,tests}
mkdir -p data/{sample_datasets,benchmarks,real_market_data}
mkdir -p examples benchmarks/{performance_tests,accuracy_analysis,scaling_studies}
mkdir -p docs scripts results/{benchmarks,visualizations,optimization_runs}

# Set up Python package structure
touch python/qaoa_portfolio/__init__.py
touch python/qaoa_portfolio/{quantum_backend.py,data_loader.py,visualization.py}
touch python/tests/__init__.py
touch python/setup.py

# Add results directory to .gitignore (already there, but ensure)
echo "results/" >> .gitignore

# Verify setup
echo "Setup verification:"
echo "Rust version: $(rustc --version)"
echo "Python version: $(python --version)"
echo "PennyLane version: $(pip show pennylane | grep Version)"
