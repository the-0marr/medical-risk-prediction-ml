import subprocess
import sys

scripts = [
    'synthetic_data_gen.py',
    'add_impurities.py',
    'data_preparation.py',
    'feature_engineering.py',
    'train_test_split.py',
    'models_training.py',
    'optimization_model.py',
    'ensemble_model.py'
]

for script in scripts:
    print(f'\nRunning {script}...')
    subprocess.run([sys.executable, script], check=True)

print('\nPipeline completed successfully.')
