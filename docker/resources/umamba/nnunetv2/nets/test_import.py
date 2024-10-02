import sys
import os

# Ensure the parent directory of 'vmamba' is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print("Current sys.path:", sys.path)
print("Current working directory:", os.getcwd())

try:
    from vmamba.SwinTransformer import PatchExpanding
    print("Successfully imported PatchExpanding")
except ImportError as e:
    print(f"ImportError: {e}")