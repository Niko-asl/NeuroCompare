"""Deprecated entry point: it imported a preprocessor class that no longer exists.
Forwards to scripts/preprocess.py, which takes the same --ad_dir/--control_dir/--output_dir arguments."""
import os
import runpy
import sys

if __name__ == "__main__":
    print("preprocess_data.py is deprecated; running scripts/preprocess.py", file=sys.stderr)
    runpy.run_path(os.path.join(os.path.dirname(__file__), "preprocess.py"), run_name="__main__")
