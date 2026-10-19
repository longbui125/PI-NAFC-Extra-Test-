"""Build the original model notebooks and the separate seed-1 budget ablation."""
from build_gemma_ngoc import build
from build_budget756 import build as build_budget756


if __name__ == '__main__':
    print(build('gemma'))
    print(build('phi'))
    print(build_budget756())
