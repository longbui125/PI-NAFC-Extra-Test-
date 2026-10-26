"""Recompute the published seed-1 table from aligned per-question scores."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/phi_pilot'))
from hooks import read, save
from metrics import summarize, write_report


def main():
    folder = ROOT/'results/seed_1'
    rows = summarize(read(folder/'reference.json'),
                     {arm:read(folder/f'{arm}.json') for arm in ('mon','rep','quota','kl')})
    save(folder/'summary.json',dict(seed=1,steps=300,pi_budget_tokens=400.86,rows=rows))
    write_report(folder,rows,'Completed Phi–Olympiad seed 1 · PI 400.86 tokens')


if __name__ == '__main__':
    main()
