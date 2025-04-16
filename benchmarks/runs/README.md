# Benchmark Run Versioning

This directory contains versioned benchmark runs. Each run is stored in a separate directory named `run_N` where N is the run number.

## Directory Structure

Each run directory contains:

- `plots/`: Generated visualization plots and CSV files
- `results/`: Raw JSON result files from benchmark runs
- `logs/`: Log files containing output from the runs
- `README.md`: Information about the specific run

## How to Use

After running benchmarks, use the versioning script to save the results:

```bash
./benchmarks/version_results.sh
```

This will:
1. Create a new run directory with an incremented number
2. Move all plots and results to the new directory (not copy)
3. Copy log files (log.txt) to the logs directory
4. Create a README with the run date and information

The script ensures that plot and result files are moved rather than copied, so there won't be duplicates in the original directories. Log files are copied rather than moved to preserve the original logs.

## Previous Runs

- `run_1`: Initial benchmarks comparing routing policies (do-nothing, gpu_only, simple, advanced, rotate) 