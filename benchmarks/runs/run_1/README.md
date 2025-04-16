# Run 1
Date: April 16, 2023

## Benchmark Results

This directory contains the initial benchmarks comparing different routing policies:

- `do-nothing`: Default routing without any policy modifications
- `gpu_only`: Only use experts that are already in GPU
- `simple`: Keep only top 4 most popular experts
- `advanced`: Use threshold-based expert selection
- `rotate`: Disable specific expert ranks

## Observations

- All runs had batch size = 4
- Hit rate was identical (0.867) for do-nothing, advanced, and rotate policies, which indicates a potential issue in how routing policies are implemented or how hit rate is calculated
- The gpu_only policy showed the best performance with lowest decode time (31.84s)
- The simple policy reduced the average number of experts per layer to 4.31 from 5.22

## Files

- `plots/`: Generated visualization plots
- `results/`: Raw JSON result files from benchmark runs 