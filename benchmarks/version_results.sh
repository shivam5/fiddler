#!/bin/bash

# Get the next run number
LATEST_RUN=$(ls -1d benchmarks/runs/run_* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_RUN" ]; then
    NEXT_RUN_NUM=1
else
    LATEST_RUN_NUM=$(echo $LATEST_RUN | sed 's/.*run_//')
    NEXT_RUN_NUM=$((LATEST_RUN_NUM + 1))
fi

# Create run directory
RUN_DIR="benchmarks/runs/run_$NEXT_RUN_NUM"
mkdir -p "$RUN_DIR/plots" "$RUN_DIR/results" "$RUN_DIR/logs"

# Count files before moving
PLOT_COUNT=$(ls -1 benchmarks/plots/ 2>/dev/null | wc -l)
RESULT_COUNT=$(ls -1 benchmarks/results/ 2>/dev/null | wc -l)
LOG_COUNT=0

# Move results and plots to the versioned directory
if [ $PLOT_COUNT -gt 0 ]; then
    mv benchmarks/plots/* "$RUN_DIR/plots/" 2>/dev/null
fi

if [ $RESULT_COUNT -gt 0 ]; then
    mv benchmarks/results/* "$RUN_DIR/results/" 2>/dev/null
fi

# Move log files if they exist
if [ -f "log.txt" ]; then
    LOG_COUNT=1
    cp log.txt "$RUN_DIR/logs/log.txt"
    echo "Copied log.txt to $RUN_DIR/logs/"
fi

if [ -f "benchmarks/log.txt" ]; then
    LOG_COUNT=$((LOG_COUNT + 1))
    cp benchmarks/log.txt "$RUN_DIR/logs/benchmarks_log.txt"
    echo "Copied benchmarks/log.txt to $RUN_DIR/logs/"
fi

echo "Results moved to $RUN_DIR"
echo "----------------------------"
echo "Plots: $PLOT_COUNT files"
echo "Results: $RESULT_COUNT files"
echo "Logs: $LOG_COUNT files"

# Create README with timestamp
echo "# Run $NEXT_RUN_NUM" > "$RUN_DIR/README.md"
echo "Date: $(date)" >> "$RUN_DIR/README.md"
echo "" >> "$RUN_DIR/README.md"
echo "## Benchmark Results" >> "$RUN_DIR/README.md"
echo "This directory contains results from benchmark run #$NEXT_RUN_NUM" >> "$RUN_DIR/README.md"

echo "Created $RUN_DIR/README.md with run information"
echo "Original files have been moved (not copied), log files have been copied" 