#!/bin/bash
# ASG Model Fitting for SOLO rounds
# Sequential processing based on successful run_all_groups_ASG_only.sh pattern

echo "Starting ASG model fitting for SOLO rounds"
echo "Total groups to process: 33"
echo "Started at: $(date)"

# Array of all group numbers with solo data
groups=(0 1 2 4 6 8 16 19 23 25 26 28 29 30 32 34 35 36 37 38 39 41 42 43 45 48 50 51 52 54 55 58 61)

# Create log directory
mkdir -p logs_ASG_solo

# Track progress
completed=0
total=${#groups[@]}

for group in "${groups[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing group $group ($((completed + 1))/$total)"
    echo "Started at: $(date)"
    echo "========================================="
    
    python3 modelFitting_ASG_solo.py $group 2>&1 | tee "logs_ASG_solo/group_${group}.log"
    
    # Check if successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Group $group ASG solo fitting completed successfully"
        completed=$((completed + 1))
        
        # Calculate and display progress
        percent=$((completed * 100 / total))
        remaining=$((total - completed))
        est_minutes_remaining=$((remaining * 12))  # 12 minutes per group estimate
        est_hours_remaining=$((est_minutes_remaining / 60))
        
        echo "Progress: $completed/$total ($percent%)"
    else
        echo "Group $group failed - check logs_ASG_solo/group_${group}.log"
        echo "Continuing with next group..."
    fi
done

echo ""
echo "========================================="
echo "ALL ASG SOLO FITTING COMPLETED!"
echo "Finished at: $(date)"
echo "========================================="
