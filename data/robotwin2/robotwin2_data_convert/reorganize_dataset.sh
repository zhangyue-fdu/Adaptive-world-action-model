#!/bin/bash
# Reorganize RobotWin dataset from new structure to old structure
# New structure: task_name/aloha-agilex_clean_50/ and task_name/aloha-agilex_randomized_500/
# Old structure: clean/task_name/ and randomized/task_name/

set -e  # Exit on error

DATASET_DIR="/data/share/robotwin_dataset"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

cd "$DATASET_DIR"

echo "=========================================="
echo "Reorganizing RobotWin Dataset Structure"
echo "=========================================="
echo "Dataset directory: $DATASET_DIR"
echo ""

# Create clean and randomized directories
mkdir -p clean randomized

# Count tasks processed
TASKS_PROCESSED=0
CLEAN_EPISODES=0
RANDOMIZED_EPISODES=0

# Process each task directory
for task_dir in */; do
    task_name="${task_dir%/}"  # Remove trailing /
    
    # Skip clean and randomized directories if they already exist
    if [[ "$task_name" == "clean" || "$task_name" == "randomized" ]]; then
        echo "Skipping $task_name directory"
        continue
    fi
    
    echo "Processing task: $task_name"
    
    # Process clean split
    if [ -d "$task_name/aloha-agilex_clean_50" ]; then
        echo "  Moving clean data..."
        mkdir -p "clean/$task_name"
        
        # Move subdirectories (qpos, videos, metas, umt5_wan)
        for subdir in qpos videos metas umt5_wan; do
            if [ -d "$task_name/aloha-agilex_clean_50/$subdir" ]; then
                echo "    Moving $subdir..."
                mv "$task_name/aloha-agilex_clean_50/$subdir" "clean/$task_name/"
                # Count episodes (using qpos files as reference)
                if [ "$subdir" == "qpos" ]; then
                    count=$(ls -1 "clean/$task_name/$subdir"/*.pt 2>/dev/null | wc -l)
                    CLEAN_EPISODES=$((CLEAN_EPISODES + count))
                fi
            fi
        done
        
        # Remove empty aloha-agilex_clean_50 directory
        if [ -d "$task_name/aloha-agilex_clean_50" ]; then
            rmdir "$task_name/aloha-agilex_clean_50" 2>/dev/null || true
        fi
    else
        echo "  Warning: clean data not found for $task_name"
    fi
    
    # Process randomized split
    if [ -d "$task_name/aloha-agilex_randomized_500" ]; then
        echo "  Moving randomized data..."
        mkdir -p "randomized/$task_name"
        
        # Move subdirectories
        for subdir in qpos videos metas umt5_wan; do
            if [ -d "$task_name/aloha-agilex_randomized_500/$subdir" ]; then
                echo "    Moving $subdir..."
                mv "$task_name/aloha-agilex_randomized_500/$subdir" "randomized/$task_name/"
                # Count episodes
                if [ "$subdir" == "qpos" ]; then
                    count=$(ls -1 "randomized/$task_name/$subdir"/*.pt 2>/dev/null | wc -l)
                    RANDOMIZED_EPISODES=$((RANDOMIZED_EPISODES + count))
                fi
            fi
        done
        
        # Remove empty aloha-agilex_randomized_500 directory
        if [ -d "$task_name/aloha-agilex_randomized_500" ]; then
            rmdir "$task_name/aloha-agilex_randomized_500" 2>/dev/null || true
        fi
    else
        echo "  Warning: randomized data not found for $task_name"
    fi
    
    # Remove empty task directory if it exists and is empty
    if [ -d "$task_name" ]; then
        # Check if directory is empty (or only contains hidden files)
        if [ -z "$(ls -A "$task_name" 2>/dev/null)" ]; then
            echo "  Removing empty task directory: $task_name"
            rmdir "$task_name"
        else
            echo "  Note: Task directory $task_name still contains files/directories"
        fi
    fi
    
    TASKS_PROCESSED=$((TASKS_PROCESSED + 1))
    echo ""
done

echo "=========================================="
echo "Reorganization Complete!"
echo "=========================================="
echo "Tasks processed: $TASKS_PROCESSED"
echo "Clean episodes: $CLEAN_EPISODES"
echo "Randomized episodes: $RANDOMIZED_EPISODES"
echo ""
echo "New structure:"
echo "  clean/"
ls -1 clean/ | head -5
if [ $(ls -1 clean/ | wc -l) -gt 5 ]; then
    echo "  ... ($(ls -1 clean/ | wc -l) tasks total)"
fi
echo "  randomized/"
ls -1 randomized/ | head -5
if [ $(ls -1 randomized/ | wc -l) -gt 5 ]; then
    echo "  ... ($(ls -1 randomized/ | wc -l) tasks total)"
fi
echo "=========================================="
