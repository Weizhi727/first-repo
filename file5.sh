#!/usr/bin/env bash
# file5.sh - List files in the repository with their sizes

# TODO: Show a summary line with the total file count at the end

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Files in repository: $REPO_DIR"
echo "-----------------------------------"

count=0
while IFS= read -r -d '' file; do
    size=$(wc -c < "$file")
    printf "  %-20s  %5d bytes\n" "$(basename "$file")" "$size"
    count=$((count + 1))
done < <(find "$REPO_DIR" -maxdepth 1 -type f ! -name ".*" -print0 | sort -z)

echo "-----------------------------------"
# Implemented TODO: display total file count
echo "Total: $count file(s)"
