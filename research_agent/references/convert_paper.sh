#!/usr/bin/env bash
# Run with the separate Marker environment activated.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 path/to/paper.pdf" >&2
    exit 1
fi
if ! command -v marker_single >/dev/null 2>&1; then
    echo "Activate the Marker environment and install marker-pdf first." >&2
    exit 1
fi

references_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_dir="$(cd "$(dirname "$1")" && pwd)"
filename="$(basename "$1")"
input_pdf="$input_dir/$filename"
if [[ ! -f "$input_pdf" || "${filename##*.}" != "pdf" ]]; then
    echo "Provide an existing .pdf file." >&2
    exit 1
fi
paper_name="${filename%.pdf}"
paper_dir="$references_dir/$paper_name"
mkdir -p "$paper_dir"
destination="$paper_dir/$filename"
if [[ "$input_pdf" != "$destination" ]]; then
    if [[ -e "$destination" ]]; then
        echo "Refusing to overwrite $destination" >&2
        exit 1
    fi
    mv "$input_pdf" "$destination"
fi

# Marker creates <output_dir>/<paper_name>, matching the folder above.
marker_single "$destination" --output_dir "$references_dir"
echo "Paper and conversion output: $paper_dir"
