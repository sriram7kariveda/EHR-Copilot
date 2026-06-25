#!/usr/bin/env bash
# Generate Synthea test data for development
# Requires: Java 11+, git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/synthea"
SYNTHEA_DIR="/tmp/synthea"

echo "=== Synthea Test Data Generator ==="

# Clone Synthea if not present
if [ ! -d "$SYNTHEA_DIR" ]; then
    echo "Cloning Synthea..."
    git clone https://github.com/synthetichealth/synthea.git "$SYNTHEA_DIR"
fi

cd "$SYNTHEA_DIR"

# Generate a small set of patients (5 patients for dev)
echo "Generating 5 synthetic patients..."
./run_synthea -p 5 \
    --exporter.fhir.export true \
    --exporter.csv.export false \
    --exporter.years_of_history 5 \
    Massachusetts Boston

# Copy output to project data dir
mkdir -p "$DATA_DIR"
cp -r "$SYNTHEA_DIR/output/fhir/"*.json "$DATA_DIR/"

echo "Generated $(ls "$DATA_DIR"/*.json 2>/dev/null | wc -l) FHIR bundles in $DATA_DIR/"
echo "Done!"
