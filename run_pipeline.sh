#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  bash run_pipeline.sh <csv_file> <ec_pools_dir> <pdb_dir> <task>

Arguments:
  csv_file      Path to the input CSV file (e.g. data_design/opt.csv)
  ec_pools_dir  Directory containing <ec>.fasta files
  pdb_dir       Directory containing <accession>.pdb files
  task          Pipeline task: opt or ph

Examples:
  bash run_pipeline.sh data_design/opt.csv data_design/ec_pools data_design/pdb opt
  bash run_pipeline.sh data_design/ph.csv data_design/ec_pools data_design/pdb ph
EOF
}

trim() {
	local value="$1"
	value="${value#${value%%[![:space:]]*}}"
	value="${value%${value##*[![:space:]]}}"
	printf '%s' "$value"
}

merge_results() {
	local result_root="$1"
	local merged_fasta="$result_root/result_all.fasta"
	local merged_count=0
	local missing_count=0

	mkdir -p "$result_root"
	: > "$merged_fasta"

	while IFS= read -r fasta_path; do
		[[ -z "$fasta_path" ]] && continue
		if [[ -f "$fasta_path" ]]; then
			cat "$fasta_path" >> "$merged_fasta"
			printf '\n' >> "$merged_fasta"
			merged_count=$((merged_count + 1))
		else
			echo "[WARN] Missing final FASTA during merge: $fasta_path" >&2
			missing_count=$((missing_count + 1))
		fi
	done < <(find "$result_root" -mindepth 2 -maxdepth 2 -type f -name '*.fasta' ! -name 'result_all.fasta' | sort)

	echo "[INFO] Curated merged FASTA: $merged_fasta entries=$merged_count missing=$missing_count"

	if [[ $merged_count -eq 0 ]]; then
		return 1
	fi

	return 0
}

if [[ $# -ne 4 ]]; then
	usage
	exit 1
fi

CSV_FILE="$1"
EC_POOLS_DIR="$2"
PDB_DIR="$3"
TASK="$(printf '%s' "$4" | tr '[:upper:]' '[:lower:]')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PY="$SCRIPT_DIR/pipeline.py"
SETUP_PY="$SCRIPT_DIR/setup_assets.py"

if [[ -f "$SETUP_PY" ]]; then
	echo "[INFO] Ensuring PED-Eval data and PatchEX-Design weights are installed..."
	if ! python "$SETUP_PY" --bundle ped-eval --bundle pipeline; then
		echo "[ERROR] Automatic asset setup failed." >&2
		exit 1
	fi
fi

if [[ ! -f "$CSV_FILE" ]]; then
	echo "[ERROR] CSV file not found: $CSV_FILE" >&2
	exit 1
fi

if [[ ! -d "$EC_POOLS_DIR" ]]; then
	echo "[ERROR] EC pool directory not found: $EC_POOLS_DIR" >&2
	exit 1
fi

if [[ ! -d "$PDB_DIR" ]]; then
	echo "[ERROR] PDB directory not found: $PDB_DIR" >&2
	exit 1
fi

if [[ ! -f "$PIPELINE_PY" ]]; then
	echo "[ERROR] pipeline.py not found: $PIPELINE_PY" >&2
	exit 1
fi

case "$TASK" in
	opt)
		CONFIG_FILE="$SCRIPT_DIR/pipeline_configs/config_temperature.yaml"
		TARGET_HEADER="temperature_optimum"
		RESULT_ROOT="$SCRIPT_DIR/PipelineResults/opt"
		;;
	ph)
		CONFIG_FILE="$SCRIPT_DIR/pipeline_configs/config_ph.yaml"
		TARGET_HEADER="ph"
		RESULT_ROOT="$SCRIPT_DIR/PipelineResults/ph"
		;;
	*)
		echo "[ERROR] Unsupported task: $TASK. Use 'opt' or 'ph'." >&2
		exit 1
		;;
esac

if [[ ! -f "$CONFIG_FILE" ]]; then
	echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
	exit 1
fi

header_line="$(head -n 1 "$CSV_FILE" | tr -d '\r')"
if [[ -z "$header_line" ]]; then
	echo "[ERROR] CSV file is empty: $CSV_FILE" >&2
	exit 1
fi

IFS=',' read -r -a headers <<< "$header_line"
accession_idx=-1
ec_idx=-1
target_idx=-1

for i in "${!headers[@]}"; do
	header="$(trim "${headers[$i]}")"
	case "$header" in
		accession)
			accession_idx=$i
			;;
		ec)
			ec_idx=$i
			;;
		"$TARGET_HEADER")
			target_idx=$i
			;;
	esac
done

if [[ $accession_idx -lt 0 || $ec_idx -lt 0 || $target_idx -lt 0 ]]; then
	echo "[ERROR] CSV header must contain accession, ec, and $TARGET_HEADER columns." >&2
	exit 1
fi

success_count=0
skip_count=0
fail_count=0
line_no=1

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
	line_no=$((line_no + 1))
	line="$(printf '%s' "$raw_line" | tr -d '\r')"

	[[ -z "$(trim "$line")" ]] && continue

	IFS=',' read -r -a fields <<< "$line"
	max_idx=$accession_idx
	[[ $ec_idx -gt $max_idx ]] && max_idx=$ec_idx
	[[ $target_idx -gt $max_idx ]] && max_idx=$target_idx

	if [[ ${#fields[@]} -le $max_idx ]]; then
		echo "[WARN] Skipping line $line_no: missing required columns." >&2
		skip_count=$((skip_count + 1))
		continue
	fi

	accession="$(trim "${fields[$accession_idx]}")"
	ec="$(trim "${fields[$ec_idx]}")"
	target_value="$(trim "${fields[$target_idx]}")"

	if [[ -z "$accession" || -z "$ec" || -z "$target_value" ]]; then
		echo "[WARN] Skipping line $line_no: empty accession/ec/target_value." >&2
		skip_count=$((skip_count + 1))
		continue
	fi

	pdb_file="$PDB_DIR/$accession.pdb"
	ec_pool="$EC_POOLS_DIR/$ec.fasta"

	if [[ ! -f "$pdb_file" ]]; then
		echo "[WARN] Skipping $accession: missing PDB file $pdb_file" >&2
		skip_count=$((skip_count + 1))
		continue
	fi

	if [[ ! -f "$ec_pool" ]]; then
		echo "[WARN] Skipping $accession: missing EC pool file $ec_pool" >&2
		skip_count=$((skip_count + 1))
		continue
	fi

	echo "[INFO] Running task=$TASK accession=$accession ec=$ec target=$target_value"
	if python "$PIPELINE_PY" --config "$CONFIG_FILE" --pdb "$pdb_file" --ec_pool "$ec_pool" --target_value "$target_value"; then
		success_count=$((success_count + 1))
	else
		echo "[ERROR] Pipeline failed for accession=$accession" >&2
		fail_count=$((fail_count + 1))
	fi
done < <(tail -n +2 "$CSV_FILE")

merge_failed=0
if ! merge_results "$RESULT_ROOT"; then
	merge_failed=1
fi

echo "[INFO] Completed task=$TASK success=$success_count skipped=$skip_count failed=$fail_count"

if [[ $fail_count -gt 0 ]]; then
	exit 1
fi

if [[ $success_count -eq 0 ]]; then
	exit 1
fi

if [[ $merge_failed -ne 0 ]]; then
	exit 1
fi
