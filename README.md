# PatchEX-Design

PatchEX-Design is an end-to-end computational pipeline for property-conditioned enzyme design. Given a protein backbone structure, an EC-restricted enzyme sequence pool, and a target operating condition, it designs full-length enzyme sequences that aim to preserve scaffold compatibility and catalytic function while moving toward a specified optimal temperature or optimal pH.

This repository also includes `PED-Eval`, an evaluation package for property-conditioned enzyme design. PED-Eval evaluates designed sequences with recovery, functional-site preservation, optional structural metrics, and optional target-property metrics based on external predictors.

## Repository Layout

```text
PatchEX-Design/
|-- pipeline.py                         # single-target PatchEX-Design pipeline
|-- run_pipeline.sh                     # batch runner for opt/ph CSV files
|-- pipeline_configs/
|   |-- config_temperature.yaml
|   `-- config_ph.yaml
|-- MapDiff/                            # inverse-folding model code
|-- models/                             # PatchEX/PatchET model definitions
|-- direct_evolution/                   # oracle-guided optimization code
|-- PED-Eval/                           # installable PED-Eval package
|-- data_design/                        # PED-Eval/design data after dataset unzip
|-- predictor_weights/                  # PED-Eval predictor weights after unzip
|-- patchex_weight/                     # PatchEX-Design oracle weights after unzip
`-- esm150/                             # local ESM model files after unzip
```

## What the Pipeline Does

PatchEX-Design runs three stages:

1. Inverse folding with MapDiff to obtain backbone-compatible residue logits.
2. EC-restricted evolutionary refinement with PSI-BLAST/PSSM information and PatchEX patch-importance weights.
3. Oracle-guided directed evolution over selected mutable residues to optimize toward the requested temperature or pH.

PED-Eval then evaluates designs from complementary perspectives: sequence recovery, secondary-structure recovery, native sequence similarity recovery (NSSR), functional-site recovery (E-FSR/C-FSR), optional structural metrics, optional PAS-style target achievement, and optional temperature-composition proxy metrics.

## Requirements

Recommended environment:

- Python 3.9 or later
- CUDA-capable PyTorch environment for practical pipeline runtime
- BLAST+ with `psiblast` on `PATH`
- TMalign on `PATH` for structure-comparison metrics
- Bash, WSL, Git Bash, or Linux/macOS shell for `run_pipeline.sh`

Install the main pipeline dependencies:

```bash
pip install -r requirements.txt
```

Install PED-Eval:

```bash
pip install -e ./PED-Eval
```

For optional PED-Eval predictor and ESMFold structure metrics, install the optional dependencies:

```bash
pip install -e "./PED-Eval[predictors,structure]"
```

## Download Data and Weights

Download the PED-Eval dataset and predictor weights from Zenodo:

[https://doi.org/10.5281/zenodo.19992035](https://doi.org/10.5281/zenodo.19992035)

Unzip the downloaded zip files into the project root first. After extraction, the project root should contain at least:

```text
data_design/
|-- opt.csv
|-- ph.csv
|-- pdb/
|-- ec_pools/
`-- functional_site_annotations/

predictor_weights/
|-- ephod/
|-- patchex_weight/
`-- patchet_pretrain_weight/
```

Then download the PatchEX-Design pipeline weights from Zenodo:

[https://doi.org/10.5281/zenodo.19992598](https://doi.org/10.5281/zenodo.19992598)

Unzip the pipeline-weight zip file into the project root. Move the MapDiff checkpoint into the `MapDiff` directory:

```bash
mv mapdiff_weight.pt MapDiff/
```

The pipeline expects these paths:

```text
MapDiff/mapdiff_weight.pt
patchex_weight/opt/model_config.yaml
patchex_weight/opt/model.safetensors
patchex_weight/ph/model_config.yaml
patchex_weight/ph/model.safetensors
esm150/
```

PED-Eval predictor mode expects predictor weights under:

```text
predictor_weights/ephod/
predictor_weights/patchex_weight/
predictor_weights/patchet_pretrain_weight/
```

## Run PatchEX-Design

### Single Enzyme: Optimal Temperature

```bash
python pipeline.py \
  --config pipeline_configs/config_temperature.yaml \
  --pdb data_design/pdb/Q96552.pdb \
  --ec_pool data_design/ec_pools/2.5.1.6.fasta \
  --target_value 37.0
```

### Single Enzyme: Optimal pH

```bash
python pipeline.py \
  --config pipeline_configs/config_ph.yaml \
  --pdb data_design/pdb/P21673.pdb \
  --ec_pool data_design/ec_pools/2.3.1.57.fasta \
  --target_value 7.47
```

### Batch Design

The batch runner reads a CSV, resolves each accession to `data_design/pdb/<accession>.pdb`, resolves each EC number to `data_design/ec_pools/<ec>.fasta`, and runs the matching task config.

```bash
bash run_pipeline.sh data_design/opt.csv data_design/ec_pools data_design/pdb opt
bash run_pipeline.sh data_design/ph.csv data_design/ec_pools data_design/pdb ph
```

Input CSV requirements:

- `opt` task: columns `accession`, `ec`, `temperature_optimum`
- `ph` task: columns `accession`, `ec`, `ph`

## PatchEX-Design Outputs

Single-run outputs are written to:

```text
PipelineResults/<task>/<accession>/
```

Typical files:

```text
ipf_logit.pt
seed.pssm
psiblast_report.txt
evolutionary_backbone_logits.npy
<accession>.fasta
<accession>_pred.pdb
report.json
```

Batch runs also merge final FASTA files into:

```text
PipelineResults/opt/result_all.fasta
PipelineResults/ph/result_all.fasta
```

These merged FASTA files can be used directly as PED-Eval inputs.

## Run PED-Eval

PED-Eval can evaluate any FASTA whose record IDs match accessions in the dataset CSV.

### Basic Evaluation

```bash
ped-eval \
  --task opt \
  --result-fasta PipelineResults/opt/result_all.fasta \
  --output-dir PED-Eval/outputs/opt

ped-eval \
  --task ph \
  --result-fasta PipelineResults/ph/result_all.fasta \
  --output-dir PED-Eval/outputs/ph
```

By default PED-Eval reads:

```text
data_design/<task>.csv
data_design/pdb/<accession>.pdb
data_design/functional_site_annotations/<task>/functional_sites_long.csv
```

Use `--data-path` or explicit path overrides if your dataset is elsewhere:

```bash
ped-eval \
  --task opt \
  --result-fasta PipelineResults/opt/result_all.fasta \
  --data-path data_design \
  --output-dir PED-Eval/outputs/opt
```

### Optional PAS Metrics

Enable PAS-style target-property metrics with a predictor and the downloaded `predictor_weights` directory.

Temperature predictors:

- `Seq2Topt`
- `PatchEX`
- `PatchET`

pH predictors:

- `Seq2pHopt`
- `PatchEX`
- `EpHod`

Examples:

```bash
ped-eval \
  --task opt \
  --result-fasta PipelineResults/opt/result_all.fasta \
  --output-dir PED-Eval/outputs/opt_pas \
  --enable-pas \
  --predictor PatchEX \
  --predictor-weights-dir predictor_weights

ped-eval \
  --task ph \
  --result-fasta PipelineResults/ph/result_all.fasta \
  --output-dir PED-Eval/outputs/ph_pas \
  --enable-pas \
  --predictor EpHod \
  --predictor-weights-dir predictor_weights
```

Defaults:

- PAS sigma is `4.0` for `opt`.
- PAS sigma is `0.3` for `ph`.
- Constraint satisfaction is reported using `round(PAS, 2) > 0`.

### Optional Structure Metrics

Use ESMFold-based structure evaluation when the required dependencies and GPU memory are available:

```bash
ped-eval \
  --task opt \
  --result-fasta PipelineResults/opt/result_all.fasta \
  --output-dir PED-Eval/outputs/opt_structure \
  --use-esmfold
```

### Optional Thermophilic Composition Proxy

For the temperature task, PED-Eval can compute the IVYWREL composition proxy:

```bash
ped-eval \
  --task opt \
  --result-fasta PipelineResults/opt/result_all.fasta \
  --output-dir PED-Eval/outputs/opt_thermo \
  --enable-thermo-proxies
```

## PED-Eval Outputs

Each PED-Eval run writes:

```text
per_accession.jsonl   # full per-accession metric objects
per_accession.csv     # flattened table for analysis
summary.json          # aggregate means, success/error counts, optional PAS summaries
```

## Python API

```python
from ped_eval import evaluate_task, write_report

report = evaluate_task(
    task="opt",
    result_fasta="PipelineResults/opt/result_all.fasta",
    data_path="data_design",
    enable_pas=True,
    predictor="PatchEX",
    predictor_weights_dir="predictor_weights",
)

write_report(report, "PED-Eval/outputs/opt_api")
print(report.summary)
```

## Troubleshooting

- If `psiblast` is not found, install BLAST+ and ensure it is on `PATH`.
- If `MapDiff/mapdiff_weight.pt` is missing, move `mapdiff_weight.pt` from the pipeline-weight extraction into `MapDiff/`.
- If PatchEX-Design cannot load oracle weights, check `patchex_weight/opt/` and `patchex_weight/ph/`.
- If PED-Eval PAS mode cannot load a predictor, check `predictor_weights/` and install `PED-Eval[predictors]`.
- If `run_pipeline.sh` fails on Windows PowerShell, run it from WSL or Git Bash.

## Citation

If you use this repository, please cite the PatchEX-Design paper:

```text
PatchEX-Design: End-to-End Computational Pipeline for Property-Conditioned Enzyme Design
```
