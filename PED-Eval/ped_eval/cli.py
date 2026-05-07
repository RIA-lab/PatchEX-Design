from __future__ import annotations

import argparse
from pathlib import Path

from .api import evaluate_task, write_report
from .predictors import allowed_predictors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PED-Eval v1: evaluate designed enzyme sequences")
    parser.add_argument("--task", required=True, choices=["opt", "ph"], help="Task name")
    parser.add_argument("--result-fasta", required=True, help="Path to curated_<task>.fasta")

    # Path overrides
    parser.add_argument(
        "--data-path",
        default=None,
        help="Base directory containing dataset CSV, PDB files, and functional site annotations. "
             "Overrides individual --dataset-csv, --pdb-dir, and --sites-long-csv defaults.",
    )
    parser.add_argument("--dataset-csv", default=None, help="Override dataset CSV path")
    parser.add_argument("--pdb-dir", default=None, help="Override native PDB directory")
    parser.add_argument("--sites-long-csv", default=None, help="Override functional site long CSV path")

    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Optional ESM-fold structure evaluation
    parser.add_argument(
        "--use-esmfold",
        action="store_true",
        default=False,
        help="Fold each predicted sequence with ESMFold and compute mean_plddt, TM-score, and RMSD.",
    )
    parser.add_argument(
        "--esmfold-output-dir",
        default=None,
        help="Directory to write predicted PDB files (default: <output-dir>/esmfold_pdbs).",
    )

    # Optional PAS-style metrics
    parser.add_argument(
        "--enable-pas",
        action="store_true",
        default=False,
        help="Enable PAS-style metrics (PAS, constraint satisfaction rate, MAE, Pearson correlation).",
    )
    parser.add_argument(
        "--enable-thermo-proxies",
        action="store_true",
        default=False,
        help="Enable thermophilic composition proxy metrics for the opt task (currently: IVYWREL).",
    )
    parser.add_argument(
        "--predictor",
        default=None,
        help="Predictor for PAS metrics. opt: Seq2Topt|PatchEX|PatchET; ph: Seq2pHopt|PatchEX|EpHod.",
    )
    parser.add_argument(
        "--pas-sigma",
        type=float,
        default=None,
        help="PAS sigma. Defaults to 4.0 for opt and 0.3 for ph.",
    )
    parser.add_argument(
        "--predictor-weights-dir",
        default=None,
        help="Directory containing predictor weights (default: ./predictor_weights).",
    )
    parser.add_argument(
        "--predictor-project-root",
        default=None,
        help="Deprecated alias used to derive weights dir as <project-root>/predictor_weights.",
    )
    parser.add_argument(
        "--predictor-batch-size",
        type=int,
        default=4,
        help="Batch size for Seq2Topt/Seq2pHopt predictor inference.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.predictor is not None:
        allowed = allowed_predictors(args.task)
        if args.predictor not in allowed:
            raise ValueError(f"Invalid predictor '{args.predictor}' for task '{args.task}'. Allowed: {allowed}")
    if args.enable_thermo_proxies and args.task != "opt":
        raise ValueError("--enable-thermo-proxies is only supported for --task opt.")

    esmfold_out = args.esmfold_output_dir or (str(Path(args.output_dir) / "esmfold_pdbs") if args.use_esmfold else None)

    report = evaluate_task(
        task=args.task,
        result_fasta=args.result_fasta,
        dataset_csv=args.dataset_csv,
        pdb_dir=args.pdb_dir,
        functional_sites_long_csv=args.sites_long_csv,
        data_path=args.data_path,
        use_esmfold=args.use_esmfold,
        esmfold_output_dir=esmfold_out,
        enable_pas=args.enable_pas,
        enable_thermo_proxies=args.enable_thermo_proxies,
        predictor=args.predictor,
        pas_sigma=args.pas_sigma,
        predictor_weights_dir=args.predictor_weights_dir,
        predictor_project_root=args.predictor_project_root,
        predictor_batch_size=args.predictor_batch_size,
    )
    output_paths = write_report(report, Path(args.output_dir))

    print(f"[PED-Eval] task={report.task}")
    print(f"[PED-Eval] n_total={report.summary['n_total']} n_success={report.summary['n_success']} n_error={report.summary['n_error']}")
    print(f"[PED-Eval] wrote CSV:     {output_paths['csv']}")
    print(f"[PED-Eval] wrote JSONL:   {output_paths['jsonl']}")
    print(f"[PED-Eval] wrote summary: {output_paths['summary']}")


if __name__ == "__main__":
    main()

