from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import importlib
import sys
from pathlib import Path
import importlib.util
from unittest.mock import patch

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("ped_eval", _PKG_ROOT / "ped_eval" / "__init__.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load ped_eval package for tests")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ped_eval", _MODULE)
_SPEC.loader.exec_module(_MODULE)
evaluate_task = _MODULE.evaluate_task
build_predictor = importlib.import_module("ped_eval.predictors").build_predictor
_SETUP_SPEC = importlib.util.spec_from_file_location(
    "patchex_setup_assets_test",
    _PKG_ROOT.parent / "setup_assets.py",
)
if _SETUP_SPEC is None or _SETUP_SPEC.loader is None:
    raise RuntimeError("Failed to load setup_assets module for tests")
_SETUP_MODULE = importlib.util.module_from_spec(_SETUP_SPEC)
sys.modules.setdefault("patchex_setup_assets_test", _SETUP_MODULE)
_SETUP_SPEC.loader.exec_module(_SETUP_MODULE)


class TestPedEvalSmoke(unittest.TestCase):
    def test_end_to_end_single_accession(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,37,HE-\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n"
                "ACC1,1,A,catalytic site\n"
                "ACC1,2,C,binding site\n",
                encoding="utf-8",
            )
            (pdb_dir / "ACC1.pdb").write_text(
                "HEADER    TEST\n"
                "SEQRES   1 A    3  ALA CYS ASP\n"
                "END\n",
                encoding="utf-8",
            )
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\n"
                "ACD\n",
                encoding="utf-8",
            )

            report = evaluate_task(
                task="opt",
                result_fasta=result_dir / "curated_opt.fasta",
                dataset_csv=data_dir / "opt.csv",
                pdb_dir=pdb_dir,
                functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
            )

            self.assertEqual(report.summary["n_total"], 1)
            self.assertEqual(report.summary["n_success"], 1)
            row = report.results[0]
            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["NSSR"], 1.0)
            self.assertAlmostEqual(row["E-FSR"], 1.0)
            self.assertAlmostEqual(row["C-FSR"], 1.0)

    def test_pas_metrics_with_mock_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,40,HHH\n"
                "ACC2,1.1.1.2,50,HHH\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n"
                "ACC1,1,A,catalytic site\n",
                encoding="utf-8",
            )
            (pdb_dir / "ACC1.pdb").write_text("SEQRES   1 A    3  ALA CYS ASP\nEND\n", encoding="utf-8")
            (pdb_dir / "ACC2.pdb").write_text("SEQRES   1 A    3  ALA CYS ASP\nEND\n", encoding="utf-8")
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\nACD\n"
                ">ACC2\nACD\n",
                encoding="utf-8",
            )

            class _FakePredictor:
                def predict_many(self, sequences):
                    _ = sequences
                    return {"ACC1": 42.0, "ACC2": 49.0}

            with patch("ped_eval.api.build_predictor", return_value=_FakePredictor()):
                report = evaluate_task(
                    task="opt",
                    result_fasta=result_dir / "curated_opt.fasta",
                    dataset_csv=data_dir / "opt.csv",
                    pdb_dir=pdb_dir,
                    functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
                    enable_pas=True,
                    predictor="Seq2Topt",
                )

            self.assertEqual(report.summary["pas_predictor"], "Seq2Topt")
            self.assertAlmostEqual(report.summary["MAE"], 1.5)
            self.assertIsNotNone(report.summary["Pearson_correlation"])
            self.assertIn("mean_PAS", report.summary)

    def test_pas_constraint_rate_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,40,HHH\n"
                "ACC2,1.1.1.2,50,HHH\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n"
                "ACC1,1,A,catalytic site\n",
                encoding="utf-8",
            )
            (pdb_dir / "ACC1.pdb").write_text("SEQRES   1 A    3  ALA CYS ASP\nEND\n", encoding="utf-8")
            (pdb_dir / "ACC2.pdb").write_text("SEQRES   1 A    3  ALA CYS ASP\nEND\n", encoding="utf-8")
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\nACD\n"
                ">ACC2\nACD\n",
                encoding="utf-8",
            )

            class _FakePredictor:
                def predict_many(self, sequences):
                    _ = sequences
                    # abs errors are 12 and 14 at sigma=4.
                    # PAS values are ~0.01 and ~0.00 after rounding to 2 decimals.
                    return {"ACC1": 52.0, "ACC2": 64.0}

            with patch("ped_eval.api.build_predictor", return_value=_FakePredictor()):
                report = evaluate_task(
                    task="opt",
                    result_fasta=result_dir / "curated_opt.fasta",
                    dataset_csv=data_dir / "opt.csv",
                    pdb_dir=pdb_dir,
                    functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
                    enable_pas=True,
                    predictor="Seq2Topt",
                )

            self.assertAlmostEqual(report.summary["constraint_satisfaction_rate"], 0.5)
            self.assertEqual(report.summary["constraint_definition"], "round(PAS, 2) > 0")

    def test_opt_thermo_proxy_ivywrel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,60,HHHH\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n"
                "ACC1,1,I,catalytic site\n",
                encoding="utf-8",
            )
            (pdb_dir / "ACC1.pdb").write_text(
                "SEQRES   1 A    4  ILE VAL TYR TRP\nEND\n",
                encoding="utf-8",
            )
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\n"
                "IVYA\n",
                encoding="utf-8",
            )

            report = evaluate_task(
                task="opt",
                result_fasta=result_dir / "curated_opt.fasta",
                dataset_csv=data_dir / "opt.csv",
                pdb_dir=pdb_dir,
                functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
                enable_thermo_proxies=True,
            )

            self.assertAlmostEqual(report.results[0]["IVYWREL"], 0.75)
            self.assertEqual(report.summary["thermo_proxy_metrics"], ["IVYWREL"])
            self.assertAlmostEqual(report.summary["mean_IVYWREL"], 0.75)
            self.assertEqual(report.summary["n_IVYWREL"], 1)
            self.assertIn("IVYWREL_analysis", report.summary)
            self.assertEqual(report.summary["IVYWREL_analysis"]["target_correlation"]["n_samples"], 1)

    def test_opt_thermo_proxy_summary_analysis_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,25,HHHH\n"
                "ACC2,1.1.1.2,45,HHHH\n"
                "ACC3,1.1.1.3,60,HHHH\n"
                "ACC4,1.1.1.4,75,HHHH\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n",
                encoding="utf-8",
            )
            for accession in ("ACC1", "ACC2", "ACC3", "ACC4"):
                (pdb_dir / f"{accession}.pdb").write_text(
                    "SEQRES   1 A    4  ALA ALA ALA ALA\nEND\n",
                    encoding="utf-8",
                )
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\nAAAA\n"
                ">ACC2\nIIAA\n"
                ">ACC3\nIVYA\n"
                ">ACC4\nIVYW\n",
                encoding="utf-8",
            )

            report = evaluate_task(
                task="opt",
                result_fasta=result_dir / "curated_opt.fasta",
                dataset_csv=data_dir / "opt.csv",
                pdb_dir=pdb_dir,
                functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
                enable_thermo_proxies=True,
            )

            self.assertAlmostEqual(report.summary["mean_IVYWREL"], 0.5625)
            self.assertEqual(report.summary["n_IVYWREL"], 4)
            self.assertIn("std_IVYWREL", report.summary)
            analysis = report.summary["IVYWREL_analysis"]
            self.assertEqual(analysis["target_correlation"]["n_samples"], 4)
            self.assertGreater(analysis["target_correlation"]["pearson_r"], 0.9)
            self.assertGreater(analysis["target_correlation"]["spearman_r"], 0.9)

            bins = {row["label"]: row for row in analysis["by_target_bin"]}
            self.assertEqual(bins["20-37°C"]["count"], 1)
            self.assertAlmostEqual(bins["20-37°C"]["mean_IVYWREL"], 0.0)
            self.assertEqual(bins["37-50°C"]["count"], 1)
            self.assertAlmostEqual(bins["37-50°C"]["mean_IVYWREL"], 0.5)
            self.assertEqual(bins["50-65°C"]["count"], 1)
            self.assertAlmostEqual(bins["50-65°C"]["mean_IVYWREL"], 0.75)
            self.assertEqual(bins["65-80°C"]["count"], 1)
            self.assertAlmostEqual(bins["65-80°C"]["mean_IVYWREL"], 1.0)

    def test_thermo_proxy_rejected_for_ph(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supported for task='opt'"):
            evaluate_task(task="ph", result_fasta="dummy.fasta", enable_thermo_proxies=True)

    def test_build_predictor_uses_updated_predictor_weights_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            weights_root = repo_root / "predictor_weights"
            (weights_root / "Seq2Topt" / "code").mkdir(parents=True)
            (weights_root / "patchex_weight" / "opt").mkdir(parents=True)
            (weights_root / "patchet_pretrain_weight" / "opt").mkdir(parents=True)

            seq2_model = weights_root / "Seq2Topt" / "code" / "model_topt_window.3_r2.0.57.pth"
            patch_cfg = weights_root / "patchex_weight" / "opt" / "model_config.yaml"
            patch_weight = weights_root / "patchex_weight" / "opt" / "model.safetensors"
            patchet_cfg = weights_root / "patchet_pretrain_weight" / "opt" / "model_config.yaml"
            patchet_weight = weights_root / "patchet_pretrain_weight" / "opt" / "model.safetensors"

            for path in (seq2_model, patch_cfg, patch_weight, patchet_cfg, patchet_weight):
                path.write_text("stub", encoding="utf-8")

            with patch("ped_eval.predictors.Seq2RegressorPredictor") as seq_ctor, \
                 patch("ped_eval.predictors.PatchInferencePredictor") as patchex_ctor, \
                 patch("ped_eval.predictors.PatchETPretrainedPredictor") as patchet_ctor:
                build_predictor(task="opt", predictor_name="Seq2Topt", project_root=repo_root)
                build_predictor(task="opt", predictor_name="PatchEX", project_root=repo_root)
                build_predictor(task="opt", predictor_name="PatchET", project_root=repo_root)

            self.assertEqual(seq_ctor.call_args.kwargs["model_path"], seq2_model.resolve())
            self.assertEqual(patchex_ctor.call_args.kwargs["model_config_path"], patch_cfg.resolve())
            self.assertEqual(patchex_ctor.call_args.kwargs["weight_path"], patch_weight.resolve())
            self.assertEqual(patchex_ctor.call_args.kwargs["project_root"], repo_root.resolve())
            self.assertEqual(patchet_ctor.call_args.kwargs["model_config_path"], patchet_cfg.resolve())
            self.assertEqual(patchet_ctor.call_args.kwargs["weight_path"], patchet_weight.resolve())
            self.assertEqual(patchet_ctor.call_args.kwargs["project_root"], repo_root.resolve())

    def test_build_predictor_falls_back_to_repo_root_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "patchex_weight" / "opt").mkdir(parents=True)

            patch_cfg = repo_root / "patchex_weight" / "opt" / "model_config.yaml"
            patch_weight = repo_root / "patchex_weight" / "opt" / "model.safetensors"
            patch_cfg.write_text("stub", encoding="utf-8")
            patch_weight.write_text("stub", encoding="utf-8")

            with patch("ped_eval.predictors.PatchInferencePredictor") as patchex_ctor:
                build_predictor(task="opt", predictor_name="PatchEX", project_root=repo_root)

            self.assertEqual(patchex_ctor.call_args.kwargs["model_config_path"], patch_cfg.resolve())
            self.assertEqual(patchex_ctor.call_args.kwargs["weight_path"], patch_weight.resolve())
            self.assertEqual(patchex_ctor.call_args.kwargs["project_root"], repo_root.resolve())

    def test_evaluate_task_progress_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data_design" / "public"
            pdb_dir = data_dir / "pdb"
            ann_dir = data_dir / "functional_site_annotations" / "opt"
            result_dir = root / "RESULT" / "opt"

            pdb_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)

            (data_dir / "opt.csv").write_text(
                "accession,ec,temperature_optimum,sec_structure\n"
                "ACC1,1.1.1.1,37,HHH\n"
                "ACC2,1.1.1.2,42,HHH\n",
                encoding="utf-8",
            )
            (ann_dir / "functional_sites_long.csv").write_text(
                "accession,pos,residue,description\n",
                encoding="utf-8",
            )
            for accession in ("ACC1", "ACC2"):
                (pdb_dir / f"{accession}.pdb").write_text(
                    "SEQRES   1 A    3  ALA CYS ASP\nEND\n",
                    encoding="utf-8",
                )
            (result_dir / "curated_opt.fasta").write_text(
                ">ACC1\nACD\n"
                ">ACC2\nACD\n",
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                report = evaluate_task(
                    task="opt",
                    result_fasta=result_dir / "curated_opt.fasta",
                    dataset_csv=data_dir / "opt.csv",
                    pdb_dir=pdb_dir,
                    functional_sites_long_csv=ann_dir / "functional_sites_long.csv",
                    auto_setup_assets=False,
                    show_progress=True,
                )

            self.assertEqual(report.summary["n_success"], 2)
            progress_output = stderr.getvalue()
            self.assertIn("[PED-Eval] Progress:", progress_output)
            self.assertIn("2/2", progress_output)
            self.assertIn("current=ACC2", progress_output)

    def test_download_progress_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "asset.bin"
            payload = b"x" * 4096

            class _FakeResponse(io.BytesIO):
                def __init__(self, data: bytes) -> None:
                    super().__init__(data)
                    self.headers = {"Content-Length": str(len(data))}

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    self.close()
                    return False

            stderr = io.StringIO()
            with patch.object(_SETUP_MODULE.urllib.request, "urlopen", return_value=_FakeResponse(payload)):
                with contextlib.redirect_stderr(stderr):
                    result = _SETUP_MODULE._download_file("https://example.com/asset.bin", destination)

            self.assertEqual(result.read_bytes(), payload)
            progress_output = stderr.getvalue()
            self.assertIn("Downloading asset.bin", progress_output)
            self.assertIn("100%", progress_output)


if __name__ == "__main__":
    unittest.main()



