"""Final pass - GRU and GRU-TCN re-run under CAUSAL regimes.

WHY THIS FILE EXISTS
--------------------
`03_code/models/deep_gru_tcn.py` (first pass) cannot be used directly for the
final pass for two reasons, both verified by reading it:

  1. Lines 101-102 write `04_results/tables/p2_deep_metrics.csv` and
     `04_results/metrics/p2_deep_metrics.json` UNCONDITIONALLY. That would overwrite
     first-pass evidence and violates the project rule that no result file is ever
     overwritten; new runs write `r1_`-prefixed outputs).
  2. It emits AGGREGATE metrics only. The final-pass statistical layer needs the
     PER-OBSERVATION test-2024 predictions so paired Diebold-Mariano and day-block
     bootstrap can be run the same way `r1_j2_stats.py` does for the interval layer.

This wrapper therefore:
  * imports the model classes and the window builder FROM `deep_gru_tcn.py` by file
    path, so the architecture, seed, optimiser and window construction are provably
    identical to the first-pass model, and that file is never edited or executed as a script;
  * reads the POST- regime file explicitly, and refuses to run unless the input
    files hash to the expected causal versions;
  * writes ONLY `r1_`-prefixed outputs, and refuses to overwrite anything;
  * emits the per-horizon test-2024 prediction arrays the DM tests need.

It deliberately does NOT run Diebold-Mariano against `02_data/interim/p2_test_pred_h*.parquet`:
those GBM predictions are PRE- and comparing them to post-causal deep predictions
would reintroduce exactly the inconsistency is about. The DM tests are run in the
a separate step against the post-causal GBM predictions.

HOW TO RUN (Windows, VS Code terminal)
--------------------------------------
    cd "D:\\GHI Forecasting\\03_code"
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

    python r1\\r1_deep_causal.py --smoke        # ~2 min, writes nothing permanent
    python r1\\r1_deep_causal.py                # RUN A - 15 epochs / 150k windows. DONE 2026-08-14
    python r1\\r1_deep_causal.py --tag full --epochs 30 --subsample 0
                                                # RUN C - 30 epochs, ALL windows. ~55-70 min

RUN C exists to close the one objection that actually threatens the Table 2
claim: "you under-trained the deep models." Run A already shows GBM leading the best
deep model by 3.09-4.08 % at every horizon; run C doubles the epochs and removes the
subsample cap entirely, so the deep models get every available 2016-2022 window. If GBM
still leads at the maximal budget, the objection is closed. A smaller-budget run cannot
answer it, which is why the 10-epoch / 100k configuration first proposed here was
dropped. (If the smaller-budget point is ever wanted:
`--tag lr001cfg --epochs 10 --subsample 100000`.)

`--subsample 0` DISABLES subsampling. The count actually trained on is printed as
"train N windows" and recorded in the `n_train` column, so the record is unambiguous
however the cap was set.

Outputs (all under 04_results/, all `r1_`-prefixed, `--tag` suffixed):
    tables/r1_p2_deep_causal.csv               aggregate metrics, 2 models x 4 horizons
    tables/r1_p2_deep_pred_h{1,3,6,12}.parquet per-observation test-2024 predictions
    metrics/r1_p2_deep_causal_provenance.json  input hashes, versions, settings, timings

Return all 6 files from each run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]          #.../GHI Forecasting
CODE = REPO / "03_code"
sys.path.insert(0, str(CODE / "utils"))
sys.path.insert(0, str(CODE / "evaluation"))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import config as CFG                                 # noqa: E402
import metrics as M                                  # noqa: E402

# ---------------------------------------------------------------------------
# Input provenance. These are the POST- (causal) artifacts as they stand on
# disk on 2026-08-14. If a hash does not match, the inputs are not the ones this
# run is specified against and the script stops rather than producing a number
# nobody can trace.
# ---------------------------------------------------------------------------
EXPECTED_SHA256 = {
    "02_data/regime_labels/yulara_regimes_5min.parquet":
        "13d9659d90dab8242049e8858523e46fd0dad708eb29ca32138e0d9070c57e6b",
    "02_data/cleaned/yulara_clean_5min.parquet":
        "5dfe4445c8ab12b0a5ea889dfe7921c6496d2e79fb37dad421feeccfdec7bf39",
    "02_data/cleaned/yulara_quality_flags.parquet":
        "4e9e40a96f99d1b9e2510d569d1ce7524830e97fc0171b9675782310cfdfc275",
}
# The PRE- file, kept only as a backup. Running against it is the exact error
# this run exists to prevent, so it is named here and rejected explicitly.
FORBIDDEN_SHA256 = {
    "c5356d70f000fe7c3ae949aeee721e479ecc87e8983c7438d3d86330b949428d":
        "yulara_regimes_5min_centered_backup.parquet (PRE-, centered clear-sky)",
}

R1_PREFIX = "r1_"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WriteGuard(Exception):
    pass


def guarded_path(path: Path, force: bool) -> Path:
    """Every output must be r1_-prefixed and must not silently overwrite."""
    if not path.name.startswith(R1_PREFIX):
        raise WriteGuard(
            f"REFUSING to write '{path.name}': project rule - a result file is never overwritten; "
            f"'{R1_PREFIX}' prefix on every output of this phase."
        )
    if path.exists() and not force:
        raise WriteGuard(
            f"REFUSING to overwrite existing file:\n  {path}\n"
            "Move or rename it first, or re-run with --force if you are certain."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def out_names(tag: str) -> dict:
    s = f"_{tag}" if tag else ""
    return {"csv": f"{R1_PREFIX}p2_deep_causal{s}.csv",
            "json": f"{R1_PREFIX}p2_deep_causal{s}_provenance.json",
            "pred": lambda h: f"{R1_PREFIX}p2_deep_pred_h{h}{s}.parquet"}


def preflight_outputs(tab_dir: Path, met_dir: Path, horizons, force: bool, tag: str = "") -> None:
    """Check EVERY output path before a single epoch is trained, so a name clash
    costs zero minutes instead of failing after the first horizon."""
    N = out_names(tag)
    planned = [tab_dir / N["csv"], met_dir / N["json"]]
    planned += [tab_dir / N["pred"](h) for h in horizons]
    blocked = []
    for p in planned:
        try:
            guarded_path(p, force)
        except WriteGuard as e:
            blocked.append(str(e))
    if blocked:
        raise SystemExit("Pre-flight output check FAILED (nothing was trained):\n\n"
                         + "\n\n".join(blocked))
    print(f"outputs: {len(planned)} paths pre-checked, all clear.", flush=True)


def load_reference_module():
    """Import deep_gru_tcn.py by path. It is NEVER edited and NEVER run as a script;
    its `main()` guard means importing it has no side effect beyond thread setup."""
    src = CODE / "models" / "deep_gru_tcn.py"
    if not src.exists():
        raise SystemExit(f"Cannot find {src}")
    spec = importlib.util.spec_from_file_location("deep_gru_tcn_ref", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256_of(src)


def check_provenance(allow_mismatch: bool) -> dict:
    prov = {}
    problems = []
    for rel, expected in EXPECTED_SHA256.items():
        p = REPO / rel
        if not p.exists():
            problems.append(f"MISSING: {rel}")
            continue
        got = sha256_of(p)
        prov[rel] = {
            "sha256": got,
            "expected_sha256": expected,
            "match": got == expected,
            "bytes": p.stat().st_size,
            "mtime_utc": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        }
        if got in FORBIDDEN_SHA256:
            problems.append(
                f"FORBIDDEN INPUT: {rel} currently holds {FORBIDDEN_SHA256[got]}. "
                "This is the pre- file. Restore the causal version before running."
            )
        elif got != expected:
            problems.append(f"HASH MISMATCH: {rel}\n    expected {expected}\n    got      {got}")

    if problems:
        msg = "Input provenance check FAILED:\n  - " + "\n  - ".join(problems)
        if not allow_mismatch:
            raise SystemExit(
                msg
                + "\n\nStopping. These hashes pin the post-causal inputs that this run is\n"
                  "specified against. If the files legitimately changed, re-run with\n"
                  "--allow-provenance-mismatch and SAY SO when you return the outputs, so the\n"
                  "mismatch is recorded rather than silently absorbed."
            )
        print("WARNING (continuing under --allow-provenance-mismatch):\n" + msg, flush=True)
    else:
        print("Input provenance OK - all three inputs are the expected post- files.", flush=True)
    return prov


def build_base() -> pd.DataFrame:
    """Exactly the frame deep_gru_tcn.main() builds, but with explicit paths."""
    clean = pd.read_parquet(CFG.DATA_CLEAN / "yulara_clean_5min.parquet")
    reg = pd.read_parquet(CFG.DATA_REGIME / "yulara_regimes_5min.parquet")
    fl = pd.read_parquet(CFG.DATA_CLEAN / "yulara_quality_flags.parquet")
    base = clean.join(reg[["ghi_cs", "kt", "is_day", "regime"]])
    base["ghi_imputed"] = fl["ghi_imputed"]
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="GRU / GRU-TCN under causal regimes")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--subsample", type=int, default=150000,
                    help="cap on training windows; pass 0 to DISABLE subsampling and train on every "
                         "available window. The count actually used is recorded in the n_train column "
                         "and printed as 'train N windows'.")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12])
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, 20k samples, h=1 only; writes to 04_results/_smoke/ and "
                         "touches no real output file.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, e.g. --tag lr001cfg. Lets a second "
                         "training-budget configuration be run without colliding with the first.")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing r1_ output")
    ap.add_argument("--allow-provenance-mismatch", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.epochs, a.subsample, a.horizons = 2, 20000, [1]

    t_start = time.time()
    print(f"repo   : {REPO}")
    print(f"python : {platform.python_version()}  ({platform.system()} {platform.machine()})")

    prov_inputs = check_provenance(a.allow_provenance_mismatch)

    try:
        ref, ref_sha = load_reference_module()
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Could not import the reference model module ({e}).\n"
            "If this is torch, install the CPU wheel first:\n"
            "    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch  # noqa: E402  (only after the friendly error above)
    print(f"torch  : {torch.__version__}")
    print(f"models : deep_gru_tcn.py sha256 {ref_sha[:16]}...  (imported read-only, never edited)")
    print(f"window : W = {ref.W} steps ({ref.W * 5} min)")

    tab_dir = (CFG.RESULTS / "_smoke") if a.smoke else CFG.TAB
    met_dir = (CFG.RESULTS / "_smoke") if a.smoke else CFG.MET
    NAMES = out_names(a.tag)
    preflight_outputs(tab_dir, met_dir, a.horizons, a.force or a.smoke, a.tag)

    base = build_base()
    print(f"base   : {len(base):,} rows, {base.index.min()} -> {base.index.max()}", flush=True)

    rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)  # noqa: E731
    rows, per_h_files = [], []

    for h in a.horizons:
        X, ykt, ycs, ygh, rg, yy, ts = ref.build_windows(base, h)
        tr = yy <= 2022
        te = yy == 2024
        Xtr, ytr = X[tr], ykt[tr]
        if a.subsample and tr.sum() > a.subsample:
            sel = np.random.RandomState(42).choice(np.where(tr)[0], a.subsample, replace=False)
            Xtr, ytr = X[sel], ykt[sel]
        n_train = int(len(Xtr))   # windows ACTUALLY trained on, whatever --subsample said
        Xte = X[te]
        y = ygh[te]
        cs = ycs[te]
        reg_te = rg[te]
        tste = pd.to_datetime(ts[te])
        _cap = f"{a.subsample:,} cap" if a.subsample else "NO subsampling - all available"
        print(f"\nh={h * 5:>2} min : train {n_train:,} windows ({_cap}) | "
              f"test-2024 {len(Xte):,} windows", flush=True)

        preds = {}
        for name, Net in [("gru", ref.GRUNet), ("gru_tcn", ref.GRUTCN)]:
            t0 = time.time()
            kt_hat = ref.train_eval(Net(), Xtr, ytr, Xte, a.epochs)
            pred = rec(kt_hat, cs)
            preds[name] = (pred, kt_hat)
            m = M.all_metrics(y, pred)
            m.update(model=name, horizon_min=h * 5, n=int(len(y)),
                     train_s=round(time.time() - t0, 1),
                     epochs=a.epochs, subsample_arg=int(a.subsample), n_train=n_train,
                     regime_basis="causal_D012")
            rows.append(m)
            print(f"  {name:<8} RMSE={m['RMSE']:8.3f}  MAE={m['MAE']:8.3f}  "
                  f"R2={m['R2']:.4f}  ({m['train_s']}s)", flush=True)

        # ---- per-observation test-2024 predictions: what the DM tests actually need
        out = pd.DataFrame(
            {
                "y_ghi": y.astype("float64"),
                "y_kt": ykt[te].astype("float64"),
                "ghi_cs": cs.astype("float64"),
                "regime": pd.Series(reg_te).astype(str).values,
                "pred_gru": preds["gru"][0].astype("float64"),
                "pred_gru_tcn": preds["gru_tcn"][0].astype("float64"),
                "kt_hat_gru": preds["gru"][1].astype("float64"),
                "kt_hat_gru_tcn": preds["gru_tcn"][1].astype("float64"),
            },
            index=pd.DatetimeIndex(tste, name="timestamp"),
        ).sort_index()
        fp = guarded_path(tab_dir / NAMES["pred"](h), a.force or a.smoke)
        out.to_parquet(fp)
        per_h_files.append(str(fp.relative_to(REPO)))
        print(f"  wrote {fp.relative_to(REPO)}  ({len(out):,} rows)", flush=True)

    df = pd.DataFrame(rows)[
        ["model", "horizon_min", "n", "RMSE", "MAE", "nRMSE", "R2",
         "train_s", "epochs", "n_train", "subsample_arg", "regime_basis"]
    ]
    csv_fp = guarded_path(tab_dir / NAMES["csv"], a.force or a.smoke)
    df.to_csv(csv_fp, index=False)

    prov = {
        "run_id": "deep_causal",
        "purpose": "GRU / GRU-TCN under causal regimes for Table 2 consistency",
        "finished_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": round(time.time() - t_start, 1),
        "smoke": bool(a.smoke),
        "args": vars(a),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "reference_module": {"path": "03_code/models/deep_gru_tcn.py", "sha256": ref_sha,
                             "edited": False, "window_steps": int(ref.W)},
        "inputs": prov_inputs,
        "outputs": [str(csv_fp.relative_to(REPO))] + per_h_files,
        "dm_tests": "NOT run here - the in-repo GBM test predictions are pre-. "
                    "Paired DM vs the post-causal GBM is run separately.",
    }
    json_fp = guarded_path(met_dir / NAMES["json"], a.force or a.smoke)
    json.dump(prov, open(json_fp, "w"), indent=2, default=str)

    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    if a.smoke:
        print("\nSMOKE TEST PASSED. Nothing permanent was written "
              "(outputs went to 04_results/_smoke/).")
        print("Now run the full job:   python r1\\r1_deep_causal.py")
    else:
        print(f"\nDONE in {prov['elapsed_s'] / 60:.1f} min. RETURN THESE FILES:")
        for f in prov["outputs"] + [str(json_fp.relative_to(REPO))]:
            print("   " + f)
        print("\nDone.")


if __name__ == "__main__":
    main()
