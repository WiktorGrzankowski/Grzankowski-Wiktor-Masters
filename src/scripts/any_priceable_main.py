from __future__ import annotations
import os
import csv
import logging
from typing import Tuple, Set, List

import pandas as pd

from pabutools.election import parse_pabulib
from pabutools.election.profile.ordinalprofile import OrdinalProfile
from pabutools.analysis import priceable
from utils.utils import (
    scale_cardinal_by_cost,
    from_approval_to_cost_cardinal,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = "../resources/All"
SOURCE_LOG  = "../output/results.txt"

STABLE      = False
EXHAUSTIVE  = True
SOLVER_SECS = 1500

MODE_LABEL  = "stable priceable" if STABLE else "priceable"
OUT_FILE    = os.path.join(
    BASE_DIR, "../output",
    "STABLE_YES_EXH_YES.txt" if (STABLE and EXHAUSTIVE) else
    "STABLE_YES_EXH_NO.txt"  if (STABLE and not EXHAUSTIVE) else
    "STABLE_NO_EXH_YES.txt"  if (not STABLE and EXHAUSTIVE) else
    "STABLE_NO_EXH_NO.txt"
)
OUT_COLS    = ["fname", "ptype", "exhaustive", "mode", "exists"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("select_and_check_any_priceable")


def ensure_output_file(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(OUT_COLS)
        log.info(f"Created output file with header: {os.path.abspath(path)}")

def read_output_df(path: str) -> pd.DataFrame:
    ensure_output_file(path)
    df = pd.read_csv(path, dtype=str)
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[OUT_COLS]

def upsert_pair(path: str, row: dict) -> None:
    df = read_output_df(path)
    mask = (df["fname"] == row["fname"]) & (df["ptype"] == row["ptype"])
    df = df.loc[~mask].copy()
    row_s = {k: (str(v) if v is not None else "") for k, v in row.items()}
    df = pd.concat([df, pd.DataFrame([row_s])], ignore_index=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log.info(f"Upserted → {row['fname']},{row['ptype']} ({row['mode']}, exh={row['exhaustive']})")

def done_pairs_in_output(path: str) -> Set[Tuple[str, str]]:
    df = read_output_df(path)
    s = set()
    for _, r in df.iterrows():
        ev = str(r["exists"]).strip()
        if ev in {"True", "False", "timeout"}:  # definitive states
            s.add((r["fname"], r["ptype"]))
    return s


def load_source_df(path: str) -> pd.DataFrame:
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise FileNotFoundError(f"SOURCE_LOG missing or empty: {path}")
    df = pd.read_csv(path, dtype=str)
    # normalize
    for c in ["fname", "res_type", "ptype", "method", "res"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df[["fname", "res_type", "ptype", "method", "res"]]

def select_pairs_to_process(df: pd.DataFrame, stable: bool, exhaustive: bool) -> List[Tuple[str, str]]:
    res_type = "exhaustive" if exhaustive else "non exhaustive"
    sub = df[df["res_type"] == res_type].copy()

    if sub.empty:
        return []

    allowed = {"not priceable", "not exhaustive"} | ({"priceable"} if stable else set())

    # group and test the condition
    grp = (
        sub.groupby(["fname", "ptype"])["res"]
           .apply(list)
           .reset_index()
    )

    def all_allowed(lst: List[str]) -> bool:
        return set(map(str.lower, lst)).issubset(allowed)

    sel = grp[grp["res"].apply(all_allowed)]
    return list(sel[["fname", "ptype"]].itertuples(index=False, name=None))


def normalize_exists(val) -> str:
    if val is True:
        return "True"
    if val is False:
        return "False"
    return "timeout"  # None / not solved → treat as timeout

def check_pair(fname: str, ptype: str) -> str:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        log.warning(f"[MISSING] {path}")
        return "timeout"

    inst, prof = parse_pabulib(path)

    if isinstance(prof, OrdinalProfile):
        log.info(f"[SKIP ordinal] {fname}")
        return "timeout"

    if ptype == "cardinal_times_cost":
        prof = scale_cardinal_by_cost(prof, inst)
    elif ptype == "approval_times_cost":
        prof = from_approval_to_cost_cardinal(prof, inst)
    elif ptype == "cardinal_standard":
        pass
    else:
        log.warning(f"[SKIP unknown ptype] {fname} @ {ptype}")
        return "timeout"

    res = priceable(
        instance=inst,
        profile=prof,
        exhaustive=EXHAUSTIVE,
        stable=STABLE,
        max_seconds=SOLVER_SECS,
    )
    ok = res.validate()  # True / False / None
    log.info(f"[{fname} @ {ptype}] priceable({MODE_LABEL}, exh={EXHAUSTIVE}) → {ok}")
    return normalize_exists(ok)


def main():
    log.info(f"SOURCE_LOG: {os.path.abspath(SOURCE_LOG)}")
    log.info(f"DATA_DIR:   {os.path.abspath(DATA_DIR)}")
    log.info(f"OUTPUT:     {os.path.abspath(OUT_FILE)}")
    log.info(f"MODE={MODE_LABEL}  EXHAUSTIVE={EXHAUSTIVE}  SOLVER_SECS={SOLVER_SECS}")

    ensure_output_file(OUT_FILE)

    # 1) Read source and select pairs to process
    src = load_source_df(SOURCE_LOG)
    candidates = select_pairs_to_process(src, STABLE, EXHAUSTIVE)
    log.info(f"Candidates (after filtering by results): {len(candidates)}")

    if not candidates:
        log.info("Nothing to do. Bye.")
        return

    # 2) Skip pairs already present with a definitive result in OUT_FILE
    already = done_pairs_in_output(OUT_FILE)
    todo = [(f, p) for (f, p) in candidates if (f, p) not in already]
    skipped = len(candidates) - len(todo)
    log.info(f"Already in output (skipped): {skipped}")
    if skipped:
        show = ", ".join([f"{f}:{p}" for (f, p) in sorted(already)][:20])
        log.info(f"Examples already done: {show}{' ...' if len(already) > 20 else ''}")

    log.info(f"To process now: {len(todo)}")

    # 3) Process pairs and upsert immediately
    for idx, (fname, ptype) in enumerate(todo, 1):
        log.info(f"[{idx}/{len(todo)}] @ ({fname}, {ptype})")
        exists = check_pair(fname, ptype)
        upsert_pair(OUT_FILE, {
            "fname": fname,
            "ptype": ptype,
            "exhaustive": str(EXHAUSTIVE),
            "mode": MODE_LABEL,
            "exists": exists,
        })

    log.info("Done.")

if __name__ == "__main__":
    main()
