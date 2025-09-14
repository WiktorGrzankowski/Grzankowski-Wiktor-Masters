from __future__ import annotations
import os
import sys
import logging
from typing import List, Tuple, Set
import pandas as pd

from pabutools.election import (
    parse_pabulib,
    CardinalProfile,
    ApprovalProfile,
    OrdinalProfile,
)
from pabutools.election.satisfaction.additivesatisfaction import Additive_Cardinal_Sat
from pabutools.analysis import priceable
from pabutools.rules import method_of_equal_shares, greedy_utilitarian_welfare

from utils.utils import (
    bounded_overspending, get_election, scale_cardinal_by_cost,
    from_approval_to_cost_cardinal, candidates_to_budget_allocation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("full_methods_runner")

data_dir   = "../resources/All"
output_log = "../output/results.txt"
DEFAULT_TIMEOUT_S = 1200

PRICEABLE_KEY         = "priceable"
STABLE_PRICEABLE_KEY  = "stable priceable"
NOT_PRICEABLE_KEY     = "not priceable"
NOT_EXHAUSTIVE_KEY    = "not exhaustive"

MES_KEY               = "MES"
MES_BOS_KEY           = "MES_BOS"
MES_INC_KEY           = "MES_INC"
UGREEDY_KEY           = "UGREEDY"
MES_INC_UGREEDY_KEY   = "MES_INC_UGREEDY"

PT_CARDINAL_STD       = "cardinal_standard"
PT_CARDINAL_TIMES     = "cardinal_times_cost"
PT_APPROVAL_TIMES     = "approval_times_cost"

METHODS = [MES_KEY, MES_BOS_KEY, MES_INC_KEY, UGREEDY_KEY, MES_INC_UGREEDY_KEY]
LOG_COLUMNS = ["fname", "res_type", "ptype", "method", "res"]


def ensure_output_with_header(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)
        logger.info(f"Created output file with header: {os.path.abspath(path)}")


def load_done_fnames(output_log: str) -> Set[str]:
    if not (os.path.exists(output_log) and os.path.getsize(output_log) > 0):
        return set()
    try:
        df = pd.read_csv(output_log, dtype=str)
        if not set(LOG_COLUMNS).issubset(df.columns):
            df = pd.read_csv(output_log, header=None, names=LOG_COLUMNS)
    except Exception:
        df = pd.read_csv(output_log, header=None, names=LOG_COLUMNS)
    for c in ("fname",):
        df[c] = df[c].astype(str).str.strip()
    return set(df["fname"].unique())


def list_pb_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        logger.error(f"data_dir does not exist: {dir_path}")
        return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(".pb")])


def upsert_for_fname(output_log: str, fname: str, rows: List[List[str]]) -> None:
    ensure_output_with_header(output_log)
    try:
        df = pd.read_csv(output_log, dtype=str)
        if not set(LOG_COLUMNS).issubset(df.columns):
            df = pd.read_csv(output_log, header=None, names=LOG_COLUMNS)
    except Exception:
        df = pd.read_csv(output_log, header=None, names=LOG_COLUMNS)

    for c in LOG_COLUMNS:
        df[c] = df[c].astype(str).str.strip()

    before = len(df)
    mask_drop = (df["fname"] == fname)
    n_drop = int(mask_drop.sum())
    df = df.loc[~mask_drop].copy()

    add_df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    out = pd.concat([df, add_df], ignore_index=True)

    tmp = output_log + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, output_log)

    logger.info(f"[{fname}] write: dropped {n_drop}, added {len(add_df)}, total {before} → {len(out)}")


def compute_allocation_for_method(inst, prof, method: str):
    if method == MES_BOS_KEY:
        election = get_election(instance=inst, profile=prof, sat_class=Additive_Cardinal_Sat)
        return candidates_to_budget_allocation(bounded_overspending(election))

    if method == MES_KEY:
        return method_of_equal_shares(inst, prof, sat_class=Additive_Cardinal_Sat)

    if method == MES_INC_KEY:
        return method_of_equal_shares(inst, prof, sat_class=Additive_Cardinal_Sat, voter_budget_increment=1)

    if method == UGREEDY_KEY:
        return greedy_utilitarian_welfare(inst, prof, sat_class=Additive_Cardinal_Sat)

    if method == MES_INC_UGREEDY_KEY:
        partial = method_of_equal_shares(inst, prof, sat_class=Additive_Cardinal_Sat, voter_budget_increment=1)
        return greedy_utilitarian_welfare(inst, prof, sat_class=Additive_Cardinal_Sat,
                                          initial_budget_allocation=partial)

    raise ValueError(f"Unknown method: {method}")


def validate_label(value: bool | None) -> str | None:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return None  # timeout / not solved


def classify_allocation(fname: str, inst, prof, alloc) -> Tuple[str | None, str | None, bool]:
    try:
        is_exhaustive = inst.is_exhaustive(projects=alloc)
    except Exception:
        logger.exception(f"[{fname}] is_exhaustive failed; assuming False")
        is_exhaustive = False

    v_price = priceable(
            instance=inst, profile=prof,
            budget_allocation=alloc, stable=False, exhaustive=False,
            max_seconds=DEFAULT_TIMEOUT_S
        ).validate()
    v_stable =        priceable(
            instance=inst, profile=prof,
            budget_allocation=alloc, stable=True, exhaustive=False,
            max_seconds=DEFAULT_TIMEOUT_S
        ).validate()

    if v_price is None or v_stable is None:
        logger.warning(f"[{fname}] classification TIMEOUT")
        return "timeout", "timeout", True

    if v_stable == True:
        non_exh = STABLE_PRICEABLE_KEY
    elif v_price == True:
        non_exh = PRICEABLE_KEY
    else:
        non_exh = NOT_PRICEABLE_KEY

    exh = non_exh if is_exhaustive else NOT_EXHAUSTIVE_KEY
    logger.info(f"[{fname}] classify → non_exh='{non_exh}', exh='{exh}', exhaustive={is_exhaustive}")
    return non_exh, exh, False


def rows_for_ptype(fname: str, inst, prof, ptype_label: str) -> Tuple[List[List[str]], bool]:
    rows: List[List[str]] = []
    for m in METHODS:
        try:
            alloc = compute_allocation_for_method(inst, prof, m)
        except Exception:
            logger.exception(f"[{fname}] allocation failed for {ptype_label} @ {m}")
            return [[fname, "non exhaustive", ptype_label, m, "timeout"],
                    [fname, "exhaustive",     ptype_label, m, "timeout"]], True

        non_exh, exh, to_flag = classify_allocation(fname, inst, prof, alloc)
        if to_flag:
            rows.extend([
                [fname, "non exhaustive", ptype_label, m, "timeout"],
                [fname, "exhaustive",     ptype_label, m, "timeout"],
            ])
            return rows, True

        rows.extend([
            [fname, "non exhaustive", ptype_label, m, non_exh],
            [fname, "exhaustive",     ptype_label, m, exh],
        ])
    return rows, False


def all_rows_for_file(fname: str, inst, prof) -> List[List[str]]:
    variants: List[Tuple[str, object]] = []
    if isinstance(prof, CardinalProfile):
        variants.append((PT_CARDINAL_STD, prof))
        variants.append((PT_CARDINAL_TIMES, scale_cardinal_by_cost(prof, inst)))
    elif isinstance(prof, ApprovalProfile):
        variants.append((PT_APPROVAL_TIMES, from_approval_to_cost_cardinal(prof, inst)))
    else:
        logger.info(f"[{fname}] Unsupported profile type {type(prof).__name__}, skipping.")
        return []

    all_rows: List[List[str]] = []
    for ptype_label, variant_prof in variants:
        logger.info(f"[{fname}] processing @ {ptype_label}")
        rows, timed_out = rows_for_ptype(fname, inst, variant_prof, ptype_label)
        if timed_out:
            logger.info(f"[{fname}] timeout detected → writing TIMEOUT for all methods/ptypes")
            timeout_rows: List[List[str]] = []
            for pl, _v in variants:
                for m in METHODS:
                    timeout_rows.append([fname, "non exhaustive", pl, m, "timeout"])
                    timeout_rows.append([fname, "exhaustive",     pl, m, "timeout"])
            return timeout_rows
        all_rows.extend(rows)

    return all_rows


def main():
    logger.info(f"DATA: {os.path.abspath(data_dir)}")
    logger.info(f"OUT:  {os.path.abspath(output_log)}")
    ensure_output_with_header(output_log)

    # 1) enumerate and skip done
    all_pbs = list_pb_files(data_dir)
    if not all_pbs:
        logger.info("No .pb files found. Exiting.")
        sys.exit(0)
    already_done = load_done_fnames(output_log)
    todo = [f for f in all_pbs if f not in already_done]
    logger.info(f"Found {len(all_pbs)} files; already done: {len(already_done)}; to process: {len(todo)}")

    # 2) pass-1: size scan (skip Ordinal profiles entirely)
    sizes: List[Tuple[int, str]] = []
    errors = 0
    for idx, fname in enumerate(todo, 1):
        path = os.path.join(data_dir, fname)
        try:
            inst, prof = parse_pabulib(path)
            if isinstance(prof, OrdinalProfile):
                logger.info(f"[size] skipping Ordinal profile: {fname}")
                continue
            n = len(prof) * len(inst)
            sizes.append((int(n), fname))
        except Exception:
            errors += 1
            logger.exception(f"[size] parse failed for {fname}")
        finally:
            try:
                del inst, prof
            except Exception:
                pass
        if idx % 100 == 0:
            logger.info(f"[size] scanned {idx}/{len(todo)} …")

    sizes.sort(key=lambda t: t[0])
    logger.info(f"Size sorting complete. Beginning passes (smallest → largest).")

    # 3) pass-2: process in that order; upsert after each file
    processed = timeouts = fails = 0
    total = len(sizes)
    for pos, (n, fname) in enumerate(sizes, 1):
        logger.info(f"[{pos}/{total}] {fname} (len(prof)*len(inst)={n}) — START")
        path = os.path.join(data_dir, fname)
        try:
            inst, prof = parse_pabulib(path)
            if isinstance(prof, OrdinalProfile):
                logger.info(f"[{fname}] OrdinalProfile (late) — skipping.")
                continue

            rows = all_rows_for_file(fname, inst, prof)
            if not rows:
                logger.info(f"[{fname}] produced 0 rows (unsupported profile), skipping write.")
                continue

            # detect timeout presence in produced rows
            if any(str(r[-1]).strip().lower() == "timeout" for r in rows):
                timeouts += 1

            upsert_for_fname(output_log, fname, rows)
            processed += 1
            logger.info(f"[{fname}] DONE (written).")
        except Exception:
            fails += 1
            logger.exception(f"[{fname}] FAILED (skipped).")

        if pos % 50 == 0:
            logger.info(f"Progress: {pos}/{total} | written={processed}, timeouts={timeouts}, failed={fails}")

    logger.info(f"ALL DONE. written={processed}, timeouts={timeouts}, failed={fails}. Output → {output_log}")


if __name__ == "__main__":
    main()
