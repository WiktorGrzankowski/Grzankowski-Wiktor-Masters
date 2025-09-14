import logging
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

LOG_FILE      = "../output/results.txt"
PROFILE_TYPES = ['cardinal_standard', 'cardinal_times_cost', 'approval_times_cost']
METHODS       = ['MES', 'MES_BOS', 'MES_INC', 'UGREEDY', 'MES_INC_UGREEDY']
RESULT_KEYS   = ['stable priceable', 'priceable', 'not priceable', 'not exhaustive']
COLORS = {
    'stable priceable': '#0072B2',
    'priceable':        '#D55E00',
    'not exhaustive':   '#CC79A7',
    'not priceable':    '#009E73',
    'any stable':       '#0072B2',
    'any priceable':    '#D55E00',
    'none priceable':   '#CCCCCC',
    'timeout': '#F0E442',
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_csv(path):
    df = pd.read_csv(path)

    results      = {pt: {m: defaultdict(int) for m in METHODS} for pt in PROFILE_TYPES}
    exh_results  = {pt: {m: defaultdict(int) for m in METHODS} for pt in PROFILE_TYPES}
    file_map     = {pt: {m: {} for m in METHODS} for pt in PROFILE_TYPES}
    file_exh_map = {pt: {m: {} for m in METHODS} for pt in PROFILE_TYPES}

    df = df.drop_duplicates(subset=['fname','res_type','ptype','method'])

    for _, row in df.iterrows():
        ptype  = row.ptype
        method = row.method
        res    = row.res
        fname  = row.fname
        if row.res_type == 'non exhaustive':
            results[ptype][method][res] += 1
            file_map[ptype][method][fname] = res
        else:
            exh_results[ptype][method][res] += 1
            file_exh_map[ptype][method][fname] = res

    logger.info("Finished parsing CSV log.")
    return results, exh_results, file_map, file_exh_map

def plot_bar(ptype, data, title, fname, ylim_max=None):
    df = pd.DataFrame(data[ptype]).T.fillna(0).astype(int)
    all_keys = RESULT_KEYS.copy()
    if "Exhaustive" not in title:
        all_keys.remove('not exhaustive')
    df = df.reindex(columns=all_keys, fill_value=0)

    counts = df.sum(axis=1)
    if counts.nunique() == 1:
        cnt = counts.iloc[0]
        logger.info(f"{ptype} {title}: {cnt} entries per method.")
    else:
        logger.warning(f"{ptype} {title}: varying counts {counts.to_dict()}")

    x = np.arange(len(df))
    bottom = np.zeros(len(df), int)

    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    for key in all_keys:
        vals = df[key]
        bars = ax.bar(x, vals, bottom=bottom,
                      label=key, color=COLORS[key],
                      edgecolor='black', linewidth=0.7)
        for r in bars:
            h = r.get_height()
            if h > 0:
                txt = ax.text(
                    r.get_x() + r.get_width()/2,
                    r.get_y() + h/2,
                    str(int(h)),
                    ha='center', va='center',
                    color='white', fontsize=16, fontweight='bold'
                )
                txt.set_path_effects([pe.Stroke(linewidth=2.2, foreground='black'), pe.Normal()])
        bottom += vals

    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Count of Instances")
    ax.set_title(f"{title} for {ptype.replace('_',' ').title()}", fontsize=14)
    ax.legend(title="Result", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(f"../output/plots/{ptype}_{fname}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_stability_pies(ptype, file_map, file_exh_map, fname):
    logger.info(f"Plotting stability pies for {ptype}")

    sizes_list = []
    for mth in METHODS:
        sf = [f for f, r in file_map[ptype][mth].items()
              if r == 'stable priceable' and f in file_exh_map[ptype][mth]]
        if not sf:
            sizes_list.append((0, 0))
        else:
            rem = sum(1 for f in sf if file_exh_map[ptype][mth][f] == 'stable priceable')
            sizes_list.append((rem, len(sf) - rem))

    n = len(METHODS)
    ncols = 3 if n > 3 else n
    nrows = math.ceil(n / ncols)

    fig_w = 4.6 * ncols
    fig_h = 4.6 * nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(ptype.replace('_', ' ').title(), fontsize=20)

    positions = []
    remaining = n
    row = 0
    while remaining > 0:
        use = min(ncols, remaining)
        start = (ncols - use) // 2  # center this row
        for col in range(start, start + use):
            positions.append((row, col))
        remaining -= use
        row += 1

    def make_autopct(total):
        def _fmt(pct):
            if pct < 0.5 or total == 0:
                return ""
            count = int(round(pct * total / 100.0))
            return f"{count}\n({pct:.1f}%)"
        return _fmt

    legend_wedges = None
    labels = ['exhaustive', 'not exhaustive']

    for (row, col), (rem, lost), mth in zip(positions, sizes_list, METHODS):
        ax = axs[row][col]
        total = rem + lost
        if total == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=20, alpha=0.6)
            ax.axis('off')
        else:
            wedges, texts, autotexts = ax.pie(
                [rem, lost],
                colors=[COLORS['stable priceable'], COLORS['not exhaustive']],
                startangle=90,
                autopct=make_autopct(total),
                textprops={'fontsize': 20, 'fontweight': 'bold'},
                wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'}
            )
            ax.axis('equal')
            for t in autotexts:
                t.set_color('white')
                t.set_path_effects([pe.Stroke(linewidth=2.2, foreground='black'), pe.Normal()])
            if legend_wedges is None:
                legend_wedges = wedges
        ax.set_title(mth.replace('_', ' '), fontsize=20)

    used = set(positions)
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axs[r][c].axis('off')

    legend = fig.legend(
        legend_wedges, labels,
        title="Stable priceability allocations",
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=20,  # label size
    )
    legend.get_title().set_fontsize(20)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(f"../output/plots/{ptype}_{fname}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_coverage_from_map(ptype, file_map_like, fname, title_suffix):
    logger.info(f"Plotting coverage ({title_suffix}) for {ptype}")

    all_files = set()
    for m in METHODS:
        all_files |= set(file_map_like[ptype][m].keys())
    total = len(all_files)
    logger.info(f"{ptype} coverage[{title_suffix}]: total instances = {total}")

    buckets = {'any stable': 0, 'any priceable': 0, 'none priceable': 0}
    for f in all_files:
        res_list = [file_map_like[ptype][m].get(f) for m in METHODS]
        if 'stable priceable' in res_list:
            buckets['any stable'] += 1
        elif 'priceable' in res_list:
            buckets['any priceable'] += 1
        else:
            buckets['none priceable'] += 1

    df = pd.DataFrame.from_dict(buckets, orient='index', columns=['count'])
    df = df.loc[['any stable','any priceable','none priceable']]

    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    max_count = df['count'].max()
    ax.set_ylim(0, max_count * 1.1 if max_count > 0 else 1)

    bars = ax.bar(
        df.index, df['count'],
        color=[COLORS[k] for k in df.index],
        edgecolor='black', linewidth=0.7
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.02 * (max_count if max_count else 1),
            str(int(h)),
            ha='center', va='bottom', fontsize=16
        )

    ax.set_ylabel("Number of Instances")
    ax.set_title(f"{ptype.replace('_',' ').title()} {title_suffix}", fontsize=14)

    fig.savefig(f"../output/plots/{ptype}_{fname}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)



def plot_any_priceable(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    if 'mode' in df.columns:
        df = df[df['mode'].str.strip().str.lower() == 'priceable']

    exists_norm = df['exists'].astype(str).str.strip().str.lower()
    df['_is_true']    = (exists_norm == 'true')
    df['_is_timeout'] = (exists_norm == 'timeout')
    df['_is_false']   = (~df['_is_true'] & ~df['_is_timeout'])  # everything else

    for ptype, group in df.groupby('ptype'):
        n_true    = int(group['_is_true'].sum())
        n_false   = int(group['_is_false'].sum())
        n_timeout = int(group['_is_timeout'].sum())

        labels = ['priceable exists', 'none priceable', 'timeout']
        values = [n_true, n_false, n_timeout]
        bar_colors = [COLORS['any priceable'], COLORS['none priceable'], COLORS['timeout']]

        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
        bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=0.7)

        top = max(values) if values else 0
        ax.set_ylim(0, top * 1.1 if top > 0 else 1)

        for bar, h in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.02 * (top if top>0 else 1),
                str(h),
                ha='center', va='bottom', fontsize=16
            )

        ax.set_title(f"{ptype.replace('_',' ').title()} — Any-Priceable Among Missing", fontsize=14)
        ax.set_ylabel("Count of Instances")

        plt.savefig(f"../output/plots/{ptype}_any_priceable.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def plot_any_stable(path, exhaustiveness_type):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    if 'mode' in df.columns:
        df = df[df['mode'].str.strip().str.lower() == 'stable priceable']

    if 'exhaustive' in df.columns and exhaustiveness_type is not None:
        ex = df['exhaustive'].astype(str).str.strip().str.lower()
        if exhaustiveness_type in ('exhaustive', 'exh', 'true', True):
            df = df[ex.isin(['true', '1', 'yes'])]
        elif exhaustiveness_type in ('non_exhaustive', 'non-exh', 'false', False):
            df = df[ex.isin(['false', '0', 'no'])]

    exists_norm = df['exists'].astype(str).str.strip().str.lower()
    df['_is_true']    = (exists_norm == 'true')
    df['_is_timeout'] = (exists_norm == 'timeout')
    df['_is_false']   = (~df['_is_true'] & ~df['_is_timeout'])

    for ptype, group in df.groupby('ptype'):
        n_true    = int(group['_is_true'].sum())
        n_false   = int(group['_is_false'].sum())
        n_timeout = int(group['_is_timeout'].sum())

        labels = ['stable exists', 'none stable', 'timeout']
        values = [n_true, n_false, n_timeout]
        bar_colors = [COLORS['any stable'], COLORS['none priceable'], COLORS['timeout']]

        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
        bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=0.7)

        top = max(values) if values else 0
        ax.set_ylim(0, top * 1.1 if top > 0 else 1)

        for bar, h in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.02 * (top if top>0 else 1),
                str(h),
                ha='center', va='bottom', fontsize=16
            )

        ax.set_title(f"{ptype.replace('_',' ').title()} — Any-Stable Priceable", fontsize=14)
        ax.set_ylabel("Count of Instances")

        plt.savefig(f"../output/plots/{ptype}_any_stable_{exhaustiveness_type}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)




def main():
    res, exh, fmap, efmap = parse_csv(LOG_FILE)

    for p in PROFILE_TYPES:
        non_c = [sum(res[p][m].values()) for m in METHODS]
        exh_c = [sum(exh[p][m].values()) for m in METHODS]
        plot_bar(
            p, res,
            "Priceability Results",
            "priceability",
            ylim_max=max(*non_c)
        )
        plot_bar(
            p, exh,
            "Exhaustive Priceability Results",
            "exh_priceability",
            ylim_max=max(*exh_c)
        )
        plot_stability_pies(p, fmap, efmap, "stability_pies")
        plot_coverage_from_map(p, efmap, "coverage_exhaustive", "— Exhaustive Coverage")
        plot_coverage_from_map(p, fmap,  "coverage_non_exhaustive", "— Coverage")

    plot_any_stable("../output/STABLE_YES_EXH_YES.txt", "exhaustive")
    plot_any_priceable("../output/STABLE_NO_EXH_YES.txt")
    plot_any_stable("../output/STABLE_YES_EXH_NO.txt", "non-exhaustive")


if __name__ == "__main__":
    main()
