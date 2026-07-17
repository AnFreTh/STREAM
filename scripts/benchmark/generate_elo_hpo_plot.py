"""
Generate ELO bar plot and CD diagram for Default vs HPO comparison.
Includes all automated metrics + all available LLM metrics.

Usage:
    python generate_elo_hpo_plot.py

Expects in /tmp/topicarena_results/:
    - v2_default_5seed_all_gpu*.csv (default automated results)
    - v3_full.csv (HPO automated results)
    - llm_scores_v2_default_5seed.csv (LLM scores for default)
    - llm_scores_v3_hpo_native_5seed.csv (LLM scores for HPO)

Will automatically include llm_intruder_accuracy if present in the v3 LLM CSV.
"""

import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import scikit_posthocs as sp

matplotlib.rcParams.update({
    'font.size': 9, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
})

# Directories are configurable via environment variables so the script runs
# anywhere. Defaults write plots next to the script under ./figures.
RESULTS_DIR = os.environ.get("TOPICARENA_RESULTS_DIR", "/tmp/topicarena_results")
OUTPUT_DIR = os.environ.get(
    "TOPICARENA_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ['BERTopicTM', 'CTM', 'CTMNeg', 'ECRTM', 'ETM', 'FASTopic', 'HyperMiner',
          'KmeansTM', 'KmeansTM_PCA', 'LDA', 'NMFTM', 'NSTM', 'NeuralLDA', 'ProdLDA', 'SawETM', 'TNTM']
DATASETS = ['BBC_News', '20Newsgroups', 'Poliblogs', 'UN', 'WHO', 'NeurIPS', 'ACL', 'Reuters',
            'NYT', 'Spotify', 'Reddit_GME', 'IMDB', 'AG_News', 'Arxiv', 'WikiText', 'PubMed', 'DBpedia', 'Yahoo_Answers']
DISPLAY = {'BERTopicTM': 'BERTopic', 'KmeansTM': 'KMeans-UMAP', 'KmeansTM_PCA': 'KMeans-PCA',
           'NMFTM': 'NMF', 'NeuralLDA': 'NeuralLDA', 'ProdLDA': 'ProdLDA', 'CTM': 'CTM',
           'CTMNeg': 'CTMNeg', 'ETM': 'ETM', 'NSTM': 'NSTM', 'FASTopic': 'FASTopic',
           'ECRTM': 'ECRTM', 'SawETM': 'SawETM', 'HyperMiner': 'HyperMiner', 'TNTM': 'TNTM', 'LDA': 'LDA'}
COLORS = {m: c for m, c in zip(sorted(DISPLAY.keys()),
          ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
           '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#000075'])}

AUTO_METRICS = ['CV', 'NPMI', 'TD', 'ISIM', 'INT', 'ISH', 'Emb_Coherence', 'Emb_TD']
LOWER_BETTER_AUTO = {'ISIM', 'ISH', 'Emb_TD'}


def load_data():
    """Load and merge all data sources."""
    # Automated metrics
    dfs_v2 = [pd.read_csv(f) for f in sorted(glob.glob(f'{RESULTS_DIR}/v2_default_5seed_all_gpu*.csv'))]
    v2_auto = pd.concat(dfs_v2, ignore_index=True)
    if 'error' in v2_auto.columns:
        v2_auto = v2_auto[v2_auto['error'].isna()]

    v3_auto = pd.read_csv(f'{RESULTS_DIR}/v3_full.csv')
    if 'error' in v3_auto.columns:
        v3_auto = v3_auto[v3_auto['error'].isna()]

    # Fill missing v3 with v2
    v3_keys = set(zip(v3_auto['dataset'], v3_auto['model'], v3_auto['seed']))
    fill_rows = [row for _, row in v2_auto.iterrows() if (row['dataset'], row['model'], row['seed']) not in v3_keys]
    v3_auto_filled = pd.concat([v3_auto, pd.DataFrame(fill_rows)], ignore_index=True)

    # LLM scores
    llm_v2 = pd.read_csv(f'{RESULTS_DIR}/llm_scores_v2_default_5seed.csv')
    llm_v3 = pd.read_csv(f'{RESULTS_DIR}/llm_scores_v3_hpo_native_5seed.csv')

    # Determine available LLM metrics
    llm_cols = ['llm_rating_mean', 'llm_label_diversity']
    if 'llm_intruder_accuracy' in llm_v2.columns:
        llm_cols.append('llm_intruder_accuracy')

    llm_v3_cols = [c for c in llm_cols if c in llm_v3.columns]

    # Merge
    merge_cols_v2 = ['dataset', 'model', 'seed'] + [c for c in llm_cols if c in llm_v2.columns]
    v2 = v2_auto.merge(llm_v2[merge_cols_v2], on=['dataset', 'model', 'seed'], how='left')

    merge_cols_v3 = ['dataset', 'model', 'seed'] + llm_v3_cols
    v3 = v3_auto_filled.merge(llm_v3[merge_cols_v3], on=['dataset', 'model', 'seed'], how='left')

    # Fill missing LLM v3 with v2 values
    for col in llm_cols:
        if col in v3.columns and col in v2.columns:
            v2_lookup = v2.set_index(['dataset', 'model', 'seed'])[col].to_dict()
            mask = v3[col].isna()
            for idx in v3[mask].index:
                key = (v3.loc[idx, 'dataset'], v3.loc[idx, 'model'], int(v3.loc[idx, 'seed']))
                if key in v2_lookup:
                    v3.loc[idx, col] = v2_lookup[key]

    # Determine which LLM metrics have enough data
    available_llm = []
    for col in llm_cols:
        if col in v3.columns and v3[col].notna().sum() > 500:
            available_llm.append(col)

    print(f"V2: {len(v2)} rows, V3: {len(v3)} rows")
    print(f"Available LLM metrics: {available_llm}")

    return v2, v3, available_llm


def build_games(v2_df, v3_df, llm_metrics):
    """Build pairwise games from all metrics."""
    v2_dm = v2_df.groupby(['dataset', 'model']).mean(numeric_only=True).reset_index()
    v3_dm = v3_df.groupby(['dataset', 'model']).mean(numeric_only=True).reset_index()

    games_by_dataset = {}
    for ds in DATASETS:
        games_by_dataset[ds] = []
        sub_def = v2_dm[v2_dm['dataset'] == ds].set_index('model')
        sub_hpo = v3_dm[v3_dm['dataset'] == ds].set_index('model')

        # Automated metrics
        for metric in AUTO_METRICS:
            col = f'{metric}@10'
            lower_better = metric in LOWER_BETTER_AUTO
            vals = {}
            for m in MODELS:
                if m in sub_def.index and col in sub_def.columns:
                    v = sub_def.loc[m, col]
                    if not np.isnan(v):
                        vals[f'{m}_def'] = v
                if m in sub_hpo.index and col in sub_hpo.columns:
                    v = sub_hpo.loc[m, col]
                    if not np.isnan(v):
                        vals[f'{m}_hpo'] = v
            _add_pairwise(games_by_dataset[ds], vals, lower_better)

        # LLM metrics (all higher = better)
        for col in llm_metrics:
            vals = {}
            for m in MODELS:
                if m in sub_def.index and col in sub_def.columns:
                    v = sub_def.loc[m, col]
                    if not np.isnan(v):
                        vals[f'{m}_def'] = v
                if m in sub_hpo.index and col in sub_hpo.columns:
                    v = sub_hpo.loc[m, col]
                    if not np.isnan(v):
                        vals[f'{m}_hpo'] = v
            _add_pairwise(games_by_dataset[ds], vals, lower_better=False)

    total = sum(len(g) for g in games_by_dataset.values())
    print(f"Total games: {total}")
    return games_by_dataset


def _add_pairwise(game_list, vals, lower_better):
    players = list(vals.keys())
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            p_a, p_b = players[i], players[j]
            v_a, v_b = vals[p_a], vals[p_b]
            if lower_better:
                if v_a < v_b: s_a = 1.0
                elif v_a > v_b: s_a = 0.0
                else: s_a = 0.5
            else:
                if v_a > v_b: s_a = 1.0
                elif v_a < v_b: s_a = 0.0
                else: s_a = 0.5
            game_list.append((p_a, p_b, s_a))


def run_bootstrap_elo(games_by_dataset, n_bootstrap=200, K=32, n_passes=8):
    """Dataset-level bootstrap ELO."""
    PLAYERS = [f'{m}_def' for m in MODELS] + [f'{m}_hpo' for m in MODELS]
    all_elos = {p: [] for p in PLAYERS}

    for boot in range(n_bootstrap):
        elo = {p: 1000 for p in PLAYERS}
        np.random.seed(boot)
        boot_datasets = np.random.choice(DATASETS, size=len(DATASETS), replace=True)
        boot_games = []
        for ds in boot_datasets:
            boot_games.extend(games_by_dataset[ds])
        for p_iter in range(n_passes):
            k = K * (1 - 0.4 * p_iter / n_passes)
            order = np.random.permutation(len(boot_games))
            for idx in order:
                p_a, p_b, s_a = boot_games[idx]
                s_b = 1.0 - s_a
                e_a = 1.0 / (1.0 + 10 ** ((elo[p_b] - elo[p_a]) / 400))
                e_b = 1.0 - e_a
                elo[p_a] += k * (s_a - e_a)
                elo[p_b] += k * (s_b - e_b)
        for p in PLAYERS:
            all_elos[p].append(elo[p])
        if (boot + 1) % 50 == 0:
            print(f'  Bootstrap {boot + 1}/{n_bootstrap}', flush=True)

    return all_elos


def plot_elo(all_elos, filename='elo_all_metrics'):
    """Generate the ELO bar plot."""
    col_def = '#5778a4'
    col_hpo = '#e49444'
    width_fat = 0.6
    width_slim = 0.38

    elo_def_mean = {m: np.mean(all_elos[f'{m}_def']) for m in MODELS}
    elo_def_lo = {m: np.percentile(all_elos[f'{m}_def'], 2.5) for m in MODELS}
    elo_def_hi = {m: np.percentile(all_elos[f'{m}_def'], 97.5) for m in MODELS}
    elo_hpo_mean = {m: np.mean(all_elos[f'{m}_hpo']) for m in MODELS}
    elo_hpo_lo = {m: np.percentile(all_elos[f'{m}_hpo'], 2.5) for m in MODELS}
    elo_hpo_hi = {m: np.percentile(all_elos[f'{m}_hpo'], 97.5) for m in MODELS}

    sorted_models = sorted(MODELS, key=lambda x: -(elo_def_mean[x] + elo_hpo_mean[x]) / 2)
    x_pos = np.arange(len(sorted_models))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, m in enumerate(sorted_models):
        d = elo_def_mean[m]
        h = elo_hpo_mean[m]
        d_err = [[d - elo_def_lo[m]], [elo_def_hi[m] - d]]
        h_err = [[h - elo_hpo_lo[m]], [elo_hpo_hi[m] - h]]
        if h >= d:
            ax.bar(i, h, width_fat, color=col_hpo, alpha=0.7, zorder=2)
            ax.errorbar(i, h, yerr=h_err, color='#333333', capsize=2, capthick=0.7, linewidth=0.7, zorder=4, fmt='none')
            ax.bar(i, d, width_slim, color=col_def, alpha=0.9, zorder=3)
            ax.errorbar(i, d, yerr=d_err, color='#333333', capsize=2, capthick=0.7, linewidth=0.7, zorder=5, fmt='none')
        else:
            ax.bar(i, d, width_fat, color=col_def, alpha=0.7, zorder=2)
            ax.errorbar(i, d, yerr=d_err, color='#333333', capsize=2, capthick=0.7, linewidth=0.7, zorder=4, fmt='none')
            ax.bar(i, h, width_slim, color=col_hpo, alpha=0.9, zorder=3)
            ax.errorbar(i, h, yerr=h_err, color='#333333', capsize=2, capthick=0.7, linewidth=0.7, zorder=5, fmt='none')

    legend_elements = [Patch(facecolor=col_def, alpha=0.85, label='Default'),
                       Patch(facecolor=col_hpo, alpha=0.75, label='HPO')]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    ax.set_ylabel('ELO Rating', fontsize=10)
    ax.set_ylim(600, 1300)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in sorted_models], rotation=45, ha='right', fontsize=8.5)
    ax.axhline(y=1000, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{RESULTS_DIR}/{filename}.png', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}.pdf/png')

    # Print table
    print(f'\n{"Model":<15} {"Default":>8} {"HPO":>8} {"Delta":>8}')
    print('-' * 45)
    for m in sorted_models:
        print(f'{DISPLAY.get(m, m):<15} {elo_def_mean[m]:>8.0f} {elo_hpo_mean[m]:>8.0f} {elo_hpo_mean[m] - elo_def_mean[m]:>+8.0f}')


def plot_cd_joint(v2_df, v3_df, llm_metrics, filename='cd_cv_hpo_joint'):
    """Generate mirrored CD diagram for CV@10."""
    def build_cv_pivot(df):
        dm = df.groupby(['dataset', 'model'])['CV@10'].mean().reset_index()
        pivot = dm.pivot(index='dataset', columns='model', values='CV@10')
        ranked = pivot.rank(axis=1, ascending=False)
        ranked.columns = [DISPLAY.get(m, m) for m in ranked.columns]
        return ranked

    pivot_def = build_cv_pivot(v2_df)
    pivot_hpo = build_cv_pivot(v3_df)

    avg_def = pivot_def.mean().sort_values()
    avg_hpo = pivot_hpo.mean().sort_values()

    cd_colors = {DISPLAY[m]: COLORS[m] for m in DISPLAY}

    sig_def = sp.posthoc_nemenyi_friedman(pivot_def.values)
    sig_def.index = pivot_def.columns
    sig_def.columns = pivot_def.columns

    sig_hpo = sp.posthoc_nemenyi_friedman(pivot_hpo.values)
    sig_hpo.index = pivot_hpo.columns
    sig_hpo.columns = pivot_hpo.columns

    fig = plt.figure(figsize=(12, 5))
    ax_def = fig.add_subplot(212)
    ax_hpo = fig.add_subplot(211, sharex=ax_def)

    sp.critical_difference_diagram(ranks=avg_def, sig_matrix=sig_def, ax=ax_def, color_palette=cd_colors)
    ax_def.xaxis.set_visible(False)
    ax_def.spines['top'].set_visible(False)

    sp.critical_difference_diagram(ranks=avg_hpo, sig_matrix=sig_hpo, ax=ax_hpo, color_palette=cd_colors)
    ax_hpo.invert_yaxis()
    ax_hpo.spines['top'].set_visible(False)
    ax_hpo.spines['bottom'].set_position('zero')
    ax_hpo.xaxis.set_ticks_position('bottom')

    plt.subplots_adjust(hspace=0)
    plt.savefig(f'{RESULTS_DIR}/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{RESULTS_DIR}/{filename}.png', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}.pdf/png')


if __name__ == '__main__':
    print("Loading data...")
    v2, v3, llm_metrics = load_data()

    print("\nBuilding games...")
    games = build_games(v2, v3, llm_metrics)

    print("\nRunning bootstrap ELO...")
    all_elos = run_bootstrap_elo(games)

    print("\nPlotting ELO...")
    plot_elo(all_elos, filename='elo_all_metrics')

    print("\nPlotting CD diagram...")
    plot_cd_joint(v2, v3, llm_metrics)

    print("\nDone!")
