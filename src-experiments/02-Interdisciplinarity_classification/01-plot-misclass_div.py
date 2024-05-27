import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
import os


def prepare_averaged_dataset(path, alg, train_size):
    df = pd.read_excel(path + f'test-00-{alg}-{train_size:04d}.xlsx', 'Sheet0')
    journal_ids = list(set(df['jID']))
    ncit = []
    misclass_diversity = []
    nart = []
    for jid in journal_ids:
        df_sub = df[df['jID'] == jid]
        ncit.append(np.mean(df_sub['ncit']))
        misclass_diversity.append(np.mean(df_sub['misclass_diversity']))
        nart.append(len(df_sub.index))

    df_j = pd.DataFrame({'jID': journal_ids, 'ncit': ncit, 'misclass_diversity': misclass_diversity, 'nart': nart})
    df_j['percentile'] = -1
    df_sources = pd.read_excel(f'../0_data/_journal-info/selected_journals.xlsx')
    journal_perc = {row['Scopus Source ID']: row['Percentile'] for _, row in df_sources.iterrows()}
    for i in df_j.index:
        df_j.at[i, 'percentile'] = journal_perc[df_j.at[i, 'jID']]

    df_j.to_excel(path + f'test-00-{alg}-{train_size:04d}-j_averaged.xlsx', sheet_name='Sheet0', index=False)


def plot_misclass_div_example(path, alg, train_size, c=2.5, indices=list(range(25))):
    def get_misclass_diversity(nums, c=2.5):
        s = np.std(nums)
        m = np.mean(nums)
        x = m + c * s
        n = np.sum(nums >= x)
        return x, n


    def plot_hist(nums, fname, xlims, bbins, div_x, div_n):
        fig, ax = plt.subplots(figsize=(3,3))
        ax.hist(nums, bins=bbins)
        ax.text(0.72, 0.90, f'CD={div_n}', fontsize=10, transform=ax.transAxes, ha='center', va='center')
        plt.plot([div_x, div_x], [0, 30], '--', c='black', lw=1.0, alpha=0.4)

        plt.xlabel(f'Certainty of a subject')
        plt.ylabel(f'Number of subjects')

        plt.xlim(xlims)
        plt.ylim((0, 30))
        # plt.yticks([0,5,10,15,20])

        plt.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
        plt.savefig(fname + '.png', dpi=400, bbox_inches='tight')
        plt.savefig(fname + '.pdf', dpi=400, bbox_inches='tight')
        plt.close()


    df_y = pd.read_excel(path + f'yscores-00-{alg}-{train_size:04d}.xlsx', index_col=0)

    path_out = path + 'classification_diversity-examples/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    for i in indices:
        nums = np.array(df_y.iloc[i])
        div_x, div_n = get_misclass_diversity(nums, c)
        plot_hist(nums, path_out + f'scores-i={i:04d}', (-2.00, 1.5), np.linspace(-2.0, 1.5, 50), div_x, div_n)



def plot_misclassDiv_ncit(ncit, misclass_diversity, title, fig_name):
    r_pearson, p_pearson = pearsonr(ncit, misclass_diversity)
    title += f'\npearson r={r_pearson:.3f}, p={p_pearson:.1e}'
    r_spearman, p_spearman = spearmanr(ncit, misclass_diversity)
    # title += f'\nspearman r = {r_spearman:.3f}, p={p_spearman:.1e}'


    slope, intercept, r_value, p_value, std_err = linregress(misclass_diversity, ncit)
    # x = np.array([min(misclass_diversity), max(misclass_diversity)])
    x = np.array([1.4, 3.4])
    y = slope * x + intercept

    _, ax = plt.subplots(figsize=(3, 3))
    plt.title(title)
    ax.scatter(misclass_diversity, ncit, s=15, alpha=.3, label='samples')
    ax.plot(x, y, '-', c='black', alpha=0.8)
    ax.text(x[-1] + 0.1, y[-1], f'{slope:.1f}', ha='left', va='center')

    # ax.text(0.5, -0.30, f'pearson r={r_pearson:.3f}, p={p_pearson:.1e}', fontsize=10, transform=ax.transAxes, ha='center', va='center')

    ax.set_xlim((1.0, 4.0))
    ax.set_ylim((-10.0, 300.0))
    ax.set_xlabel('Classification diversity')
    ax.set_ylabel(r'Average $N_{cit}$')
    plt.grid(alpha=0.4, ls='--', lw=0.2, c='black')
    plt.savefig(fig_name + '.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.savefig(fig_name + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.close()

    return r_pearson, p_pearson, r_spearman, p_spearman, slope



def compute_MD_Ncit_correlation(path, alg, train_size, ws, h=5):
    df_j = pd.read_excel(path + f'test-00-{alg}-{train_size:04d}-j_averaged.xlsx', 'Sheet0')
    df_j = df_j[df_j['nart'] >= 10]

    path_out = path + f'ws={ws}/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    data = {
        'pct': [],
        'pearson': [],
        'pearson p': [],
        'spearman': [],
        'spearman p': [],
        'regression': []
    }
    for pct in range(0, 93, h):
        df_j_sub = df_j[(df_j['percentile'] >= pct) & (df_j['percentile'] <= pct+ws)]
        title = f'percentile=[{pct},{pct+ws}]'
        fig_name = path_out + f'CD-ncit-[{pct:02d}_{pct+ws:02d}]'
        r_pearson, p_pearson, r_spearman, p_spearman, slope = plot_misclassDiv_ncit(df_j_sub['ncit'], df_j_sub['misclass_diversity'], title, fig_name)

        data['pct'].append(pct + ws/2)
        data['pearson'].append(r_pearson)
        data['pearson p'].append(p_pearson)
        data['spearman'].append(r_spearman)
        data['spearman p'].append(p_spearman)
        data['regression'].append(slope)

    pd.DataFrame(data).to_csv(path_out + 'data-CD-Ncit-correlation.csv',index=False)


def plot_overview(path, ws=9, key='regression'):
    df = pd.read_csv(path + f'ws={ws}/data-CD-Ncit-correlation.csv')

    fig, ax = plt.subplots(sharex=True, figsize=(8, 3))

    if key == 'regression':
        r_top10 = plt.Rectangle((90, -10), 10, 70, fc='grey', ec=None, alpha=0.4)
        ax.add_patch(r_top10)
        ax.text(0.95, 0.94, f'top 10', fontsize=10, transform=ax.transAxes, ha='center', va='center')

    ax.plot([0, 100], [0, 0], '-', c='black', linewidth=1.0)
    nums = df[key]
    if key == 'pearson p':
        nums = - np.log10(nums)

    ax.plot(df['pct'], nums, 'o-', c='C0', linewidth=2.0)
    ax.set_xlabel('Journal percentile')
    ax.set_ylabel(key.capitalize())
    ax.set_xlim((0, 100))
    if key == 'regression':
        ax.set_ylabel('Regression coefficient')
        ax.set_ylim((-10, 60))
    if key == 'pearson p':
        ax.plot([0, 100], -np.log10([0.05, 0.05]), '--', c='black', alpha=0.5)
        ax.set_ylabel(r'$-\log_{10}{(p)}$')

    ax.set_xticks([i * 10 for i in range(11)])

    ax.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
    plt.savefig(path + f'ws={ws}/CD-Ncit-{key}.png', dpi=400, bbox_inches='tight')
    plt.savefig(path + f'ws={ws}/CD-Ncit-{key}.pdf', dpi=400, bbox_inches='tight')
    # plt.close()
    plt.show()


path = f'results/'
alg = 'LinSVC'
train_size = 500

# prepare_averaged_dataset(path, alg, train_size)

# plot_misclass_div_example(path, alg, train_size, c=2.5, indices=list(range(25)))
#
# compute_MD_Ncit_correlation(path, alg, train_size, ws=9)
#
# plot_overview(path, ws=9, key='regression')

plot_overview(path, ws=9, key='pearson')
plot_overview(path, ws=9, key='pearson p')
# plot_overview(path, ws=9, key='spearman')





