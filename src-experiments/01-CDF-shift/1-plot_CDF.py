import codecs, json, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patheffects as path_effects


def plot_CDF(path, algorithms, t_sizes, n_tests, fpath_out):
    NN = 200
    bbins = np.linspace(-0.5, 3.5, NN)
    dx = bbins[1] - bbins[0]

    for t in t_sizes:
        print(t)
        for alg in algorithms:
            print(f' {alg}')
            data = {'correct': [], 'incorrect': []}
            for i_seed in range(n_tests):
                with pd.ExcelFile(path + f'test-{i_seed:02d}-{alg}-{t:04d}.xlsx') as xls:
                    df = pd.read_excel(xls, 'Sheet0')
                mask_correct = np.zeros(df.index.shape, dtype=bool)
                idx = np.where(df['subject'] == df['classification'])[0]
                mask_correct[idx] = True

                data['correct'].append(np.array(df['ncit'][mask_correct], dtype=float))
                data['incorrect'].append(np.array(df['ncit'][~mask_correct], dtype=float))
                print(f'accuracy = {len(idx) / len(df.index):.3f}')

            dist = 0.0
            plt.figure(figsize=(4, 4))
            clrs = {'correct': 'C0', 'incorrect': 'C3'}
            for key, val in data.items():
                ncit = np.concatenate(val)
                print(key, ncit.shape)
                m = np.mean(ncit)
                ncit[np.where(ncit == 0)[0]] = 0.1

                с, _ = np.histogram(np.log10(ncit), bins=bbins, density=True)
                с /= (dx * с).sum()
                nums_cdf = np.cumsum(с * dx)
                integral = np.cumsum(nums_cdf * dx)
                dist += integral[-1] if key == 'correct' else -integral[-1]

                # plt.plot(bbins[1:], nums_cdf, '-', linewidth=2.0, label=f'{key.capitalize()}, m={m:.1f}')
                plt.plot(bbins[1:], nums_cdf, '-', c=clrs[key], linewidth=2.0,
                         label=f'{key.capitalize()}, ' + r'$\overline{N_{cit}} = $' + f'{m:.1f}')

            # print(dist, len(df.index))

            plt.xlim((-0.2, 3.0))
            plt.ylim((0.0, 1.03))
            # plt.legend(loc='lower right')
            plt.legend(loc='best')
            # plt.title(f'{alg}, {t}\nd={dist:.3f}')
            plt.xlabel(r'$\log_{10}(N_{cit})$')
            plt.ylabel('Cumulative probability')
            plt.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
            plt.savefig(fpath_out + '.png', dpi=400, bbox_inches='tight')
            plt.savefig(fpath_out + '.pdf', dpi=400, bbox_inches='tight')
            plt.close()
            # plt.show()


def prepare_window_data(path, t_size, i_seed, ws):
    NN = 200
    bbins = np.linspace(-0.5, 3.5, NN)
    dx = bbins[1] - bbins[0]
    path_CDF = path + 'CDF-separate/'
    if not os.path.exists(path_CDF):
        os.makedirs(path_CDF)

    res = 'percentile,n articles,shift,accuracy\n'
    for pct in range(0, 93, 4):
        data = {'correct': [], 'incorrect': []}
        with pd.ExcelFile(path + f'test-{i_seed:02d}-{t_size:04d}-p={pct:02d}-{pct + ws:02d}.xlsx') as xls:
            df = pd.read_excel(xls, 'Sheet0')
        mask_correct = np.zeros(df.index.shape, dtype=bool)
        idx = np.where(df['subject'] == df['classification'])[0]
        mask_correct[idx] = True

        data['correct'].append(np.array(df['ncit'][mask_correct], dtype=float))
        data['incorrect'].append(np.array(df['ncit'][~mask_correct], dtype=float))
        # print(f'accuracy = {len(idx)/len(df.index):.3f}')

        dist = 0.0
        plt.figure(figsize=(4, 4))
        for key, val in data.items():
            ncit = np.concatenate(val)
            # print(key, ncit.shape)
            m = np.mean(ncit)
            ncit[np.where(ncit == 0)[0]] = 0.1

            с, _ = np.histogram(np.log10(ncit), bins=bbins, density=True)
            с /= (dx * с).sum()
            nums_cdf = np.cumsum(с * dx)
            integral = np.cumsum(nums_cdf * dx)
            dist += integral[-1] if key == 'correct' else -integral[-1]

            # plt.plot(bbins[1:], nums_cdf, '-', linewidth=2.0, label=f'{key}, mean={m:4.2f}')
            plt.plot(bbins[1:], nums_cdf, '-', linewidth=2.0,
                     label=f'{key.capitalize()}, ' + r'$\overline{N_{cit}} = $' + f'{m:.1f}')

        # print(f'{pct+5},{len(df.index)},{dist:.4f},{len(idx)/len(df.index):.3f}')
        res += f'{pct + ws // 2},{len(df.index)},{dist:.4f},{len(idx) / len(df.index):.3f}\n'

        plt.xlim((-0.2, 3.0))
        plt.ylim((0.0, 1.03))
        plt.legend(loc='lower right')
        plt.title(f'{t_size}, percentile [{pct:2d}, {pct + ws:2d}]\n{dist:.4f}')
        plt.xlabel(r'$\log_{10}(N_{cit})$')
        plt.ylabel('Cumulative probability')
        plt.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
        plt.savefig(path_CDF + f'{t_size:04d}-p={pct:02d}-{pct + ws:02d}.png', dpi=400, bbox_inches='tight')
        plt.close()
        # plt.show()

    with codecs.open(path + f'data-CDF_shift.txt', 'w') as fout:
        fout.write(res)


def plot_window_overview(path):
    with codecs.open(path + f'data-CDF_shift.txt', 'r') as fin:
        keys = fin.readline().strip().split(',')

        data = {k: [] for k in keys}
        for line in fin:
            vals = line.strip().split(',')
            for i, k in enumerate(keys):
                data[k].append(float(vals[i]))

        fig, ax = plt.subplots(sharex=True, figsize=(8, 3))

        # r_top10 = plt.Rectangle((90, -0.15), 10, 0.3, fc='#02a7e3', ec=None, alpha=0.6)
        r_top10 = plt.Rectangle((90, -0.15), 10, 0.3, fc='grey', ec=None, alpha=0.4)
        ax.add_patch(r_top10)
        ax.text(0.95, 0.94, f'top 10', fontsize=10, transform=ax.transAxes, ha='center', va='center')

        ax.plot([0,100], [0,0], '-', c='black', linewidth=1.0)
        ax.plot(data['percentile'], data['shift'], 'o-', c='C3', linewidth=2.0)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('CDF shift')
        ax.set_xlim((0, 100))
        ax.set_ylim((-0.15, 0.15))

        ax.set_xticks([i*10 for i in range(11)])

        ax.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
        plt.savefig(path + f'CDF_shift-window-overview.png', dpi=400, bbox_inches='tight')
        plt.savefig(path + f'CDF_shift-window-overview.pdf', dpi=400, bbox_inches='tight')
        # plt.close()
        plt.show()




t_sizes = [500]
algorithms = ['LinSVC']
# algorithms = ['MLPC', 'NB', 'LogReg', 'LinSVC']
n_tests = 1


# plot_CDF('results/all/', algorithms, t_sizes, n_tests, f'results/CDF-all')
# plot_CDF('results/top10e/', algorithms, t_sizes, n_tests, f'results/CDF-top10e')


alg = 'LinSVC'
t_size = 500
i_seed = 0
ws = 8
path = f'results-window/{t_size:04d}-{alg}-{ws}/'

plot_window_overview(path)
