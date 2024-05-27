import codecs, json, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from aux_network_manipulation import *


def make_cit_reversed():
    fname = 'adjL-citation.json'
    adjL = json.load(codecs.open(fname, 'r'))

    adjL_r = {}
    for u in adjL:
        for v, w in adjL[u].items():
            if not v in adjL_r:
                adjL_r[v] = {}
            adjL_r[v][u] = w

    json.dump(adjL_r, codecs.open('adjL-citation-reversed.json', 'w'), indent=2)


def old():
    def calc_similarity(sim_key):
        data = {}
        for key in ['cit', 'cit_M', 'mis']:
            if key == 'cit':
                fname = 'adjL-citation.json'
                name = 'citation'
            elif key == 'cit_M':
                fname = 'adjL-citation-multi.json'
                name = 'citation multi'
            elif key == 'mis':
                fname = 'adjL-misclass-LinSVC-0250.json'
                name = 'misclassification'

            sizes, edges = load_adjL_135(fname)
            data[key] = {'sizes': sizes, 'edges': edges, 'name': name}

        norm_type = 'likelihood' # 'o_size', 'o_size_cutoff'
        clust_method = 'louvain' # 'sciences'

        M_range = list(range(100, 1201, 100))
        data_sim = {
            'cit v citm': [],
            'cit v cit_R': [],
            'citm v citm_R': [],
            'cit_R v citm_R': [],
            'mis v mis_R': [],
            'mis v cit': [],
            'mis_R v citm_R': []
        }

        for M in M_range:
            print(f'M={M}', flush=True)
            edges_cit = get_M_edges(data['cit']['edges'], data['cit']['sizes'], M, norm_type, is_directed=True)
            edges_cit_R = get_M_edges_regularized(data['cit']['edges'], data['cit']['sizes'], M, norm_type, is_directed=True)
            edges_citm = get_M_edges(data['cit_M']['edges'], data['cit_M']['sizes'], M, norm_type, is_directed=True)
            edges_citm_R = get_M_edges_regularized(data['cit_M']['edges'], data['cit_M']['sizes'], M, norm_type, is_directed=True)
            edges_mis = get_M_edges(data['mis']['edges'], data['mis']['sizes'], M, norm_type, is_directed=True)
            edges_mis_R = get_M_edges_regularized(data['mis']['edges'], data['mis']['sizes'], M, norm_type, is_directed=True)

            # sim = calc_similarity(edges_cit, edges_mis, sim_key)
            print(f'  edges prepared.', flush=True)

            # print(M, calc_similarity(edges_cit, edges_citm, 'JI'))

            data_sim['cit v citm'].append(calc_similarity(edges_cit, edges_citm, sim_key))
            data_sim['cit v cit_R'].append(calc_similarity(edges_cit, edges_cit_R, sim_key))
            data_sim['citm v citm_R'].append(calc_similarity(edges_citm, edges_citm_R, sim_key))
            data_sim['cit_R v citm_R'].append(calc_similarity(edges_cit_R, edges_citm_R, sim_key))
            data_sim['mis v mis_R'].append(calc_similarity(edges_mis, edges_mis_R, sim_key))
            data_sim['mis v cit'].append(calc_similarity(edges_mis, edges_cit, sim_key))
            data_sim['mis_R v citm_R'].append(calc_similarity(edges_mis_R, edges_citm_R, sim_key))

        pd.DataFrame(data_sim, index=M_range).to_csv(f'compare-networks-{sim_key}.csv')



    # calc_similarity(sim_key = 'JI')
    # calc_similarity(sim_key = 'clusters')

    sim_key = 'JI'
    sim_key = 'clusters'
    df = pd.read_csv(f'compare-networks-{sim_key}.csv', index_col=0)


    fig, ax = plt.subplots(figsize=(6,6))
    for c in df.columns:
        ax.plot(df.index, df[c], label=c)

    ax.set_ylim(0., 1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(alpha=0.4, ls='--', lw=0.2, c='black')
    plt.title(f'Network similarity, {sim_key}')

    plt.savefig(f'network-similarity-{sim_key}.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.show()






def load_network(key='cit'):
    def load_adjL_undirected(fname):
        adjL = json.load(codecs.open(fname, 'r'))

        sizes = {u: (sum(adjL[u].values()) if u in adjL else 0.0) for u in subjects_135}

        edges_raw = {}
        for u in adjL:
            if u not in subjects_135:
                continue
            for v in adjL[u]:
                if (v not in subjects_135) or (v == u) or adjL[u][v] == 0:
                    continue
                key = tuple(sorted([u, v]))
                # print(key, adjL[u][v])
                edges_raw[key] = edges_raw.get(key, 0) + adjL[u][v]

        edges = []
        for (u, v), w in edges_raw.items():
            # edges.append((u,v, w / np.sqrt(sizes[u] * sizes[v])))
            edges.append((u, v, w / (sizes[u] * sizes[v])))

        return sizes, sorted(edges, key=lambda x: x[2], reverse=True)


    if key == 'cit':
        # TODO: temporary change here
        fname = 'adjL-citation-135.json'
        name = 'citation'
        sizes, edges = load_adjL_135(fname)
    elif key == 'cit_r':
        fname = 'adjL-citation-reversed.json'
        name = 'citation reversed'
        sizes, edges = load_adjL_135(fname)
    elif key == 'mis':
        fname = 'adjL-misclass-LinSVC-0250.json'
        name = 'misclassification'
        sizes, edges = load_adjL_135(fname)
    elif key == 'cit-undirected':
        fname = 'adjL-citation.json'
        name = 'citation undirected'
        sizes, edges = load_adjL_undirected(fname)
    elif key == 'mis-undirected':
        fname = 'adjL-misclass-LinSVC-0250.json'
        name = 'misclassification undirected'
        sizes, edges = load_adjL_undirected(fname)

    return sizes, edges, name


def subj_edges_count(edges):
    same = 0
    different = 0

    for u, v, w in edges:
        if u[:2] == v[:2]:
            same += 1
        else:
            different += 1

    return same / len(edges), different / len(edges)


def plot_subject_table_LATEX():
    colors = json.load(codecs.open("colors.json", 'r'))['original']
    subj_dict = json.load(codecs.open("../resources/subjects.json", 'r'))
    sci_dict = json.load(codecs.open("../resources/sciences-full.json", 'r'))

    N = 45

    rows = [[] for i in range(N)]
    j = 0
    for i in range(len(subjects_135)):
        s = subjects_135[i]
        if j == 0 or subjects_135[i][:2] != subjects_135[i-1][:2]:
            print('\t\midrule')
            print(f'\t\multicolumn{{2}}{{c}}{{{sci_dict[s[:2]]}}} \\\\')
            j += 1

        c = colors[s[:2]][1:]
        print(f'\t\cellcolor[HTML]{{{c}}}{s} & {subj_dict[s]} \\\\')


        j += 1
        if j >= N:
            input()
            j = 0


def plot_subject_table():
    colors = json.load(codecs.open("colors.json", 'r'))['original']
    # subj_dict = json.load(codecs.open("../0_data/_journal-info/subjects.json", 'r'))
    subj_dict = json.load(codecs.open("shades/names-multiline.json", 'r'))
    sci_dict = json.load(codecs.open("../0_data/_journal-info/sciences-full.json", 'r'))

    fig, ax = plt.subplots(figsize=(12, 10))

    fsize = 6

    x, y = 0.0, 0.0
    dx = 0.5
    dy = 0.5
    prev = '0000'
    for s in subjects_135:
        if s in ['1700', '2700', '3100']:
            x += 5.0
            y = 0.0

        if s[:2] != prev[:2]:
            y -= dy/3
        txt_code = ax.text(x, y, s, color='black', fontsize=fsize, fontweight=500, ha='center', va='center')
        txt_code.set_path_effects([path_effects.Stroke(linewidth=1.4, foreground='white', alpha=0.8), path_effects.Normal()])
        ax.text(x + dx, y, subj_dict[s], color='black', fontsize=fsize, fontweight=500, ha='left', va='center')

        rect = patches.Rectangle((x-0.35, y - dy/2), 0.7, dy, linewidth=1, facecolor=colors[s[:2]])
        ax.add_patch(rect)

        y -= dy
        prev = s

    ax.set_xlim(-0.5, 18.6)
    ax.set_ylim(-20.5, 0.5)

    plt.axis('off')

    plt.savefig(f'network-subject-table.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.savefig(f'network-subject-table.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
    # plt.show()
    plt.close()




data = {}
for key in ['cit', 'mis']:
    sizes, edges, name = load_network(key)
    data[key] = {'sizes': sizes, 'edges': edges, 'name': name}


norm_type = 'likelihood'
for M in [500]:
# for M in [750, 1000]:
    edges = {
        'cit': get_M_edges(data['cit']['edges'], data['cit']['sizes'], M, norm_type, is_directed=True),
        'mis': get_M_edges(data['mis']['edges'], data['mis']['sizes'], M, norm_type, is_directed=True)}

    clusters = make_clusters(edges['mis'], method='sciences')

    plot_network(edges['mis'], 100, clusters, f'network-misclass-{norm_type}-M={M}', '', is_directed=True)
    plot_network(edges['cit'], 100, clusters, f'network-citation-{norm_type}-M={M}-135_f', '', is_directed=True)


# plot_subject_table()
# plot_subject_table_LATEX()
