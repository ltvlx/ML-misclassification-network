import codecs, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon

from networkx.algorithms.community import greedy_modularity_communities, asyn_lpa_communities, k_clique_communities
import community as community_louvain

from scipy.integrate import simps

subjects_135 = [
    '1000', '1100', '1102', '1103', '1104', '1105', '1106', '1107', '1108', '1109', '1110', '1111', '1200', '1202',
    '1207', '1208', '1210', '1211', '1212', '1213', '1300', '1303', '1307', '1311', '1314', '1400', '1402', '1406',
    '1407', '1408', '1409', '1500', '1600', '1602', '1605', '1606', '1700', '1702', '1705', '1712', '1900', '1902',
    '1904', '1906', '1907', '1908', '1909', '1910', '1911', '2000', '2002', '2200', '2202', '2204', '2205', '2208',
    '2209', '2210', '2300', '2312', '2404', '2500', '2600', '2601', '2602', '2604', '2608', '2613', '2700', '2703',
    '2705', '2706', '2707', '2708', '2711', '2712', '2713', '2714', '2715', '2717', '2719', '2720', '2723', '2724',
    '2725', '2727', '2728', '2729', '2730', '2731', '2732', '2733', '2734', '2735', '2736', '2738', '2739', '2740',
    '2741', '2745', '2746', '2748', '2800', '2900', '3000', '3003', '3004', '3005', '3100', '3101', '3104', '3106',
    '3107', '3200', '3202', '3203', '3204', '3207', '3300', '3301', '3304', '3305', '3308', '3309', '3312', '3314',
    '3315', '3317', '3318', '3320', '3400', '3404', '3500', '3504', '3612']


def load_adjL_135(fname):
    adjL = json.load(codecs.open(fname, 'r'))

    edges = []
    sizes = {u: (sum(adjL[u].values()) if u in adjL else 0.0) for u in subjects_135}

    for u in adjL:
        if u not in subjects_135:
            continue
        for v in adjL[u]:
            if (v not in subjects_135) or (v == u):
                continue
            edges.append((u, v, adjL[u][v]))

    return sizes, sorted(edges, key=lambda x: x[2], reverse=True)


def get_M_edges(edges, sizes, M, norm_type, is_directed=True):
    def normalize_by_out_size(edges, sizes):
        return sorted(
            [(u, v, w / sizes[u]) for (u, v, w) in edges],
            key=lambda x: x[2],
            reverse=True
        )

    def normalize_by_out_size_cutoff(edges, sizes, cutoff_threshold=10):
        return sorted(
            [(u, v, w / sizes[u] if w > cutoff_threshold else 0.0) for (u, v, w) in edges],
            key=lambda x: x[2],
            reverse=True
        )

    def normalize_by_likelihood(edges, sizes, is_directed=True):
        edges_out = []
        if is_directed:
            p_node = {x: sizes[x] / sum(sizes.values()) for x in sizes}
            for u, v, w in edges:
                # _unlikely rare_  <  (mean - 2.58*std)  <  _99% CI_  <  mean + 2.58*std  <   _unlikely frequent_
                Ni = sizes[u]
                p = p_node[v]
                mean = Ni * p
                std = np.sqrt(Ni * (1. - p) * p)
                z = (w - mean) / (std + 1e-8)
                edges_out.append((u, v, z))
        else:
            N = sum(sizes.values())
            for u, v, w in edges:
                p = sizes[u] * sizes[v] / (N ** 2)
                mean = p * N
                std = np.sqrt(N * (1. - p) * p)

                z = (w - mean) / std
                edges_out.append((u, v, z))

        return sorted(edges_out, key=lambda x: x[2], reverse=True)

    if norm_type == 'likelihood':
        edges_normalized = normalize_by_likelihood(edges, sizes, is_directed=True)
    elif norm_type == 'o_size':
        edges_normalized = normalize_by_out_size(edges, sizes)
    elif norm_type == 'o_size_cutoff':
        edges_normalized = normalize_by_out_size_cutoff(edges, sizes)
    else:
        raise KeyError("Unknown `norm_type`. Please use one of: ['likelihood','o_size','o_size_cutoff']")

    if is_directed:
        return edges_normalized[:M]
    else:
        edges_out = {}
        for i, (u, v, w) in enumerate(edges_normalized):
            u, v = min(u, v), max(u, v)
            if (u, v) not in edges_out:
                edges_out[(u, v)] = w

            if len(edges_out) >= M:
                break

        return [(u, v, w) for (u, v), w in edges_out.items()]


def get_M_edges_regularized(edges, sizes, M, norm_type, is_directed=True):
    rng = np.random.default_rng()
    med = int(np.median(list(sizes.values())))
    # med = int(np.mean(list(sizes.values())))
    s_min, s_max = med // 2, med

    edges_reg_count = {}
    for t in range(100):
        sizes_reg = dict(sizes)
        for u in sizes_reg:
            sizes_reg[u] += rng.integers(s_min, s_max)

        edges_M = get_M_edges(edges, sizes_reg, M, norm_type, is_directed)

        for u, v, w in edges_M:
            edges_reg_count[(u, v)] = edges_reg_count.get((u, v), 0.0) + 1.0

    return [(u, v, w) for (u, v), w in sorted(edges_reg_count.items(), key=lambda x: x[1], reverse=True)[:M]]


def make_clusters(edges, method='greedy_modularity'):
    if method == 'sciences':
        c = {}
        for subj in subjects_135:
            sci = subj[:2]
            c[sci] = c.get(sci, []) + [subj]
        # c = [list(_c) for _c in c.values()]
        return c
    else:
        G = nx.Graph()
        G.add_nodes_from(subjects_135)
        G.add_weighted_edges_from(edges)

        if method == 'greedy_modularity':
            c = list(greedy_modularity_communities(G))
        elif method == 'asyn_lpa':
            c = list(asyn_lpa_communities(G))
        elif method == 'k_clique':
            c = list(k_clique_communities(G, 6))
        elif method == 'louvain':
            partition = community_louvain.best_partition(G)
            n = max(partition.values()) + 1
            c = [[] for _ in range(n)]
            for subj, i in partition.items():
                c[i].append(subj)

        return sorted([list(_c) for _c in c], key=lambda x: len(x), reverse=True)


def color_clusters(clusters):
    colors = json.load(codecs.open("colors.json", 'r'))

    if len(clusters) <= 27:
        colors = colors['original']
    else:
        colors = colors['extended']

    colored_clusters = {}
    for clust in clusters:
        counts = {}
        for subj in clust:
            counts[subj[:2]] = counts.get(subj[:2], 0) + 1

        for sci, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if sci not in colored_clusters:
                colored_clusters[sci] = list(clust)
                colors.pop(sci)
                break
        else:
            sci = list(colors.keys())[0]
            colors.pop(sci)
            colored_clusters[sci] = list(clust)
    return colored_clusters


def read_shapes(key):
    shape = []
    with codecs.open(f'shades/shape-{key}.txt', 'r') as fin:
        for line in fin:
            a = line.strip().split(',')
            x = float(a[0])
            y = float(a[1])
            shape.append((x,y))
    return shape


def Chaikin_corner_cutting(shape):
    shape_out = []

    shape.append(shape[0])
    for i in range(len(shape)-1):
        x0, y0 = shape[i]
        x1, y1 = shape[i+1]

        shape_out.append((0.75*x0 + 0.25*x1, 0.75*y0 + 0.25*y1))
        shape_out.append((0.25*x0 + 0.75*x1, 0.25*y0 + 0.75*y1))

    return shape_out


def plot_network(edges, sizes, clusters, fname, title, is_directed):
    def split_edges(G):
        e_both = []
        e_single = []
        for e in G.edges():
            if (e[1], e[0]) in G.edges():
                e_both.append(e)
            else:
                e_single.append(e)
        return e_both, e_single

    pos = json.load(codecs.open('pos.json', 'r'))


    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(subjects_135)
    G.add_weighted_edges_from(edges)
    print(f'\n{title};\n nx: {len(G.edges())} edges')

    colors = json.load(codecs.open("colors.json", 'r'))
    clust_colors = colors['original'] if len(clusters) <= 27 else colors['extended']

    fig, ax = plt.subplots(figsize=(16, 12))
    for sci, subjects in clusters.items():
        nx.draw_networkx_nodes(G, pos, nodelist=subjects, node_size=sizes, node_color=clust_colors[sci],
                               edgecolors='#5c5c5c', linewidths=0.5)
    if is_directed:
        e_both, e_single = split_edges(G)
        w_single = 1.0
        w_both = 1.0

        nx.draw_networkx_edges(G, pos, node_size=sizes, edgelist=e_both, width=w_both, edge_color='black',
                               arrowstyle='->', arrowsize=8)
        nx.draw_networkx_edges(G, pos, node_size=sizes, edgelist=e_single, width=w_single, edge_color='black',
                               arrowstyle='->', arrowsize=8)
    else:
        nx.draw_networkx_edges(G, pos, node_size=sizes, width=1.0, edge_color='black')

    poly_lines = json.load(codecs.open('shades/polygons.json', 'r'))
    for s_key in ['11', '12', '13', '14', '16', '17', '19', '22', '26', '27', '31', '33']:
        polygon_smooth = Polygon(poly_lines[s_key], fill=True, color=clust_colors[s_key], edgecolor=None, alpha=0.3, zorder=0)
        ax.add_patch(polygon_smooth)

    poly_titles = json.load(codecs.open('shades/titles.json', 'r'))
    for s_key in poly_titles:
        vals = poly_titles[s_key]
        xy = vals['pos']
        text = vals['text']
        txt = ax.text(xy[0], xy[1], text, color=clust_colors[s_key], fontsize=14, fontweight=500, ha='center')

    ax.plot([.179, .262], [.365, .430], ':', color=clust_colors['13'], lw=5.0, zorder=0, alpha=0.4)
    ax.plot([.620, .580], [.220, .490], ':', color=clust_colors['22'], lw=5.0, zorder=0, alpha=0.4)

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    plt.title(title)
    plt.grid(alpha=0.4, linestyle='--', linewidth=0.2, color='black')
    plt.axis('off')

    lab_dict = nx.draw_networkx_labels(G, pos, font_size=6)
    for _, txt in lab_dict.items():
        txt.set_path_effects(
            [path_effects.Stroke(linewidth=0.8, foreground='white', alpha=0.6), path_effects.Normal()])

    plt.savefig(fname + '.png', dpi=400, bbox_inches='tight')
    plt.savefig(fname + '.pdf', dpi=400, bbox_inches='tight')
    plt.close()







def calc_similarity(edges_1, edges_2, method='JI'):
    def similarity_JI(edges_1, edges_2):
        edges_1 = set([(u, v) for u, v, _ in edges_1])
        edges_2 = set([(u, v) for u, v, _ in edges_2])
        return len(edges_1 & edges_2) / len(edges_1 | edges_2)

    def distance_editing(edges_1, edges_2):
        edges_1 = set([(u, v) for u, v, _ in edges_1])
        edges_2 = set([(u, v) for u, v, _ in edges_2])
        return len(edges_1 | edges_2) - len(edges_1 & edges_2)

    def similarity_clusters(edges_1, edges_2):
        c_1 = make_clusters(edges_1, method='louvain')
        c_2 = make_clusters(edges_2, method='louvain')

        n_1 = len(c_1)
        n_2 = len(c_2)

        A = np.zeros((n_1, n_2))
        for i in range(n_1):
            subj_1 = set(c_1[i])
            for j in range(n_2):
                subj_2 = set(c_2[j])
                jac = len(subj_1 & subj_2) / len(subj_1 | subj_2)
                A[i, j] = jac
        # df = pd.DataFrame(data=A, index=[str(x) for x in clust_miscl], columns=[str(x) for x in clust_cit])
        # df.to_csv(f'community_matching-JI-miscl-cit.csv')

        # return np.sum(A) / min(len(clust_miscl), len(clust_cit))
        return np.sum(A) / max(n_1, n_2)

    def distance_spectral(edges_1, edges_2):
        def calculate_rho_density(n, adj_M, fx):
            diag_M = np.zeros((n, n), dtype=int)

            for i in range(n):
                row = adj_M[i]
                diag_M[i, i] = sum(row)
                # print(row, sum(row))

            lapl_M = diag_M - adj_M

            om = np.sort(np.linalg.eigvals(lapl_M))
            imag_part = np.imag(om)
            assert(np.max(np.abs(imag_part)) < 1e-8)
            mask = np.abs(om) < 1e-8
            om[mask] = 0.0
            om = np.real(om)

            G = nx.from_numpy_matrix(adj_M)
            spec = nx.linalg.spectrum.laplacian_spectrum(G)

            om = np.sqrt(om)

            def rho(x, omega, C=1.0):
                res = 0.0
                gamma = 0.08
                for o in omega[1:]:
                    res += gamma / ((x - o) ** 2 + gamma ** 2)
                return C * res

            fy = []
            for x in fx:
                fy.append(rho(x, om))
            C = simps(fy, fx)
            fy = []
            for x in fx:
                fy.append(rho(x, om, 1 / C))

            return np.array(fy)

        def distance(x, y1, y2):
            return simps((y1 - y2) ** 2, x)

        n = len(subjects_135)
        A1 = np.zeros((n, n))
        for (u, v, w) in edges_1:
            u = subjects_135.index(u)
            v = subjects_135.index(v)
            A1[u, v] = 1.0
            A1[v, u] = 1.0

        A2 = np.zeros((n, n))
        for (u, v, w) in edges_2:
            u = subjects_135.index(u)
            v = subjects_135.index(v)
            A2[u, v] = 1.0
            A2[v, u] = 1.0

        fx = np.linspace(0.0, 10.0, 2 ** 10 + 1)
        fy_1 = calculate_rho_density(n, A1, fx)
        fy_2 = calculate_rho_density(n, A2, fx)
        d = distance(fx, fy_1, fy_2)

        # adj_M_shuf = switch_mtr(1000, e_mis, sec_n)
        # fy_shuf = calculate_rho_density(sec_n, adj_M_shuf, fx)

        # plt.figure(figsize=(6, 6))
        # plt.grid(alpha = 0.5, linestyle = '--', linewidth = 0.2, color = 'black')
        # plt.xlabel('omega')
        # plt.ylabel('rho')
        # title_text = "Eps(misclass, multis): %5.4f\n"%distance(fx, fy_mul, fy_mis)
        # title_text += "Misclassification and its shuffle: %5.4f\n"%distance(fx, fy_mis, fy_shuf)
        # title_text += "Multisection and misclass shuffle: %5.4f\n"%distance(fx, fy_mul, fy_shuf)
        # plt.title(title_text)

        # plt.plot(fx, fy_1, '-', c="C0", markersize=4.0, markerfacecolor='none', markeredgewidth=1.0, label="1")
        # plt.plot(fx, fy_2, '-', c="C1", markersize=4.0, markerfacecolor='none', markeredgewidth=1.0, label="2")
        # # plt.plot(fx, fy_shuf, '-', c="black", markersize=4.0, markerfacecolor='none', markeredgewidth=1.0, label="shuffled miscl")
        # plt.title(f'{d:f}')
        # plt.legend(loc="best")
        # # plt.savefig('Spectral_densities.png', dpi=300, bbox_inches = 'tight')
        # plt.show()

        return d

    if method == 'JI':
        return similarity_JI(edges_1, edges_2)
    elif method == 'clusters':
        return similarity_clusters(edges_1, edges_2)
    elif method == 'editing':
        return distance_editing(edges_1, edges_2)
    elif method == 'spectral':
        return distance_spectral(edges_1, edges_2)


if __name__ == "__main__":
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

    # norm_type = 'o_size'
    # norm_type = 'o_size_cutoff'
    norm_type = 'likelihood'

    clust_method = 'louvain'
    # clust_method = 'sciences'

    sim_key = 'JI'
    sim_key = 'clusters'
    M_range = list(range(100, 1201, 100))
    # M_range = [200]
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
        edges_cit_R = get_M_edges_regularized(data['cit']['edges'], data['cit']['sizes'], M, norm_type,
                                              is_directed=True)
        edges_citm = get_M_edges(data['cit_M']['edges'], data['cit_M']['sizes'], M, norm_type, is_directed=True)
        edges_citm_R = get_M_edges_regularized(data['cit_M']['edges'], data['cit_M']['sizes'], M, norm_type,
                                               is_directed=True)
        edges_mis = get_M_edges(data['mis']['edges'], data['mis']['sizes'], M, norm_type, is_directed=True)
        edges_mis_R = get_M_edges_regularized(data['mis']['edges'], data['mis']['sizes'], M, norm_type,
                                              is_directed=True)

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

