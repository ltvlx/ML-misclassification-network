import codecs, json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aux_network_manipulation import subjects_135


def make_adjL_cit_135():
    fname = 'adjL-citation.json'
    adjL = json.load(codecs.open(fname, 'r'))

    res = {}
    for u in adjL:
        if u not in subjects_135:
            continue
        else:
            res[u] = {}
            for v, w in adjL[u].items():
                if v not in subjects_135:
                    continue
                else:
                    res[u][v] = w

    json.dump(res, codecs.open('adjL-citation-135_filtered.json', 'w'), indent=2)


def make_mtr(adjL):
    M = np.zeros((135, 135), dtype=int)
    for u in adjL:
        for v in adjL[u]:
            i = subjects_135.index(u)
            j = subjects_135.index(v)
            if i != j:
                M[i, j] = adjL[u][v]
    return M


############################################################################################################
# # Filter subjects from the citation network
# make_adjL_cit_135()


############################################################################################################
# # Check the matrices
# adjL_cit = json.load(codecs.open('adjL-citation-135.json', 'r'))
# M_cit = make_mtr(adjL_cit)
#
# adjL_mis = json.load(codecs.open('adjL-misclass-LinSVC-0250.json', 'r'))
# M_mis = make_mtr(adjL_mis)
#
# for i in range(len(subjects_135)):
#     print(f"{i:3d}, {subjects_135[i]}, {M_cit[i,i]:6d}, {M_cit[i,i] / np.sum(M_cit[i,:]) :.3f}")
#
# fig, ax = plt.subplots(ncols=2, figsize=(12,6))
# ax[0].imshow(M_cit)
# ax[1].imshow(M_mis)
# plt.show()



############################################################################################################
# Plot barcharts with correct and incorrect articles by subjects
def figure_correct_incorrect_stacked(adjL_mis):
    sizes = {}
    for u in adjL_mis:
        tot = 0
        cor = 0
        mis = 0
        for v in adjL_mis[u]:
            w = adjL_mis[u][v]
            tot += w
            if u == v:
                cor += w
            else:
                mis += w
        sizes[u] = {
            "total": tot,
            "correct": cor,
            "misclassification": mis,
        }

    subj = sorted(sizes.keys())
    y_cor = []
    y_mis = []
    for key in subj:
        print(key, sizes[key]["total"], sizes[key]["correct"], sizes[key]["misclassification"])
        y_cor.append(sizes[key]["correct"])
        y_mis.append(sizes[key]["misclassification"])


    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(subj)))
    ax.bar(x, y_cor, color="C0", label="Correct")
    ax.bar(x, y_mis, color="C3", bottom=y_cor, label="Incorrect")
    ax.set_xlabel('Subject code')
    ax.set_ylabel('Number of articles')
    ax.set_xlim(-1, 135)
    ax.set_ylim(0.0, 8000)
    ax.legend(loc='upper right')

    ticks = [1, 12, 25, 36, 51, 68, 86, 104, 118]
    labels = [subj[i] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, ha="center")

    ax.grid(alpha=0.5, ls='--', lw=0.2, c='black')

    plt.savefig(f'subject-correct_incorrect-bars.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.savefig(f'subject-correct_incorrect-bars.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.show()


def figure_total_misIn_misOut(adjL_mis):
    subj = sorted(adjL_mis.keys())

    x = list(range(len(subj)))
    y = {
        "total": np.zeros(135, dtype=int),
        "mis_in": np.zeros(135, dtype=int),
        "mis_out": np.zeros(135, dtype=int),
    }
    for u in adjL_mis:
        i = subj.index(u)
        for v in adjL_mis[u]:
            j = subj.index(v)
            w = adjL_mis[u][v]
            y["total"][i] += w
            if u != v:
                y["mis_out"][i] += w
                y["mis_in"][j] += w


    fig, ax = plt.subplots(nrows=2, figsize=(9, 6))

    ax[0].bar(x, y["mis_out"], color="C1", label="from subject")
    ax[0].bar(x, -y["mis_in"], color="C2", label="into subject")
    ax[0].set_ylabel('Number of incorrectly\nclassified articles')
    ax[0].legend()

    colors = json.load(codecs.open("colors.json", 'r'))['original']
    data_total = {code: {"color": color, "x": [], "y": []} for code, color in colors.items()}
    for i, s in enumerate(subj):
        code = s[:2]
        data_total[code]["x"].append(i)
        data_total[code]["y"].append(y["total"][i])
    for code in data_total:
        x = data_total[code]["x"]
        _y = data_total[code]["y"]
        ax[1].bar(x, _y, color=data_total[code]["color"])
    ax[1].set_ylabel('Total number of articles')

    ticks = [1, 12, 25, 36, 51, 68, 86, 104, 118]
    labels = [subj[i] for i in ticks]
    for i in range(2):
        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(labels, ha="center")
        ax[i].grid(alpha=0.5, ls='--', lw=0.2, c='black')
        ax[i].set_xlim(-1, 135)
    ax[0].set_ylim(-6000, 6000)
    ax[1].set_ylim(0.0, 8000)
    ax[1].set_xlabel('Subject code')



    #
    # for i in range(len(subj)):
    #     print(f"{subj[i]},{y['total'][i]},{y['total'][i] - y['mis_out'][i]},{y['mis_out'][i]},{y['mis_in'][i]}")
    df = pd.DataFrame(
        data={
            "subject": subjects_135,
            "total articles": y['total'],
            "correct": y['total'] - y['mis_out'],
            "misclass out": y['mis_out'],
            "misclass in": y['mis_in'],
        },
    )
    df.to_csv("classification_data_overview.csv", index=False)

    plt.tight_layout()
    # plt.savefig(f'subject-mis_in_out-bars.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
    # plt.savefig(f'subject-mis_in_out-bars.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.show()





adjL_mis = json.load(codecs.open('adjL-misclass-LinSVC-0250.json', 'r'))

# figure_correct_incorrect_stacked(adjL_mis)
figure_total_misIn_misOut(adjL_mis)

