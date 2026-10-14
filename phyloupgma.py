"""Week 6 - UPGMA phylogenetic tree from the Slide 25 multiple alignment."""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

aln = {"S1": "A-CATATC-AT-", "S2": "A-GATATT-AG-",
       "S3": "AACAGATC-T--", "S4": "G-CAT--CGATT"}
names = list(aln)

def dist(a, b):
    """Differing columns; columns that are gaps in BOTH sequences are ignored."""
    return sum(1 for x, y in zip(a, b) if not (x == y == "-") and x != y)

D = {frozenset(p): dist(aln[p[0]], aln[p[1]])
     for p in itertools.combinations(names, 2)}
M = np.array([[0 if a == b else D[frozenset((a, b))] for b in names] for a in names], float)
print("Distance matrix:\n", M)

def upgma(names, D):
    D, size, height = dict(D), {n: 1 for n in names}, {n: 0.0 for n in names}
    tree, clusters = {n: n for n in names}, list(names)
    while len(clusters) > 1:
        a, b = min(itertools.combinations(clusters, 2), key=lambda p: D[frozenset(p)])
        d = D[frozenset((a, b))]; h = d / 2; new = f"({a},{b})"
        tree[new] = (tree[a], tree[b], h)          # (left, right, height)
        size[new], height[new] = size[a] + size[b], h
        for c in clusters:                          # size-weighted average
            if c not in (a, b):
                D[frozenset((new, c))] = (size[a]*D[frozenset((a, c))] +
                                          size[b]*D[frozenset((b, c))]) / size[new]
        clusters = [c for c in clusters if c not in (a, b)] + [new]
        print(f"merge {a} + {b}: d = {d:.3f}, height = {h:.3f}")
    return tree[clusters[0]]

root = upgma(names, D)

# --- cross-check with SciPy's average linkage (= UPGMA) ---
Z = linkage(squareform(M), method="average")
print("SciPy heights:", Z[:, 2] / 2)

# --- draw the tree (leaf order follows the tree) ---
ycount = [0]
def draw(node, ax):
    """returns (x_position_of_node, height)"""
    if isinstance(node, str):
        y = ycount[0]; ycount[0] += 1
        ax.text(-0.08, y, node, ha="right", va="center", fontsize=13, fontweight="bold")
        return y, 0.0
    l, r, h = node
    yl, hl = draw(l, ax); yr, hr = draw(r, ax)
    ax.plot([hl, h, h, hr], [yl, yl, yr, yr], "k-", lw=2)   # elbow connector
    ax.plot(h, (yl + yr) / 2, "o", color="#c0392b", ms=7)
    ax.text(h + 0.05, (yl + yr) / 2 + 0.12, f"h={h:.2f}", fontsize=9, color="#c0392b")
    return (yl + yr) / 2, h

fig, ax = plt.subplots(figsize=(7, 3.6))
draw(root, ax)   # x = height, y = leaf index
ax.set_xlim(-0.05, root[2] + 0.9); ax.set_ylim(-0.6, 3.6)
ax.set_xlabel("UPGMA height (distance / 2)")
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]
ax.set_title("UPGMA tree of S1-S4 (Slide 25 alignment)")
plt.tight_layout(); plt.savefig("tree.png", dpi=200)