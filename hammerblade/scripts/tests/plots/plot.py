import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import seaborn as sns
import numpy as np
import sys
fig,ax = plt.subplots(1)
hb_mc = np.zeros((8,16))
print(hb_mc)
print()
tiles = [[(y,x) for x in range(16)] for y in range(8)]
with open("dram_stall.txt", "r") as f:
    logs = f.readlines()
entries = [720856]
for log in logs:
    log = log.split()
    print(log)
    bsg_y = int(log[0].split("_")[1])
    bsg_x = int(log[0].split("_")[2])
    stall = int(log[1])
    entries.append(stall)
    tiles[bsg_y][bsg_x] = stall
for row in tiles:
    print(row)
print()
max_stall = max(entries)
print("max = ")
print(max_stall)
print()
for y in range(8):
    for x in range(16):
        hb_mc[y,x] = tiles[y][x] / float(max_stall)
print(hb_mc)
ax = sns.heatmap(hb_mc, linewidth=0.5, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
ax.set_xlim(-1,17)
ax.set_ylim(17,-1)
plt.show()
