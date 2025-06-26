import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import sys

L = float(sys.argv[4])
T = float(sys.argv[3])
L_STEP = float(sys.argv[1])
T_STEP = float(sys.argv[2])

# os.system(f"./solver {L_STEP} {T_STEP} {T} {L}")

def frames():
    cnt = 0
    with open("out/data.out", mode="r") as file:
        for num, line in enumerate(file):
            # if num % int(1 / T_STEP / 4) == 0:
            cnt += 1
            yield (cnt, np.array(line[:-2].split(" "), dtype=np.float64))

fig, ax = plt.subplots()
# ax.set_xlabel("x")
# ax.set_ylabel("temp")
# plot_line,  = ax.plot(np.arange(0.0, 5.01, 0.5), np.zeros(11, dtype=np.float64))
# ax.set_ylim((0.0, 7.5))
# 
# def animate(data):
#     plot_line.set_ydata(data)
#     return [plot_line]

x = np.arange(0.0, L + 0.01, L_STEP)
segments = np.concatenate([np.column_stack([x[:-1], np.repeat(0.5, len(x) - 1)])[:, np.newaxis, :], np.column_stack([x[1:], np.repeat(0.5, len(x) - 1)])[:, np.newaxis, :]], axis=1)
lc = LineCollection(segments, capstyle="butt", linewidth=30, cmap="inferno")
_ , init_data = next(frames())
init_data = np.convolve(init_data, np.repeat(0.5, 2), mode="valid")
lc.set_array(init_data)
lc = ax.add_collection(lc)
fig.colorbar(lc)
ax.set_ylim((0.0, 1.0))
ax.set_xlim((0, L))
title = ax.set_title("t=0")
frame_len = 0.05

def animate(inp):
    step, data = inp
    data_proc = np.convolve(data, np.repeat(0.5, 2), mode="valid")
    lc.set_array(data_proc)
    title.set_text(f"t={step * frame_len:.3f}")
    return [lc, title]

gif_length = 10.0

interval = int(gif_length / (T / 0.05) * 1000)

anim = animation.FuncAnimation(fig, func=animate, frames=frames(), blit=True, interval=interval, cache_frame_data=False)
anim.save("out/heat.gif", writer="pillow")
