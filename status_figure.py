from matplotlib.animation import FuncAnimation
# from utils.plots import animate
import pandas as pd
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from itertools import count
import matplotlib
plt.style.use("fivethirtyeight")
plt.style.use('dark_background')
matplotlib.use('TkAgg')
index = count()

def animate(i):
    conf_list = pd.read_csv('./conf_list.csv')
    # print(conf_list[:,0])
    plt.cla()
    plt.grid(False)
    # plt.figure(figsize=(19,7))
    plt.plot(conf_list["climb"] if len(conf_list["climb"]) <= 200 else conf_list["climb"][-200:], label="climb", color="red", fillstyle='full', linewidth=1)
    plt.plot(conf_list["fall"] if len(conf_list["fall"]) <= 200 else conf_list["fall"][-200:], label="fall", color="orange", linewidth=1, fillstyle='full')
    plt.plot(conf_list["walk"] if len(conf_list["walk"]) <= 200 else conf_list["walk"][-200:], label="walk", color="green", linewidth=1, fillstyle='full')
    # plt.plot(conf_list["run"] if len(conf_list["run"]) <= 200 else conf_list["run"][-200:], label="run", color="green", linewidth=1, fillstyle='full')
    # plt.plot(conf_list["sit"] if len(conf_list["sit"]) <= 200 else conf_list["sit"][-200:], label="sit", color="green", linewidth=1, fillstyle='full')

    plt.legend(loc="upper left")
    # plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, frames=np.arange(5, 10), interval=1)

plt.tight_layout()
plt.show()


import random
# from itertools import count
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import matplotlib
# plt.style.use("fivethirtyeight")
# matplotlib.use('TkAgg')
#
# x_vals = []
# y_vals = []
#
# index = count()
#
#
# def animate(i):
#     data = pd.read_csv('data.csv')
#     x = data['x_value']
#     y1 = data['total_1']
#     y2 = data['total_2']
#
#     plt.cla()
#
#     plt.plot(x, y1, label='Channel 1')
#     plt.plot(x, y2, label='Channel 2')
#
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#
#
# ani = FuncAnimation(plt.gcf(), animate, frames=np.arange(0, 100), interval=10)
#
# plt.tight_layout()
# plt.show()


#
# xx=np.arange(0,10,0.01)
# yy=xx*np.exp(-xx)
#
# path = Path(np.array([xx,yy]).transpose())
# patch = PathPatch(path, facecolor='none')
# plt.gca().add_patch(patch)
#
# im = plt.imshow(xx.reshape(yy.size,1),
#                 cmap=plt.cm.Reds,
#                 interpolation="bicubic",
#                 origin='lower',
#                 extent=[0,10,-0.0,0.40],
#                 aspect="auto",
#                 clip_path=patch,
#                 clip_on=True)
# plt.show()