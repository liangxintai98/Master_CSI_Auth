import matplotlib.pyplot as plt
import numpy as np


labels = ['RAS_key_length', 'AES_key_length']
csi = [2.76, 2.75]
data1 = [4.72, 4.25]
data2 = [7.13,5.00]
data3 = [13.32,6.43]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3*width/8, csi, width/4, label='Men')
rects2 = ax.bar(x - width/8 , data1, width/4, label='Women')
rects3 = ax.bar(x + width/8, data2, width/4, label='no')
rects4 = ax.bar(x + 3*width/8, data3, width/4, label='no')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CPU usage(%)')
ax.set_ylim((0,15))
ax.set_title('CPU usage experiments with different key_length')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)

fig.tight_layout()

plt.grid(axis='y', color = '0.9', linestyle = '--')

plt.show()