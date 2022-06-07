from turtle import color
import numpy as np
import matplotlib.pyplot as plt

barwidth = 0.2
step_sizes = ['10', '30', '50', '70', '100']

data = np.load('method_deviations.npy', allow_pickle=True).item()

print()

section1 = [data[i][j] for i in data for j in data[i] if j == '0' and i in step_sizes] # least curve
section2 = [data[i][j] for i in data for j in data[i] if j == '1' and i in step_sizes] # medium curve
section3 = [data[i][j] for i in data for j in data[i] if j == '2' and i in step_sizes] # most curve

section1_avg = [np.average(np.array(x)) for x in section1]
section2_avg = [np.average(np.array(x)) for x in section2]
section3_avg = [np.average(np.array(x)) for x in section3]

section1_std = [np.std(np.array(x)) for x in section1]
section2_std = [np.std(np.array(x)) for x in section2]
section3_std = [np.std(np.array(x)) for x in section3]

print(section1_std)
# x position of bars
r1 = np.arange(len(section1))
r2 = [x + barwidth for x in r1]
r3 = [x + 2*barwidth for x in r1]
plt.grid(1, zorder=0)
plt.bar(r1, section1_avg, width=barwidth, color='cornflowerblue', edgecolor='black', yerr=section1_std, capsize=4, zorder=3)
plt.bar(r2, section2_avg, width=barwidth, color='limegreen', edgecolor='black', yerr=section2_std, capsize=4, zorder=3)
plt.bar(r3, section3_avg, width=barwidth, color='orange', edgecolor='black', yerr=section3_std, capsize=4, zorder=3)

plt.xticks([r + barwidth for r in range(len(section1_avg))], step_sizes)
plt.ylabel('Deviation [mm]')
plt.xlabel('Step size [mm]')

plt.show()


