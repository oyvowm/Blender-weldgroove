from turtle import color
import numpy as np
import matplotlib.pyplot as plt

barwidth = .7
#step_sizes = ['10', '30', '50', '70', '100']
step_sizes = ['10', '20']


data = np.load('method_deviations.npy', allow_pickle=True).item()


section1 = [data[i][j] for i in data for j in data[i] if j == '0' and i in step_sizes] # least curve
section2 = [data[i][j] for i in data for j in data[i] if j == '1' and i in step_sizes] # medium curve
section3 = [data[i][j] for i in data for j in data[i] if j == '2' and i in step_sizes] # most curve
section4 = [data[i][j] for i in data for j in data[i] if j == '3' and i in step_sizes] # least curve
section5 = [data[i][j] for i in data for j in data[i] if j == '4' and i in step_sizes] # medium curve
section6 = [data[i][j] for i in data for j in data[i] if j == '5' and i in step_sizes] # most curve

if len(step_sizes) > 1:
    section1 = np.average(np.array(section1), axis=0)
    section2 = np.average(np.array(section2), axis=0)
    section3 = np.average(np.array(section3), axis=0)
    section4 = np.average(np.array(section4), axis=0)
    section5 = np.average(np.array(section5), axis=0)
    section6 = np.average(np.array(section6), axis=0)


section1_avg = np.average(np.array(section1))
section2_avg = np.average(np.array(section2))
section3_avg = np.average(np.array(section3))
section4_avg = np.average(np.array(section4))
section5_avg = np.average(np.array(section5))
section6_avg = np.average(np.array(section6))

section1_std = np.std(np.array(section1))
section2_std = np.std(np.array(section2))
section3_std = np.std(np.array(section3))
section4_std = np.std(np.array(section4))
section5_std = np.std(np.array(section5))
section6_std = np.std(np.array(section6))

# x position of bars
#r1 = np.arange(6)
r1 = [x for x in range(1,7)]
ax = plt.gca()
ax.set_aspect(4)
plt.grid(1, zorder=0)
plt.bar(r1[0], section1_avg, width=barwidth, color='cornflowerblue', edgecolor='black', yerr=section1_std, capsize=4, zorder=3)
plt.bar(r1[1], section2_avg, width=barwidth, color='limegreen', edgecolor='black', yerr=section2_std, capsize=4, zorder=3)
plt.bar(r1[2], section3_avg, width=barwidth, color='orange', edgecolor='black', yerr=section3_std, capsize=4, zorder=3)
plt.bar(r1[3], section4_avg, width=barwidth, color='orange', edgecolor='black', yerr=section4_std, capsize=4, zorder=3)
plt.bar(r1[4], section5_avg, width=barwidth, color='limegreen', edgecolor='black', yerr=section5_std, capsize=4, zorder=3)
plt.bar(r1[5], section6_avg, width=barwidth, color='cornflowerblue', edgecolor='black', yerr=section6_std, capsize=4, zorder=3)

#plt.xticks([r + barwidth for r in range(len(section1_avg))], step_sizes)
plt.ylabel('Error [mm]')
plt.xlabel('Section')

plt.savefig("test.png",bbox_inches='tight')
plt.show()


