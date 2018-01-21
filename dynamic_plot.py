import matplotlib.pyplot as plt
import numpy as np

class dynamic_plot():
    def __init__(self, xlim=(-100,2000), ylim=(-100,2000)):
        plt.ion()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        self.hl, = ax.plot([], [])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    def update_line(self, new_x, new_y):

        self.hl.set_xdata(np.append(self.hl.get_xdata(), new_x))
        self.hl.set_ydata(np.append(self.hl.get_ydata(), new_y))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



dp = dynamic_plot()

for i in range(100,1000):
    y = np.random.randint(0, i+1)

    # hl.set_xdata(np.append(hl.get_xdata(), i))
    # hl.set_ydata(np.append(hl.get_ydata(), y))
    dp.update_line(i, y)

    # hl.set_ydata( y)
    # fig.canvas.draw()
