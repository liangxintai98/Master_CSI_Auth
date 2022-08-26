import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def heatmap3d(subcarrier_new,phase_error_output):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)

    X = np.array(subcarrier_new)
    frame_no = []
    line_no = np.shape(phase_error_output)[0]
    for i in range(line_no):
        frame_no.append(i)
    Y = np.array(frame_no)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(phase_error_output) 

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-np.pi, np.pi)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.00f}')

    # Add a color bar which maps values to colors.
    cbar_ax = fig.add_axes([0.84, 0.2, 0.03, 0.55])
    fig.colorbar(surf,cbar_ax)

    # plt.savefig('/Users/liangxintai/Desktop/1.png',dpi=300)

    plt.show()
