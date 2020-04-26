import numpy as np
import csv
import os
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import sys

def fig_spt_vc(latmin,latmax,lonmin,lonmax,inputfile,outputfig):
    n_lat = int(latmax - latmin)
    n_lon = int(lonmax - lonmin)
    i,j = 0,0
    data = np.zeros((n_lat,n_lon))
    with open(inputfile,'r') as f:
         reader = csv.reader(f) 
         for row in reader:
             for col in row:
                 if col == "NA":
                    data[i,j] = np.nan
                 else:
                    data[i,j] = float(col)
                 j= j+1
             i = i+1  
             j = 0 
    fig = plt.figure(figsize=(10,6))
    #fig.subplots_adjust(wspace=0.1,hspace=0.1)
    gs  = GridSpec(2,1, left=0.09, right=0.98,top=0.95,bottom=0.01, height_ratios=[40,1], wspace=0.25,hspace=0.4)
    ax  = fig.add_subplot(gs[0,0])
    plt.title("Variance contribution",size=14)
    #m = Basemap(projection='robin', resolution='l', area_thresh=10000,lat_0=0,lon_0=0, ax=ax)
    m = Basemap(projection='cyl', resolution='l', area_thresh=10000, llcrnrlon=lonmin, urcrnrlon=lonmax,llcrnrlat=latmin, urcrnrlat=latmax,ax=ax)
    r_lat = np.arange(latmin, latmax)
    r_lon = np.arange(lonmin, lonmax)
    x, y  = m(*np.meshgrid(r_lon, r_lat))
    lim = np.linspace(0, 6, 7)
    n_color = ["r", "sandybrown", "lightgreen", "lightseagreen", "dodgerblue", "b"]
    my_colors = mpl.colors.ListedColormap(['b', 'tomato', 'darkturquoise', 'yellowgreen', 'seagreen', 'silver'], 'indexed')
    ctf_sd = ax.contourf(x, y, data.squeeze(), lim, cmap=my_colors, zorder=1)
    # ctf_sd = ax.contourf(x, y, dat_plot_cv_new.squeeze(), lim, cmap='rainbow')
    m.drawcoastlines(linewidth=2, zorder=3)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color="white", lakes=True, zorder=2)

    cbar_ax2 = fig.add_axes([0.17, 0.1, 0.7, 0.02])
    cb_2 = fig.colorbar(ctf_sd, cax=cbar_ax2, orientation="horizontal")
    cb_2.set_ticks(np.linspace(0.5, 5.5, 6))

    cb_2.set_ticklabels(['Scalar\n(precipitation)', 'Scalar\n(temperature)', 'Baseline\nresidence time', 'CUE', 'GPP','C storage\npotential'])
    for l in cb_2.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(8)
    fig.tight_layout()
    #plt.show()
    plt.savefig(outputfig,dpi=300)
    return "success!"

if __name__ == '__main__':
    args          = sys.argv
    latmin        = int(args[1])
    latmax        = int(args[2])
    lonmin        = int(args[3])
    lonmax        = int(args[4])
    input_data    = args[5]
    out_figure    = args[6]
    re_fig        = fig_spt_vc(latmin,latmax,lonmin,lonmax,input_data,out_figure)
    print(latmin,latmax,lonmin,lonmax,input_data,out_figure)
    print(re_fig)
