import os

import numpy as np
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from datetime import datetime

from tqdm import tqdm

import cv2

print(f'Updated module {datetime.now()}')

plt.rcParams.update({'axes.linewidth' : 1,
                     'ytick.major.width' : 1,
                     'ytick.minor.width' : 1,
                     'xtick.major.width' : 1,
                     'xtick.minor.width' : 1,
                     'xtick.labelsize': 10, 
                     'ytick.labelsize': 10,
                     'axes.labelsize': 12,
                     'font.family': 'Serif',
                      'figure.figsize': (8,8)
                    })


types = ['X', 'O', 'B', 'A', 'F', 'G', 'K', 'M']
spectrals = np.array([1e99, 30000, 10000, 7500, 6000, 5200, 3700, 2400])
rgbs = [(1, 1, 1), # X, temp class just for setting upper bound 
        (175/255, 201/255, 1),       # O
        (199/255, 216/255, 1),       # B
        (1,       244/255, 243/255), # A 
        (1,       229/255, 207/255), # F 
        (1,       217/255, 178/255), # G 
        (1,       199/255, 142/255), # K 
        (1,       166/255, 81/255)]  # M

solar_mass_g = 1.9885 * 10**33

rxns = ['pp', 'cno', 'tri_alpha', 'c_alpha', 'n_alpha', 'o_alpha', 'ne_alpha', 'na_alpha', 'mg_alpha', 'si_alpha', 's_alpha',
         'ar_alpha', 'ca_alpha', 'ti_alpha', 'fe_co_ni', 'c12_c12', 'c12_o16', 'o16_o16']

# cmap1 = mpl.colormaps['Set1']
# cmap2 = mpl.colormaps['Set2']

# colors1 = cmap1(np.linspace(0,1, int(len(rxns)/2))) 
# colors2 = cmap2(np.linspace(0,1, int(len(rxns)/2)))
# rxn_colors = np.concat((colors1,colors2))

rxn_colors = ['darkred', 'tomato', 'limegreen', 'teal', 'dodgerblue', 'darkblue', 'darkorchid', 'purple', 'magenta',
               'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']






def plot_colors(y, xlim=[30000, 0], ax=None, alpha=0.5):
    if ax is None:
        ax = plt.gca()

    ax.fill_betweenx(y, spectrals[0], spectrals[1], color=rgbs[1], alpha=alpha)
    ax.fill_betweenx(y, spectrals[1], spectrals[2], color=rgbs[2], alpha=alpha)
    ax.fill_betweenx(y, spectrals[2], spectrals[3], color=rgbs[3], alpha=alpha)
    ax.fill_betweenx(y, spectrals[3], spectrals[4], color=rgbs[4], alpha=alpha)
    ax.fill_betweenx(y, spectrals[4], spectrals[5], color=rgbs[5], alpha=alpha)
    ax.fill_betweenx(y, spectrals[5], spectrals[6], color=rgbs[6], alpha=alpha)
    ax.fill_betweenx(y, spectrals[6], spectrals[7], color=rgbs[7], alpha=alpha)

    ax.set_xlim(xlim)






def plot_HR(DF, idx=None, fig=None, ax=None, xlim=[10000, 3500], ylim=[0.1, 10000], linecolor='black', dotcolor='red', age_plot=False, bysize=True, byrxn=True):
    # if fig is not None:
        # fig.clf()

    if fig is None:
        fig = plt.figure()
    
    if ax is None:
        ax=fig.gca()

    ax.cla()

    plot_colors(np.arange(ylim[0], ylim[1]), xlim=xlim, ax=ax)

    ax.plot(10**DF['log_Teff'], 10**DF['log_L'], lw=1, color=linecolor)

    if idx is not None:
        if bysize is False:
            s=20
        if bysize is True:
            s=10*10**DF.iloc[idx]['log_R']
        if byrxn is True:
            # rxns[np.argmax(DF.iloc[a][rxns])]
            dotcolor = rxn_colors[np.argmax(DF.iloc[idx][rxns])]


        ax.scatter(10**DF.iloc[idx]['log_Teff'], 10**DF.iloc[idx]['log_L'], color=dotcolor, zorder=100, s=s)

    ax.set_yscale('log')
    ax.set_ylim(ylim)

    ax.set_xlabel('Effective Temperature [K]')
    ax.set_ylabel(r'Luminosity [L / L$_\odot$]')




def plot_convection_circles(DF, idx, fig=None, base_color=None, ax=None, age_plot=False, age_scale=10**-9, age_label='Gyr', originalburnplot=False):
    # if fig is not None:
        # fig.clf()
        
    if fig is None:
        fig = plt.figure()


    colors = {'O':  (175/255, 201/255, 1),
            'B': (199/255, 216/255, 1),
            'A': (1,244/255, 243/255),
            'F': (1, 229/255, 207/255),
            'G': (1, 217/255, 178/255),
            'K': (1, 199/255, 142/255),
            'M': (1, 166/255, 81/255)
            }
        
    if base_color is None:
        base_color = colors['M']


    if ax is None:
        ax=fig.gca()
    
    ax.cla()

    initial_mass = DF.iloc[0]['star_mass']

    df = DF.iloc[idx]
    mass = df['star_mass']
    age = df['star_age']

    teff = 10**df.log_Teff
    index = np.argmin(spectrals > teff) 
    surface_color = rgbs[index]


    base = plt.Circle((0,0), mass, color=base_color)
    conv_core = plt.Circle((0,0), df['mass_conv_core'], color='gray')
    conv1_top = plt.Circle((0,0), df['conv_mx1_top'] * mass, color='gray')
    conv1_bot = plt.Circle((0,0), df['conv_mx1_bot'] * mass, color=base_color)
    conv2_top = plt.Circle((0,0), df['conv_mx2_top'] * mass, color='gray')
    conv2_bot = plt.Circle((0,0), df['conv_mx2_bot'] * mass, color=base_color)

    if originalburnplot:
        h_check = False
        if df['epsnuc_M_5'] != -20:
            h = mpl.patches.Annulus((0,0), df['epsnuc_M_8'] / solar_mass_g, (df['epsnuc_M_8'] - df['epsnuc_M_5']) / solar_mass_g, hatch='\\\\', alpha=0)
            h_check=True
        
        he_check=False
        if df['epsnuc_M_1'] != -20:
            he = mpl.patches.Annulus((0,0), df['epsnuc_M_4']  / solar_mass_g, (df['epsnuc_M_4'] - df['epsnuc_M_1']) / solar_mass_g, hatch='////', alpha=0)
            he_check=True
    else:
        if df['burn_type_1'] == 0:
            anyburning=False
        else:
            anyburning=True
            burnpatch = mpl.patches.Annulus

   
    surface = mpl.patches.Annulus((0,0), mass + 0.1, 0.05, color=surface_color)

    ax.add_patch(base)
    ax.add_patch(conv2_top)
    ax.add_patch(conv2_bot)
            
    ax.add_patch(conv1_top)
    ax.add_patch(conv1_bot)

    ax.add_patch(conv_core)

    if originalburnplot:
        if h_check:
            ax.add_patch(h)
        if he_check:
            ax.add_patch(he)
    
    if anyburning:
        ax.add_patch(burnpatch)

    ax.add_patch(surface)


    if age_plot is True:
        ax.text(x=0.8, y=0.95, s=f'Age: {age_scale*df['star_age']:.2f} {age_label}',
                    transform=ax.transAxes)

                
    ax.set_xlim(-1.2 * initial_mass, 1.2 * initial_mass)
    ax.set_ylim(-1.2 * initial_mass, 1.2 * initial_mass)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_both(DF, idx, fig=None, axs=None, age_plot=True):
    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    else:
        axs = fig.get_axes()

    plot_HR(DF, idx=idx, fig=fig, ax=axs[0])
    plot_convection_circles(DF, idx, fig=fig, ax=axs[1], age_plot=age_plot)

    # # fig.clf()
    # axs[0].cla()
    # axs[1].cla()


def create_image_folder(history_file, plotfunction, folder_name='./images', start=0, end=None, age_plot=False):
    if end is None:
        end = len(pd.read_table(history_file)) - 5
    
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    DF = pd.read_table(history_file, skiprows=5, sep=r'\s+')

    if plotfunction is not plot_both:
        fig=plt.figure()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
 

    for i in tqdm(range(start, end)):
        plotfunction(DF, idx=i, fig=fig, age_plot=age_plot)

        fig.savefig(f'{folder_name}/img{i}.png', dpi=100)
        

def create_video(image_folder, video_name='starmovie.avi', fps=15):
    images = [img for img in sorted(os.listdir(image_folder), key=lambda x: os.path.getmtime(os.path.join(image_folder, x))) if img.endswith((".jpg", ".jpeg", ".png"))]
    
    print("Images:", images)
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)),)
    video.release()
    cv2.destroyAllWindows()
    print(f"Video {video_name} generated successfully!")



def movie(history_file, plotfunction=plot_convection_circles, image_folder_name='./images', video_name='starmovie.avi', age_plot=False, fps=15, end=None):
    print(f'creating folder of images: {image_folder_name}')
    create_image_folder(history_file=history_file, plotfunction=plotfunction, folder_name=image_folder_name, age_plot=age_plot, end=end)

    print(f'creating video {video_name}')
    create_video(image_folder_name, video_name=video_name, fps=fps)

