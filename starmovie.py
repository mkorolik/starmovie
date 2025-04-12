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






def plot_convection_circles(DF, idx, fig=None, base_color=None, ax=None, age_plot=False, age_scale=10**-9, age_label='Gyr'):
    if fig is None:
        fig = plt.figure()

    fig.clf()

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
    


    initial_mass = DF.iloc[0]['star_mass']

    df = DF.iloc[idx]
    mass = df['star_mass']
    age = df['star_age']

    teff = 10**df.log_Teff
    index = np.argmin(spectrals > teff) 
    surface_color = rgbs[index]


    base = plt.Circle((0,0), mass, color=base_color)
    conv_core = plt.Circle((0,0), df['mass_conv_core'], color='firebrick')
    conv1_top = plt.Circle((0,0), df['conv_mx1_top'] * mass, color='firebrick')
    conv1_bot = plt.Circle((0,0), df['conv_mx1_bot'] * mass, color=base_color)
    conv2_top = plt.Circle((0,0), df['conv_mx2_top'] * mass, color='firebrick')
    conv2_bot = plt.Circle((0,0), df['conv_mx2_bot'] * mass, color=base_color)

    h_check = False
    if df['epsnuc_M_5'] != -20:
        h = mpl.patches.Annulus((0,0), df['epsnuc_M_8'] / solar_mass_g, (df['epsnuc_M_8'] - df['epsnuc_M_5']) / solar_mass_g, hatch='\\\\', alpha=0)
        h_check=True
    
    he_check=False
    if df['epsnuc_M_1'] != -20:
        he = mpl.patches.Annulus((0,0), df['epsnuc_M_4']  / solar_mass_g, (df['epsnuc_M_4'] - df['epsnuc_M_1']) / solar_mass_g, hatch='////', alpha=0)
        he_check=True
   
    surface = mpl.patches.Annulus((0,0), mass + 0.1, 0.05, color=surface_color)

    ax.add_patch(base)
    ax.add_patch(conv2_top)
    ax.add_patch(conv2_bot)
            
    ax.add_patch(conv1_top)
    ax.add_patch(conv1_bot)

    ax.add_patch(conv_core)

    if h_check:
        ax.add_patch(h)
    if he_check:
        ax.add_patch(he)

    ax.add_patch(surface)


    if age_plot is True:
        ax.text(x=0.8, y=0.95, s=f'Age: {age_scale*df['star_age']:.2f} {age_label}',
                    transform=ax.transAxes)

                
    ax.set_xlim(-1.2 * initial_mass, 1.2 * initial_mass)
    ax.set_ylim(-1.2 * initial_mass, 1.2 * initial_mass)
    ax.set_xticks([])
    ax.set_yticks([])



def create_image_folder(history_file, folder_name='./images', start=0, end=None, age_plot=False):
    if end is None:
        end = len(pd.read_table(history_file)) - 5
    
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    DF = pd.read_table(history_file, skiprows=5, sep=r'\s+')


    fig=plt.figure()

    for i in tqdm(range(start, end)):
        plot_convection_circles(DF, i, fig=fig, age_plot=age_plot)

        fig.savefig(f'{folder_name}/img{i}.png', dpi=100)


def create_video(image_folder, video_name='starmovie.avi', fps=15):
    images = [img for img in sorted(os.listdir('./images'), key=lambda x: os.path.getmtime(os.path.join('./images', x))) if img.endswith((".jpg", ".jpeg", ".png"))]
    
    print("Images:", images)
    
    for img in images:
        if img.endswith(".pdf"):
            pages = convert_from_path(img)
            for page in pages:
                page.save("{img}.jpg", "jpg")

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    # Appending images to video
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)),)

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print(f"Video {video_name} generated successfully!")



def movie(history_file, image_folder_name='./images', video_name='starmovie.avi', age_plot=False, fps=15, end=None):
    print(f'creating folder of images: {image_folder_name}')
    create_image_folder(history_file=history_file, folder_name=image_folder_name, age_plot=age_plot, end=end)

    print(f'creating video {video_name}')
    create_video(image_folder_name, video_name=video_name, fps=fps)

