import matplotlib.pyplot as plt
import cv2
import numpy as np 

def show_points_plt(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    

def show_points(output_file, pts_2D):
    image = cv2.imread(output_file, cv2.IMREAD_UNCHANGED) # [H, W, 3] or [H, W, 4]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points_plt(pts_2D, plt.gca())
    plt.axis('off')
    plt.savefig(output_file.replace('.png','_pts.png'))
    plt.close()


def show_mask_plt(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_mask(output_file, mask):
    image = cv2.imread(output_file, cv2.IMREAD_UNCHANGED) # [H, W, 3] or [H, W, 4]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask_plt(mask, plt.gca())
    plt.axis('off')
    plt.savefig(output_file.replace('.png','_mask.png'))
    plt.close()
