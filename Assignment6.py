# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage import feature

# %%
def task1_1(img):
    """
    straight line Hough Transform. The input image has to be a edge detection
    (means a binary image).
    """
    #define variables
    width, height = img.shape
    thetas = np.deg2rad(np.linspace(-90, 90, 181)) #angle
    diag = np.ceil(np.sqrt(width**2 + height**2)) #max distance is diagonal: pythagoras
    rhos = np.arange(-diag, diag+1) #distance
    output = np.zeros((len(rhos), len(thetas)))

    #run over all non zero values -> edges
    edges_y, edges_x = np.nonzero(img)
    for i in range(len(edges_y)):
        x = edges_x[i]
        y = edges_y[i]

      #run over all thetas cause of the formula: rho = x*cos(theta) + y*sin(theta)
        for t in range(len(thetas)):
            theta = thetas[t]
            rho = x*np.cos(theta) + y*np.sin(theta)
            #rho index
            r = np.argmin(np.absolute(rhos-rho))
            # r = np.nonzero(np.abs(rhos - rho) == np.min(np.abs(rhos - rho)))[0][0]
            #hough transform:output[r, t] will be added by 1
            output[r, t] += 1

    return output, thetas, rhos #thus we have the same output as scikit-image

# %%
def task1_2():
    img = imread('Week 7/cross.png', as_gray=True)
    methods = [task1_1, hough_line]

    plt.figure(dpi=200)
    plt.imshow(img, cmap='gray', aspect='auto')
    plt.title('Origin')
    plt.colorbar()

    titles = ['my implementation', ['scikit-image']]
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), dpi=200)
    for i, ax in enumerate(axes.flat):
        h, theta, dist = methods[i // 2](img)
        if i % 2 == 0:
            im = ax.imshow(h, cmap='gray', aspect='auto')
            ax.set_title('Hough transform with {}'.format(titles[i // 2]))
            fig.colorbar(im , ax=ax)
        else:
            ax.set_title('Overlayed result with {}'.format(titles[i // 2]))
            imo = ax.imshow(img, cmap='gray', aspect='auto')
            _,bestTheta, bestD = hough_line_peaks(h, theta, dist)
            x = np.arange(5, 95, 1)
            y1 = (bestD[0] - x * np.cos(bestTheta[0])) / np.sin(bestTheta[0])
            y2 = (bestD[1] - x * np.cos(bestTheta[1])) / np.sin(bestTheta[1])
            ax.plot(x, y1 ,'-r')
            ax.plot(x, y2 , '-r')

            fig.colorbar(imo, ax=ax)

# %%
def make_seg(shape, x_cord, y_cord, rad):
    mask = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for c in range(len(x_cord)):
                r_ref = np.linalg.norm([i-y_cord[c], j-x_cord[c]])
                if r_ref < rad[c]:
                    mask[i,j] = 1
    return mask

def task1_3():
    img = imread('Week 7/coins.png', as_gray=True)

    radii = np.arange(3,30,2)
    edges = feature.canny(img, sigma=3, low_threshold=10, high_threshold=50)
    hough_res = hough_circle(edges, radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii_par = hough_circle_peaks(hough_res, radii, total_num_peaks=10)

    img_seg = make_seg(img.shape, cx, cy, radii_par)
    
    images = [img, edges, hough_res[0], img_seg]
    titles = ['Origin', 'After canny', 'Hough circle based on edges', 'Coins segmentation']


    rows = 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(13,8), dpi=200)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(images[i], cmap='gray', aspect='auto')
        ax.set_title(titles[i])
        fig.colorbar(im , ax=ax)

# %%
if __name__== "__main__":
    task1_2()
    task1_3()


