# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_minimum
from skimage.feature import peak_local_max
from skimage.transform import warp
from scipy.ndimage import gaussian_filter
from skimage import util, img_as_float

# %%
# Helper functions for Question 2
# Translation matrix
def Tt(t):
    """
    :param t: translation vector
    :return: a 3x3 matrix.
    """
    tt = np.matrix(np.diag(np.ones(3)))
    tt[0,2] = t[0]; tt[1,2] = t[1]
    return tt

# Scaling matrix
def Ts(s):
    """
    Create a 3x3 matrix with the first and second rows and columns set to the value of the input s
    
    :param s: the scale factor in the x and y directions
    :return: a matrix of size 3x3.
    """
    ts = np.matrix(np.diag(np.ones(3)))
    ts[0,0] = s; ts[1,1] = s
    return ts

# Rotation matrix
def Tr(theta):
    """
    Given an angle theta, return a 3x3 rotation matrix
    
    :param theta: rotation angle
    :return: A matrix
    """
    # print(tr)
    return np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def task2_2():
    img = np.ones((51, 51))
    img = np.pad(img, pad_width=100, constant_values=(0,0))

    theta = np.pi / 10; t = (10.4, 15.7); s = 2; c = (np.ceil(img.shape[0] / 2), np.ceil(img.shape[1] / 2))

    res = warp(img, (Tt(t) @ Tt(c) @ Tr(theta) @ Ts(s) @ (Tt(c).I)).I, order=0)

    imgs = [img, res]
    titles = ['Origin', 'Transformed image']
    fig, axes = plt.subplots(1, 2, layout='constrained', dpi=200)
    for index, ax in enumerate(axes.flat):
        ax.imshow(imgs[index], cmap='gray')
        ax.set_title(titles[index])

    fig.savefig('task2-2.png', bbox_inches='tight')

# %%
# Helper function for Question 3
def R(img, k, sigma, alpha):
    Lx = gaussian_filter(img, sigma, order=(0,1))
    Ly = gaussian_filter(img, sigma, order=(1,0))
    Lxy = gaussian_filter(Lx * Ly, k * sigma)
    Lxx = gaussian_filter(Lx ** 2, k * sigma)
    Lyy = gaussian_filter(Ly ** 2, k * sigma)
    detA = Lxx * Lyy - Lxy ** 2
    traceA = Lxx + Lyy
    return sigma ** 4 * (detA - alpha * traceA ** 2)

def find_corner_harris(img, k, sigma, alpha, num_points=350):
    scale_space = np.array([
        R(img, k=k, sigma=s, alpha=alpha) for s in sigma
        ])
    return peak_local_max(scale_space, num_peaks=num_points)

def task3_1():
    img = img_as_float(
        imread('Week 8/modelhouses.png', as_gray=True))
    scale_levels = 30
    sigma = np.logspace(0, 5, scale_levels, base=2)
    # print(sigma)
    ks = [1, 3, 5]
    alphas = [0.05, 0.15, 0.3]
    # ks = [1]
    # alphas = [0.05]
    # optimal performance
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained', dpi=200)
    imoutput = ax.imshow(img, cmap="gray")
    coor = find_corner_harris(img, k=1, sigma=sigma, alpha=0.05)
    ax.plot(coor[:, 2], coor[:, 1], 'r.')
    x,y = coor.shape
    for i in range(x):
        circle = plt.Circle((coor[i, 2], coor[i, 1]), 
                        sigma[coor[i, 0]],
                        color='r',
                        fill = False)
        ax.add_artist(circle)
    ax.set_title('$K=1, \\alpha = 0.05$')
    fig.colorbar(imoutput, ax=ax)
    fig.savefig('task3-optimal.png', bbox_inches='tight')
    # tests
    fig, axes = plt.subplots(3 , 3, figsize=(15, 15), dpi=200)
    for index, ax in enumerate(axes.flat):
        imoutput = ax.imshow(img, cmap="gray", aspect='auto')
        if index < 3:
            coor = find_corner_harris(img, k=ks[0], sigma=sigma, alpha=alphas[index])
            ax.set_title(f'$K={ks[0]}, \\alpha = {alphas[index]}$')
        elif index <= 5:
            coor = find_corner_harris(img, k=ks[1], sigma=sigma, alpha=alphas[index-3])
            ax.set_title(f'$K={ks[1]}, \\alpha = {alphas[index-3]}$')
        else:
            coor = find_corner_harris(img, k=ks[2], sigma=sigma, alpha=alphas[index-6])
            ax.set_title(f'$K={ks[2]}, \\alpha = {alphas[index-6]}$')
        fig.colorbar(imoutput, ax=ax)
        ax.plot(coor[:, 2], coor[:, 1], 'r.')
        x,y = coor.shape
        for i in range(x):
            circle = plt.Circle((coor[i, 2], coor[i, 1]), 
                            sigma[coor[i, 0]],
                            color='r',
                            fill = False)
            ax.add_artist(circle)
    fig.savefig('task3-tests.png', bbox_inches='tight')

# %%
def task4_1():
    img = imread('Week 8/hand.tiff', as_gray=True)
    sp_noise = util.random_noise(img, mode="s&p")
    global_threshold = 87
    noise_global_threshold = 0.14

    binary = img < global_threshold
    binary_sp = sp_noise < noise_global_threshold

    imgs = [img, sp_noise, binary, binary_sp]
    titles = ['Origin', 'Noise', 'Global threshold', 'Global threshold with noise']

    fig, axes = plt.subplots(2, 2, figsize = (8,8), layout='constrained', dpi=200)
    for index, ax in enumerate(axes.flat):
        imoutput = ax.imshow(imgs[index], cmap='gray', aspect='auto')
        ax.set_title(titles[index])
        fig.colorbar(imoutput, ax=ax)
    fig.savefig('task4-1.png', bbox_inches='tight')

# %%
# Helper functions for Question 4
def _histogram_based_segmentation(img, counts, bin_edges): 
    """
    Given an image, the function calculates the threshold that maximizes the inter-class variance
    between the foreground and background pixels.
    
    :param img: The image to be segmented
    :param counts: The histogram counts for the image
    :param bin_edges: The edges of the bins in which the samples are counted
    :return: The threshold and the binary image.
    """
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(counts * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((counts * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = np.ceil(bin_mids[:-1][index_of_max_val])
    binary = img > threshold
    return threshold, binary

def task4_3():

    img_coin = imread('Week 8/coins.png', as_gray=True)
    img_euro = imread('Week 8/overlapping_euros1.png', as_gray=True)

    # compute the histograms with numpy
    his_coin, coin_edges = np.histogram(img_coin, bins  = 256)
    his_euro, eu_edges = np.histogram(img_euro, bins  = 256)
    # compute the threshhold
    thresh_coin, img_coin_thresh = _histogram_based_segmentation(img_coin, counts=his_coin, bin_edges=coin_edges)
    thresh_euro, img_euro_thresh = _histogram_based_segmentation(img_euro, counts=his_euro, bin_edges=eu_edges)

    imgs_histo = [img_coin, img_euro]
    thresholds = [thresh_coin, thresh_euro]
    titles = ['coins.png','overlapping_euros1.png','intensity histogram \n of coins','intensity histogram \n of overlapping_euros1']
    # Plot the histograms
    fig, axes = plt.subplots(2, 2, figsize = (14,10), layout='constrained', dpi=200)
    for index, ax in enumerate(axes.flat):
        if index < 2:
            res = ax.imshow(imgs_histo[index], cmap='gray', aspect='auto')
            fig.colorbar(res , ax=ax)
        else:
            res = ax.hist(imgs_histo[index-2].ravel(), 256, [0,257])
            ax.axvline(thresholds[index-2], color='r')
            ax.set_xlabel('intensity')
            ax.set_ylabel('times')
        ax.set_title(titles[index])
    fig.savefig('task4-3-histo.png', bbox_inches='tight')
    # Plot the threshhold
    imgs_thres = [img_coin, img_euro, img_coin_thresh, img_euro_thresh]
    titles_thres = ['coins.png', 'overlapping_euros1.png',
     f'Found threshhold = {thresh_coin}', f'Found threshhold = {thresh_euro}']

    fig, axes = plt.subplots(2, 2, figsize = (14,10), layout='constrained', dpi=200)
    for index, ax in enumerate(axes.flat):
        res = ax.imshow(imgs_thres[index], cmap='gray', aspect='auto')
        ax.set_title(titles_thres[index])
        fig.colorbar(res, ax=ax)

    fig.savefig('task4-3-thres.png', bbox_inches='tight')

# %%
# Helper functions for Quesiont 5
def _degradation(img, kernel, noise_level=0.05):
    """
    Given an image, a kernel, and a noise level, 
    this function will return the degraded image and the kernel
    
    :param img: The image to be degraded
    :param kernel: the degradation function, e.g. a motion blur kernel
    :param noise_level: the standard deviation of the noise added to the image
    :return: the degraded image and the transfer function.
    """
    noise = np.random.normal(0, noise_level, img.shape)
    F = np.fft.fft2(img / img.max())
    H = np.fft.fft2(kernel / kernel.sum(), img.shape)
    N = np.fft.fft2(noise)
    G = H * F + N
    return G, H

def _direct_inverse_filtering(img, kernel, noise_level=0.05):
    """
    It performs direct inverse filtering on the degraded image.
    
    :param img: the degraded image
    :param kernel: the degradation operator, e.g. a blurring kernel
    :param noise_level: the noise level of the input degraded image
    :return: The direct inverse filtering method returns the restored image.
    """
    G, H = _degradation(img, kernel, noise_level)
    return G / H

def _wiener_filter(img, kernel, K=0.05, noise_level=0.05):
    """
    Given an image, a degradation kernel, and a noise level, 
    return the Wiener filtered image
    
    :param img: The degraded image
    :param kernel: the degradation kernel (e.g. motion blur kernel)
    :param K: The parameter that controls the strength of the denoising
    :param noise_level: the noise level of the degraded image
    :return: The Wiener filter.
    """
    G, H = _degradation(img, kernel, noise_level)
    return (1 / H) * H * np.conj(H) / (H * np.conj(H) + K) * G

def task5_1():
    img = imread('Week 8/trui.png', as_gray=True)
    kernels = [np.ones((3,3)), np.diag(np.ones(5))]
    noises = [0, 0.005, 0.01, 0.1]
    kernels_name = ['np.ones((3,3))', 'np.diag(np.ones(5))']
    
    plt.figure(dpi=200)
    plt.imshow(img, cmap='gray')
    plt.title('Origin')
    plt.colorbar()
    plt.savefig('origin.png', bbox_inches='tight')

    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize = (15,7), layout='constrained', dpi=300)
    i = 0
    for index, ax in enumerate(axes.flat):
        # if index == 0:
        #     imoutput = ax.imshow(img, cmap='gray', aspect='auto')
        #     fig.colorbar(imoutput, ax=ax)
        #     ax.set_title(titles[index])
        # else:
        res, _ = _degradation(img, kernels[i],noises[index%cols])
        imoutput = ax.imshow(abs(np.fft.ifft2(res)), cmap='gray', aspect='auto')
        ax.set_title('kernel = {} \n noise level = {}'.format(kernels_name[i], noises[index%cols]),)
        fig.colorbar(imoutput, ax=ax)
        if index == cols-1:
            i+=1
    fig.savefig('task5-1.png',bbox_inches='tight')

def task5_2():
    img = imread('Week 8/trui.png', as_gray=True)
    kernels = np.diag(np.ones(5))
    noises = [0, 0.0005, 0.005 , 0.01, 1]

    fig, axes = plt.subplots(2, 3, figsize = (12,7), layout='constrained', dpi=200)
    for index, ax in enumerate(axes.flat):
        if index == 0:
            deg, _ = _degradation(img, kernels, noise_level=noises[0])
            imoutput = ax.imshow(abs(np.fft.ifft2(deg)), cmap='gray', aspect='auto')
            fig.colorbar(imoutput, ax=ax)
            ax.set_title('Degraded Image with \n kernel = np.diag(np.ones(5))')
        else:
            res = _direct_inverse_filtering(img, kernels, noise_level=noises[index-1])
            imoutput = ax.imshow(abs(np.fft.ifft2(res)), cmap='gray', aspect='auto')
            ax.set_title('Recovered with \n noise_level = {}'.format(noises[index-1]))
            fig.colorbar(imoutput, ax=ax)
    fig.savefig('task5-2.png',bbox_inches='tight')

def task5_3():
    img = imread('Week 8/trui.png', as_gray=True)
    kernel = np.diag(np.ones(5))
    noises = [0, 0.01 , 0.1, 1]
    ks = [0.005, 0.05, 0.5, 1, 1.5]
    for i, n in enumerate(noises):
        fig, axes = plt.subplots(2, 3, figsize = (12,7), layout='constrained', dpi=200)
        for index, ax in enumerate(axes.flat):
            if index == 0:
                deg, _ = _degradation(img, kernel, noise_level=noises[0])
                imoutput = ax.imshow(abs(np.fft.ifft2(deg)), cmap='gray', aspect='auto')
                fig.colorbar(imoutput, ax=ax)
                ax.set_title('Degraded Image with \n kernel = np.diag(np.ones(5))')
            else:
                res = _wiener_filter(img, kernel, K=ks[index - 1] , noise_level=n)
                imoutput = ax.imshow(abs(np.fft.ifft2(res)), cmap='gray', aspect='auto')
                ax.set_title('Recovered with \n noise_level = {}, K = {}'.format(n,ks[index - 1]))
                fig.colorbar(imoutput, ax=ax)
        fig.savefig(f'task5-3-{i}.png', bbox_inches='tight')

# %%
if __name__== "__main__":
    # task2_2()
    # task3_1()
    # task4_1()
    # task4_3()
    task5_1()
    # task5_2()
    # task5_3()


