# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.filters import sobel,prewitt
from skimage import util
from skimage.morphology import binary_dilation
import timeit

# %%
def task1_3():
    seed = 5
    img = imread("./Week 4/eight.tif", as_gray=True)
    variance = [0.01,0.05]
    gaussian_noise_1 = util.random_noise(img, mode="gaussian", seed=seed, var=variance[0])
    gaussian_noise_2 = util.random_noise(img, mode="gaussian", seed=seed, var=variance[1])

    grid = plt.GridSpec(nrows=2, ncols=8, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (40, 9), dpi=150)

    gaussian_noise = [gaussian_noise_1, gaussian_noise_2]
    for index, noise_img in enumerate(gaussian_noise):
        i = 0
        res_sobely = sobel(gaussian_noise[index], axis=0)
        res_sobelx = sobel(gaussian_noise[index], axis=1)
        sobel_required = res_sobelx**2 + res_sobely**2

        res_prewittx = prewitt(gaussian_noise[index], axis=1)
        res_prewitty = prewitt(gaussian_noise[index], axis=0)
        prewitt_required = res_prewittx**2 + res_prewitty**2

        plt.subplot(grid[index, i])
        plt.imshow(img, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("eight.tif with variance {}".format(variance[index])) 

        plt.subplot(grid[index, i+1])
        plt.imshow(noise_img, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("eight.tif with variance {}".format(variance[index]))

        plt.subplot(grid[index, i+2])
        plt.imshow(res_sobelx, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("sobel filter with \n x derivative and variance {}".format(variance[index]))

        plt.subplot(grid[index, i+3])
        plt.imshow(res_sobely, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("sobel filter with \n y derivative and variance {}".format(variance[index]))        

        plt.subplot(grid[index, i+4])
        plt.imshow(sobel_required, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("sobel filter with squared \n gradient magnitude and variance {}".format(variance[index]))

        plt.subplot(grid[index, i+5])
        plt.imshow(res_prewittx, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("prewitty filter with \n x derivative and variance {}".format(variance[index]))

        plt.subplot(grid[index, i+6])
        plt.imshow(res_prewitty, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("prewitty filter with \n y derivative and variance {}".format(variance[index]))  

        plt.subplot(grid[index, i+7])
        plt.imshow(prewitt_required, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("prewitt filter with squared \n gradient magnitude and variance {}".format(variance[index]))

    plt.savefig("task1-3.png",bbox_inches='tight')

task1_3()

# %%
# CDF
def cdf(histogram):
    return np.cumsum(histogram) / np.sum(histogram)

def floating_point_img(img, res_cdf):
    # img = np.array(img)
    fp_img = res_cdf[img]
    return fp_img

# inverse function
def pseudo_inverse(res_cdf, l):
    return np.min(np.where(res_cdf >= l))

# midway specifications functions
def msf(img1_hist, img2_hist, target_img):
    target_hist, _ = np.histogram(target_img.ravel(),256,[0,256])
    cumsum_img1 = cdf(img1_hist)
    cumsum_img2 = cdf(img2_hist)
    target_cdf = cdf(target_hist)

    # fp_target = floating_point_img(target_img, target_cdf)
    
    # (r,c) = fp_target.shape
    # msf_img = fp_target.copy()
    # for x in range(0,r):
    #     for y in range(0,c):
    #         msf_img[x,y] = (pseudo_inverse(cumsum_img1, fp_target[x,y]) + pseudo_inverse(cumsum_img2, fp_target[x,y])) / 2

    # floating_point_img(target_img, cumsum_img1)
    cdf_img1_inverse = img1_hist.copy()
    for i, x in enumerate(target_cdf):
        cdf_img1_inverse[i] = pseudo_inverse(cumsum_img1, x)
    
    cdf_img2_inverse = img2_hist.copy()
    for i, x in enumerate(target_cdf):
        cdf_img2_inverse[i] = pseudo_inverse(cumsum_img2, x)
    
    # print(cdf_img1_inverse)

    ms_value = (cdf_img1_inverse + cdf_img2_inverse)/2 
    msf_result = floating_point_img(target_img, ms_value)
    # msf_result = (floating_point_img(target_img, cdf_img1_inverse) + floating_point_img(target_img, cdf_img2_inverse))/2
    return msf_result
    # return msf_img


def task2_2():
    img1 = (imread("./Week 4/movie_flicker/movie_flicker1.tif", as_gray=True) * 255).astype(int)
    img2 = (imread("./Week 4/movie_flicker/movie_flicker2.tif", as_gray=True) * 255).astype(int)

    # img1 = imread("./Week 4/movie_flicker/movie_flicker1.tif")
    # img2 = imread("./Week 4/movie_flicker/movie_flicker2.tif")

    img1_hist, _ = np.histogram(img1.ravel(),256,[0,256])
    img2_hist, _ = np.histogram(img2.ravel(),256,[0,256])
    
    average = (cdf(img1_hist) + cdf(img2_hist)) /2

    grid = plt.GridSpec(nrows=3, ncols=2)
    plt.figure(figsize = (15, 10), dpi=300)

    targets = [img1, img2]
    colors = ['r','g','b','m']
    names = ['flicker1', 'flicker2']
    # img1_cdf = cdf(img1_hist)
    # floating_point_img(targets[0],img1_cdf)

    for index, target in enumerate(targets):
        col = 0
        msf_res = msf(img1_hist, img2_hist, target)
        msf_hist,_ = np.histogram(msf_res.ravel(),256,[0,256])
        msf_cdf = cdf(msf_hist)

        target_hist,_ = np.histogram(target.ravel(),256,[0,256])
        target_cdf = cdf(target_hist)

        plt.subplot(grid[index, col])
        plt.imshow(target, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("Origin")

        plt.subplot(grid[index, col+1])
        plt.imshow(msf_res, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title("Midway specification")

        plt.subplot(grid[2,:])
        plt.plot(msf_cdf, label="Midway specifications for {}".format(names[index]))
        plt.plot(target_cdf, label="{}".format(names[index]))
    plt.plot(average,label="average")
    plt.title("CDF")
    plt.ylabel("Cumulative intensity")
    plt.xlabel("Pixel Range")
    plt.legend()

    plt.savefig("task2-2.png",bbox_inches='tight')
    plt.show()

task2_2()

# %%
def gaussian_kernel(size=3, sigma=1):
    """Returns a 2D Gaussian kernel.
    Parameters
    ----------
    size : float, the kernel size (will be square)

    sigma : float, the sigma Gaussian parameter in frequency

    Returns
    -------
    out : array, shape = (size, size)
      an array with the centered gaussian kernel
    """
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()

def scale_fft(img, sigma):
    # gaussian kernel and compute the 2d discrete Fourier Transform, shift
    img_gaussian = gaussian_kernel(img.shape[0], sigma)
    # compute the 2-dimensional discrete Fourier Transform.
    # shift the zero-frequency component to the center of the spectrum
    img_ft = np.fft.fftshift(np.fft.fft2(img))
    # Compute the 2-dimensional inverse discrete Fourier Transform.
    img_ift = np.fft.ifft2(np.fft.ifftshift(img_ft*img_gaussian))
    return abs(img_ift)

def task3_1():
    # img = (imread("./Week 4/trui.png", as_gray=True) * 255).astype(int)
    img = imread("./Week 4/trui.png", as_gray=True)
    sigmas = np.linspace(1,4,5)
    # print(sigmas)
    grid = plt.GridSpec(nrows=1, ncols=6)
    plt.figure(figsize = (23, 4), dpi=300)
    
    plt.subplot(grid[0, 0])
    plt.imshow(img, cmap="gray",aspect="auto")
    plt.colorbar()
    plt.title("Origin")
    
    for index, sigma in enumerate(sigmas):
        res_img = scale_fft(img,sigma)

        plt.subplot(grid[0, index+1])
        plt.imshow(res_img, cmap="gray",aspect="auto")
        plt.colorbar()
        plt.title(r"$\sigma$ = {}".format(sigma))
    plt.savefig("task3-1.png",bbox_inches='tight')
    plt.show()

task3_1()

# %%
def conv_fft(image, kernel):
    fft_image = np.fft.fft2(image)
    fft_kernel = np.fft.fft2(kernel, image.shape)
    # print(fft_kernel.shape)
    fft_new_image = fft_image * fft_kernel
    new_image = np.fft.ifft2(fft_new_image, image.shape)
    # print(new_image[20,20].real)
    return new_image.real

def derivative2d(image, dx, dy):
    kernel_x = [[1, -1]]
    kernel_y = [[1], [-1]]
    derivative = image.copy()
    if dx !=0 :
        for i in range(dx):
            derivative = conv_fft(derivative, kernel_x)
    if dy!=0 :
        for i in range(dy):
            derivative = conv_fft(derivative, kernel_y)
    return derivative

def task3_2():
    img = imread("./Week 4/trui.png", as_gray=True)
    derivative_num = 3

    grid = plt.GridSpec(nrows=3, ncols=3)
    plt.figure(figsize = (18, 15), dpi=300)

    for dx in range(derivative_num):
        for dy in range(derivative_num):
            res = derivative2d(img, dx, dy)
            if dx == 0 and dy == 0:
                plt.subplot(grid[dx, dy])
                plt.imshow(img, cmap="gray",aspect="auto")
                plt.colorbar()
                plt.title("Origin")
            else:
                plt.subplot(grid[dx, dy])
                plt.imshow(res, cmap="gray",aspect="auto")
                plt.colorbar()
                plt.title("x-orders = {}\ny-orders = {}".format(dx,dy))
    
    plt.savefig("task3-2.png",bbox_inches='tight')
    plt.show()

task3_2()

# %%
def task4_1():
    # digit A generated by https://tools.withcode.uk/binaryimage/
    digit_A = [[1, 1, 1, 0, 1, 1, 1],
		[1, 1, 0, 1, 0, 1, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 0, 0, 0, 0, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 1, 1, 1, 0, 1]]

    digit_A = np.array(digit_A) - 1
    digit_A = np.int64(digit_A<0)
   
    mask_t = np.array([[1,1,1],[0,1,0],[0,1,0]]) 
    mask_mosaic = np.array([[0,1],[1,0]])
    mask_1D = np.array([[0,0,1]])

    mask_1 = np.array([[1,1,1],[0,0.5,0],[0,1,0]]) 
    mask_2 = np.array([[0,1],[1,0.5]])
    mask_3 = np.array([[0,0.5,1]])

    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    #axs[0].axis('off')
    axs[0].imshow(mask_1, cmap="gray", aspect='auto')
    axs[0].set_title("Mask 1")
    #axs[1].axis('off')
    axs[1].imshow(mask_2, cmap="gray", aspect='auto')
    axs[1].set_title("Mask 2")
    #axs[2].axis('off')
    axs[2].imshow(mask_3, cmap="gray", aspect='auto')
    axs[2].set_title("Mask 3")
    plt.savefig("centre pixel.png")
    fig.show()

    digitA_t_dilation = binary_dilation(digit_A, mask_t)
    digitA_tile_dilation = binary_dilation(digit_A, mask_mosaic)
    digitA_1D_dilation = binary_dilation(digit_A.reshape(1, digit_A.size), selem=mask_1D.reshape(1,3)).reshape(7,7)

    grid = plt.GridSpec(nrows=2, ncols=4)
    plt.figure(figsize = (8,4), dpi=300)

    plt.subplot(grid[1,0])
    plt.imshow(digit_A, cmap='gray')
    plt.title('Original image')
    plt.subplot(grid[1,1])
    plt.imshow(digitA_t_dilation, cmap='gray')
    plt.title('Mask 1')
    plt.subplot(grid[1,2])
    plt.imshow(digitA_tile_dilation, cmap='gray')
    plt.title('Mask 2')
    plt.subplot(grid[1,3])
    plt.imshow(digitA_1D_dilation, cmap='gray')
    plt.title('Mask 3')

    plt.savefig("task4-1.png", bbox_inches="tight")
task4_1()

# %%
def add_padding(img,x):
    i, j = img.shape
    # print(i,j)
    if i != j:
        img_pad = np.zeros((7,7))
        img_pad[(7-i)//2:i+(7-i)//2,(7-j)//2:j+(7-j)//2] = img[:]
    # elif i % 2 ==0 and j % 2 ==0:
    #     img_pad = np.zeros((i+2*x-1,j+2*x-1))
    #     img_pad[x-1:i+x-1,x-1:j+x-1] = img[:]        
    else:   
        img_pad = np.zeros((i+2*x,j+2*x))
        img_pad[x:i+x,x:j+x] = img[:]
    return img_pad

def task4_2():
    # digit A generated by https://tools.withcode.uk/binaryimage/
    digit_A =  [[1, 1, 1, 0, 1, 1, 1],
		[1, 1, 0, 1, 0, 1, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 0, 0, 0, 0, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 1, 1, 1, 0, 1],
		[1, 0, 1, 1, 1, 0, 1]]

    digit_A = np.array(digit_A) - 1
    digit_A = np.int64(digit_A<0)
   
    mask_t = np.array([[1,1,1],[0,1,0],[0,1,0]]) 
    mask_mosaic = np.array([[0,1],[1,0]])
    mask_1D = np.array([[0,0,1]])
    # print(mask_1D.shape)

    mask_t_pad = add_padding(mask_t, 2)
    mask_mosaic_pad = add_padding(mask_mosaic, 3)
    mask_1D_pad = add_padding(mask_1D, 4)
    
    digitA_t_dilation = binary_dilation(mask_t_pad, digit_A)
    digitA_tile_dilation = binary_dilation(mask_mosaic_pad, digit_A)
    digitA_1D_dilation = binary_dilation(mask_1D_pad, digit_A)

    grid = plt.GridSpec(nrows=2, ncols=4)
    plt.figure(figsize = (8,4), dpi=300)

    plt.subplot(grid[1,0])
    plt.imshow(digit_A, cmap='gray')
    plt.title('Original image')
    plt.subplot(grid[1,1])
    plt.imshow(digitA_t_dilation, cmap='gray')
    plt.title('Mask 1')
    plt.subplot(grid[1,2])
    plt.imshow(digitA_tile_dilation, cmap='gray')
    plt.title('Mask 2')
    plt.subplot(grid[1,3])
    plt.imshow(digitA_1D_dilation, cmap='gray')
    plt.title('Mask 3')

    plt.savefig("task4-2.png", bbox_inches="tight")
    
task4_2()


