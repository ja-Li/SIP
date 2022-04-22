from cmath import exp, pi
from email.mime import image
import cv2
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, util
from skimage.morphology import square
from skimage.filters import gaussian, rank
from scipy.signal import medfilt2d

# Implementation the gamma transform of a gray, rgb or hsv image
def gamma_transform(img, gamma = 0.25, mode='gray'):
  img = np.array(img)
  if mode == 'gray':
    assert len(img.shape) == 2, 'The image should be (H,W)'
    img = ((img / 255) ** gamma) * 255 
  elif mode == 'rgb':
    assert len(img.shape) == 3, 'The image should be (H,W,3)'
    img[:, :, 0] = ((img[:, :, 0] / 255) ** gamma) * 255
    img[:, :, 1] = ((img[:, :, 1] / 255) ** gamma) * 255
    img[:, :, 2] = ((img[:, :, 2] / 255) ** gamma) * 255
  elif mode == 'hsv':
    # using skimage to deal with hsv, because in the skimage,
    # the range of v is [0,1]. so we need not to divide 100
    assert len(img.shape) == 3, 'The image should be (H,W,3)'
    # img[:, :, 2] = ((img[:, :, 2] / 100)**gamma) * 100
    img[:, :, 2] = ((img[:, :, 2])**gamma)
    return img, hsv2rgb(img)
  else:
    print('Do not support this mode: ', mode)
  return Image.fromarray(img.astype('uint8'))

# task 1.1, we load a image and convert it to grayscale image
# then we apply the gamma_transform function to this image
def transform_gray_image():
  img = Image.open('./images/basic_image.jpg').convert('L')
  # print(img.shape)
  gamma = [1, 0.25, 0.5, 0.75, 2, 3]
  fig, axs = plt.subplots(2, 3, figsize=(15,8))
  for i in range(2):
    for j in range(3):
      if i == 0 and j==0:
        axs[i, j].imshow(img, cmap="gray")
        axs[i, j].set_title('original grayscale image')
      else:
        axs[i, j].imshow(gamma_transform(img ,gamma[i*3 + j], 'gray'), cmap="gray")
        axs[i, j].set_title(f'gamma={gamma[i*3 + j]}')
  fig.suptitle('Gamma transform of a gray scale image')
  fig.show()
  plt.savefig('task1-1.png')

# task 2.1, we load a rgb image then we apply the 
# gamma_transform function with rgb mode to this image
def transform_RGB_image():
  img = Image.open('./images/autumn.tif')
  gamma = [1, 0.3, 0.6, 1.5 , 2, 3]
  fig, axs = plt.subplots(2, 3, figsize=(15,8))
  for i in range(2):
    for j in range(3):
      if i == 0 and j==0:
        axs[i, j].imshow(img)
        axs[i, j].set_title('original image')
      else:
        axs[i, j].imshow(gamma_transform(img ,gamma[i*3 + j], 'rgb'))
        axs[i, j].set_title(f'gamma={gamma[i*3 + j]}')
  fig.suptitle('gammm correction RGB')
  fig.show()
  plt.savefig('task1-2.png')


# task 2.1, we load a rgb image then convert it to hsv, 
# Then using gamma_transform function with hsv mode to get the 
# hsv image and the corresponding rgb image.
def transform_HSV_image():
  img = imread('./images/autumn.tif')
  img_hsv_origin = rgb2hsv(img)
  gamma = [0, 0.3, 0.6, 2.0, 3.0]
  fig, axs = plt.subplots(5, 2, figsize=(12,16))
  plt.hsv()
  for i in range(len(gamma)):
    if i == 0:
      axs[i, 0].imshow(img)
      axs[i, 0].set_title('original rgb image')
      axs[i, 1].imshow(img_hsv_origin)
      axs[i, 1].set_title('original hsv image')
    else:
      img_hsv, img_rgb = gamma_transform(img_hsv_origin, gamma[i], 'hsv')
      axs[i, 0].imshow(img_rgb)
      axs[i, 0].set_title(f'gamma={gamma[i]}, rgb')
      axs[i, 1].imshow(img_hsv)
      axs[i, 1].set_title(f'gamma={gamma[i]}, hsv')
  fig.suptitle('gammm correction HSV, and the related rgb image')
  fig.show()
  plt.savefig('task1-3.png', bbox_inches='tight')

def cumulative_histogram(histogram):
    return np.cumsum(histogram) / np.sum(histogram)

def task2_1():
    img = imread('./images/pout.tif', 0)
    hist, _ = np.histogram(img.ravel(),256,[0,256])
    cumhist = cumulative_histogram(hist)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].hist(img.ravel(),256,[0,256])
    axs[0].set_title("Histogram for 'pout.tif'")
    axs[0].set_xlabel('Intensity of pixel')
    axs[0].set_ylabel('Times')

    axs[1].plot(cumhist)
    axs[1].set_title("CDF for 'pout.tif'")
    axs[1].set_xlabel('Intensity of pixel')
    axs[1].set_ylabel('Cumulative intensity')

    fig.suptitle('Task 2.1')
    fig.savefig('task2-1.png')
    fig.show()

def floating_point_img(img, cdf):
    # img = np.array(img)
    fp_img = cdf[img]
    return fp_img

def task2_2():
    img = imread('./images/pout.tif')

    hist, _ = np.histogram(img.ravel(),256,[0,256])
    cumhist = cumulative_histogram(hist)
    fp_img = floating_point_img(img, cumhist)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    i1 = axs[0].imshow(img, cmap="gray",aspect="auto",vmax=255, vmin=0)
    axs[0].set_title("Original image")
    fig.colorbar(i1, ax=axs[0])

    i2 = axs[1].imshow(fp_img, cmap="gray",aspect="auto")
    axs[1].set_title("Floating Point image")
    fig.colorbar(i2, ax=axs[1])
    
    fig.savefig('task2-2.png')
    fig.show()

# Task 2_3
def pseudo_inverse(cdf, l):
    return np.min(np.where(cdf >= l))

def histogram_matching(img1, img2):
    img1_hist, _ = np.histogram(img1.ravel(),256,[0,256])
    img2_hist, _ = np.histogram(img2.ravel(),256,[0,256])

    cumsum_img1 = cumulative_histogram(img1_hist)
    cumsum_img2 = cumulative_histogram(img2_hist)

    fp_img1 = floating_point_img(img1, cumsum_img1)
    (r,c) = fp_img1.shape
    output_img = fp_img1.copy()
    for x in range(0,r):
        for y in range(0,c):
            output_img[x,y] = pseudo_inverse(cumsum_img2, fp_img1[x,y])

    return cumsum_img1, cumsum_img2, output_img

def task2_4():
    img1 = imread('./images/pout.tif', as_gray=True)
    img2 = imread('./images/trees.tif', as_gray=True)

    cumsum_img1, cumsum_img2, result = histogram_matching(img1, img2)
    result_hist, _ = np.histogram(result.ravel(),256,[0,256])
    cumsum_result = cumulative_histogram(result_hist)

    grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.2)

    plt.figure(figsize = (15, 8))  

    plt.subplot(grid[0, 0])  
    plt.imshow(img1, cmap="gray",aspect="auto",vmax=255, vmin=0)
    plt.colorbar()
    plt.title("Image 1")

    plt.subplot(grid[0, 1])  
    plt.imshow(img2, cmap="gray",aspect="auto", vmax=255, vmin=0)
    plt.colorbar()
    plt.title("Image 2")

    plt.subplot(grid[0, 2])  
    plt.imshow(result, cmap="gray",aspect="auto", vmax=255, vmin=0)
    plt.colorbar()
    plt.title("Output image")

    plt.subplot(grid[1, :])
    plt.plot(cumsum_img1, color='b', label="Image 1")
    plt.plot(cumsum_img2, color='r', label="Image 2")
    plt.plot(cumsum_result, color='g', label="Output image")
    plt.title("CDF")
    plt.ylabel("Cumulative intensity")
    plt.xlabel("Pixel intensitiy")
    plt.legend()
    plt.savefig('task2-4.png')
    

def my_equalize(img):
    constant = np.linspace(0, 255, num=256)
    cumsum_img, cumsum_constant, result_img = histogram_matching(img, constant)
    return result_img

def task2_5():
    img = imread('./images/pout.tif')

    my_equalize_img = my_equalize(img)
    # equalized_img = exposure.equalize_hist(img) * 255
    img_eq    = exposure.equalize_hist(img)
    cdf_eq, _ = exposure.cumulative_distribution(img_eq, 256)

    img_hist, _ = np.histogram(img.ravel(),256,[0,256])
    cumsum_img = cumulative_histogram(img_hist)

    my_equalize_hist, _ = np.histogram(my_equalize_img.ravel(),256,[0,256])
    cumsum_my_equalized = cumulative_histogram(my_equalize_hist)


    grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.2)

    plt.figure(figsize = (15, 8))  

    plt.subplot(grid[0, 0])  
    plt.imshow(img, cmap="gray",aspect="auto",vmax=255, vmin=0)
    plt.colorbar()
    plt.title("Source")

    plt.subplot(grid[0, 1])  
    plt.imshow(my_equalize_img, cmap="gray",aspect="auto",vmax=255, vmin=0)
    plt.colorbar()
    plt.title("After my equalize")

    plt.subplot(grid[0, 2])  
    plt.imshow(img_eq, cmap="gray",aspect="auto")
    plt.colorbar()
    plt.title("After skimage")

    plt.subplot(grid[1, :3])
    plt.plot(cumsum_img, color='b', label="CDF for Source")
    plt.plot(cumsum_my_equalized, color='r', label="CDF for After my equalized")
    plt.plot(cdf_eq, color='g', label="CDF for After skimage")
    plt.title("CDF")
    plt.ylabel("Cumulative intensity")
    plt.xlabel("Pixel intensitiy")
    plt.legend()
    plt.savefig('task2-5.png')


def filters(img=None, mode='mean', kernel_size=3, sigma=1):
    if img is None:
        img = imread('./images/eight.tif')
    k = kernel_size
    if mode == 'mean':
        res_img = rank.mean(image=img, footprint=square(k))
    elif mode == 'median':
        res_img = medfilt2d(input=img, kernel_size=k)
    elif mode == 'gaussian':
        res_img = gaussian(img, sigma)
    
    return res_img


# effect of increasing kernel size
def task3_1():
    
    img = imread('./images/eight.tif')
    sp_noise = util.random_noise(img, mode="s&p")
    gaussian_noise = util.random_noise(img, mode="gaussian")

    images = [img,sp_noise,gaussian_noise]
    images_name = ['original','sp_noise','gaussian_noise']
    type = ['mean','median']
    grid = plt.GridSpec(nrows=9, ncols=3)
    plt.figure(figsize = (15, 33)) 
    kernel = [0,3,5,13,23]
    # row = 0
    for row, k in enumerate(kernel):
        if k == 0:
            for index, image in enumerate(images):
                    plt.subplot(grid[0,index])
                    plt.imshow(image, cmap="gray",aspect="auto")
                    plt.colorbar()
                    plt.title(images_name[index])
        else:
            for index, image in enumerate(images):
                for t in type:
                    if t == 'mean':
                        plt.subplot(grid[row,index])
                        plt.imshow(filters(image, mode=t, kernel_size=k), cmap="gray",aspect="auto")
                        plt.colorbar()
                        plt.title(images_name[index] + " with "+ t +" filter and k ="+ str(k))
                    else:
                        plt.subplot(grid[row+4,index])
                        plt.imshow(filters(image, mode=t, kernel_size=k), cmap="gray",aspect="auto")
                        plt.colorbar()
                        plt.title(images_name[index] + " with "+ t +" filter and k ="+ str(k))
    # plt.suptitle('Mean& Median filters with increasing k for 3 type images')
    plt.savefig("3-1-1.png", bbox_inches='tight')

# computional time
def task3_1_2():
    import timeit
    import_module="""
from skimage.io import imread
from skimage import exposure, util
from skimage.morphology import square
from skimage.filters import gaussian, rank
from scipy.signal import medfilt2d

original = imread("eight.tif")
sp_noise = util.random_noise(original, mode="s&p")
gaussian_noise = util.random_noise(original, mode="gaussian")
"""

    execution_time_mean = []
    execution_time_median=[]
    # grid1 = plt.GridSpec(nrows=3, ncols=1, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (15, 15)) 
    colors = ['r','g','b','c','m','y']
    image = ['original','sp_noise','gaussian_noise']
    for index, i in enumerate(image):
        for k in np.arange(1,26,2):        
            execution_time_median.append(timeit.timeit(stmt='medfilt2d('+i+', kernel_size='+str(k)+')',setup=import_module, number=100))
            execution_time_mean.append(timeit.timeit(stmt='rank.mean(image='+i+', footprint=square('+str(k)+'))',setup=import_module, number=100))

        plt.scatter(np.arange(1,26,2),execution_time_mean, color=colors[index], label=i+" mean filter")
        plt.scatter(np.arange(1,26,2),execution_time_median, color=colors[index+3], label=i+" median filter")
        # plt.plot(cumsum_skimage_equalized, color='g', label="CDF for After skimage")
        plt.title("computional times for 3 types images")
        plt.ylabel("time for 100 executions (unit s)")
        plt.xlabel("kernel size")
        plt.legend()

        execution_time_mean = []
        execution_time_median = []

    plt.savefig('Computational_time.png')



def task3_2():
    img = imread('./images/eight.tif')
    # sp_noise = util.random_noise(img, mode="s&p")
    gaussian_noise = util.random_noise(img, mode="gaussian")

    grid = plt.GridSpec(nrows=1, ncols=5,wspace=0.2, hspace=0.2)
    plt.figure(figsize = (21, 4)) 
    kernel = [0,3,5,13,23]
    # row = 0
    for col, k in enumerate(kernel):
        if k == 0:
            plt.subplot(grid[0,col])
            plt.imshow(img, cmap="gray",aspect="auto")
            plt.colorbar()
            plt.title('After Gaussian noise')
        else:
            plt.subplot(grid[0,col])
            plt.imshow(filters(gaussian_noise, mode='gaussian', kernel_size=k), cmap="gray",aspect="auto")
            plt.colorbar()
            plt.title("kernel size = {k}".format(s = 5, k=str(k)))
    plt.suptitle('Gassian filter with sigma=5 and increasing k')
    plt.savefig("3-2.png")
    
# task3_2()


def task3_3():
    img = imread('./images/eight.tif')
    # sp_noise = util.random_noise(img, mode="s&p")
    gaussian_noise = util.random_noise(img, mode="gaussian")

    kernel = [0,1,2,5,9,20]
    grid = plt.GridSpec(nrows=1, ncols=6,wspace=0.2, hspace=0.2)
    plt.figure(figsize = (30, 4)) 
    for col, k in enumerate(kernel):
        if k == 0:
            # for index, image in enumerate(images):
            plt.subplot(grid[0,col])
            plt.imshow(gaussian_noise, cmap="gray",aspect="auto")
            plt.colorbar()
            plt.title('After Gaussian noise')
        else:
            plt.subplot(grid[0,col])
            plt.imshow(filters(gaussian_noise, mode='gaussian', kernel_size=k*k, sigma=round(k/3,2)), cmap="gray",aspect="auto")
            plt.colorbar()
            plt.title("sigma = {s}, kernel size = {k}".format(s = round(k/3,2), k=str(k*k)))
    plt.suptitle('Gassian filter with increasing sigma and k')
    plt.savefig("3-3.png")
    

def gauss_noise(img):
  img = np.array(img) / 255
  mean = 0
  stddev = 0.09
  img = img + np.random.normal(mean, stddev, (img.shape[0], img.shape[1]))
  img = np.clip(img*255, 0, 255)
  return Image.fromarray(img.astype('uint8'))

def kernel(mode='gauss', kernel_size=3, sigma=2):
  k = kernel_size
  kernel = np.ones([k,k])
  if mode == 'gauss':
      xs = np.linspace(-(k-1)/2., (k-1)/2., k)
      x,y = np.meshgrid(xs, xs)
      kernel = np.exp(-(x**2 + y**2) / (2*sigma**2))
  return kernel

def conv2d(img, kernel, tau = 1.0, stride=1, padding=True):
  if padding:
    img = np.pad(img, pad_width=int((kernel.shape[0]-1)/2))
  H = img.shape[0]
  W = img.shape[1]
  New_H = int(1 + (H - kernel.shape[0])/ stride)
  New_W = int(1 + (W - kernel.shape[1])/ stride)
  img_res = np.ones((New_H, New_W))
  for i in range(New_H):
    for j in range(New_W):
      cur_windows = img[i*stride:(i * stride + kernel.shape[0]), j*stride:(j * stride + kernel.shape[1])]
      center_pixel = cur_windows[int((kernel.shape[0]-1)/2), int((kernel.shape[0]-1)/2)]
      cur_windows_reduce_center = cur_windows - center_pixel
      weight = np.exp(-(cur_windows_reduce_center**2/2*tau**2)) * kernel
      img_res[i, j] = (weight * cur_windows).sum() / weight.sum()
  return img_res


def filter(img, mode='gauss', kernel_size=3, sigma=2, tau=1.0):
  if mode == 'gauss':
      gauss_kernel = kernel(mode, kernel_size=kernel_size,sigma=sigma)
      img = conv2d(img, gauss_kernel, tau=tau)
  return Image.fromarray(img.astype('uint8'))

def filter_img():
  img = Image.open('./images/eight.tif').convert('L')
  img = gauss_noise(img)
  tau  = [0.01, 0.1, 1, 10, 100]
  sigma= [0.01, 0.1, 1, 10]
  fig, axs = plt.subplots(4, 5, figsize=(15,10))
  for i in range(len(sigma)):
    for j in range(len(tau)):
      if i == 0 and j==0:
        axs[i,j].imshow(img, cmap='gray')
        axs[i,j].set_title('origin image')
      else:
        axs[i,j].imshow(filter(img=img, kernel_size=9, sigma=sigma[i], tau=tau[j]), cmap='gray')
        axs[i,j].set_title(r'$\sigma=$'+str(sigma[i])+', $\\tau$='+str(tau[j]))
  fig.suptitle('Gauss')
  fig.show()
  plt.savefig('task4-3.png', bbox_inches='tight')

if __name__== "__main__":
  #task 1.1-1.3
  transform_gray_image()
  transform_RGB_image()
  transform_HSV_image()
  task2_1()
  task2_2()
  task2_4()
  task2_5()
  task3_1()
  task3_1_2()
  task3_2()
  task3_3()
  #task 4.2
  filter_img()
