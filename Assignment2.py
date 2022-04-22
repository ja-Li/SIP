# %%
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.io import imread
from skimage import exposure
from scipy import fft
from scipy import signal
import timeit

# page 14 https://pdfs.semanticscholar.org/7b31/68425864dcef39a88dce22962d9487706eaa.pdf

def task2_1():
  img = imread('./Images/trui.png',as_gray=True)
  fft_img = fft.fft2(img)
  fft_shift_img = fft.fftshift(fft_img)

  grid = plt.GridSpec(nrows=1, ncols=3, wspace=0.2, hspace=0.2)
  plt.figure(figsize = (21, 5))  
  
  plt.subplot(grid[0, 0])
  plt.imshow(img, cmap="gray", vmax=255, vmin=0)
  plt.colorbar()
  plt.title("Source")

  plt.subplot(grid[0, 1])
  plt.imshow(10*np.log10(abs(fft_img)**2) , cmap="jet")
  # plt.imshow(10*np.log10(abs(fft_img)**2) / np.max(np.max(np.log(abs(fft_img)))), cmap="jet")
  plt.colorbar()
  plt.title('10*log10(abs(fft2)**2)')

  

  plt.subplot(grid[0, 2])
  # plt.imshow(np.log(abs(fft_shift_img)) / np.max(np.max(np.log(abs(fft_shift_img)))), cmap="jet")
  plt.imshow(10*np.log10(abs(fft_shift_img)**2), cmap="jet")
  plt.colorbar()
  plt.title('10*log10(abs(fft_shift_img)**2)')
  plt.savefig('task2-1.png',bbox_inches='tight')

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

def convolution_fft(image, kernel):
    fft_image = fft.fft2(image)
    fft_kernel = fft.fft2(kernel, image.shape)
    fft_new_image = fft_image * fft_kernel
    new_image = fft.ifft2(fft_new_image, image.shape)
    return new_image.real

np.random.seed(5)
def task2_2():
    img = imread('./Images/trui.png',as_gray=True)
    time_nf = []
    time_fft=[]
    grid = plt.GridSpec(nrows=5, ncols=3, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (21, 20)) 
    for n in np.arange(1,5):
        kernel = np.random.randint(0,5,(n**3,n**3))
    #   kernel =  [(1,2,2,1),(2,0,1,2),(1,2,2,1),(3,0,2,1)] 
        kernel = np.asarray(kernel)  
        start1 = timeit.default_timer()
        new_image1 = conv2d(img, kernel)
        time_nf.append(timeit.default_timer()-start1)
        start2 = timeit.default_timer()
        new_image2 = convolution_fft(img, kernel)
        time_fft.append(timeit.default_timer()-start2)
        
        plt.subplot(grid[n-1, 0])
        plt.imshow(img, cmap="gray", vmax=255, vmin=0)
        plt.colorbar()
        plt.title("Source")

        plt.subplot(grid[n-1, 1])
        plt.imshow(new_image1, cmap="jet")
        # plt.imshow(10*np.log10(abs(fft_img)**2) / np.max(np.max(np.log(abs(fft_img)))), cmap="jet")
        plt.colorbar()
        plt.title('nested for loop with kernel size=({},{})'.format(n**3,n**3))

        plt.subplot(grid[n-1, 2])
        # plt.imshow(np.log(abs(fft_shift_img)) / np.max(np.max(np.log(abs(fft_shift_img)))), cmap="jet")
        plt.imshow(new_image2, cmap="jet")
        plt.colorbar()
        plt.title('fft with kernel size=({},{})'.format(n**3,n**3))

    plt.subplot(grid[4,:])
    plt.scatter(np.arange(1,5),time_nf, label="nest for loop")
    plt.scatter(np.arange(1,5),time_fft, label="fft")
    plt.xlabel("kernel size")
    plt.ylabel("time / s")
    plt.title("time for different size of kernel")
    plt.legend()
    plt.savefig('task2-2.png',bbox_inches='tight')
    # plt.show()

def task2_2_2():
    name =['autumn.tif','bowl_fruit.png','carpark.png','trui.png','football.jpg']
    grid = plt.GridSpec(nrows=6, ncols=3, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (21, 24))
    kernel =  [(1,2,1),(2,0,2),(1,2,1)] 
    kernel = np.asarray(kernel)
    time_nf = []
    time_fft=[]
    for n, pic in enumerate(name):
        img = imread('./Images/{}'.format(pic), as_gray=True)

        # for n in np.arange(1,5):
        start1 = timeit.default_timer()
        new_image1 = conv2d(img, kernel)
        time_nf.append(timeit.default_timer()-start1)
        start2 = timeit.default_timer()
        new_image2 = convolution_fft(img, kernel)
        time_fft.append(timeit.default_timer()-start2)
        
        plt.subplot(grid[n, 0])
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.title("Source")

        plt.subplot(grid[n, 1])
        plt.imshow(new_image1, cmap="jet")
        # plt.imshow(10*np.log10(abs(fft_img)**2) / np.max(np.max(np.log(abs(fft_img)))), cmap="jet")
        plt.colorbar()
        plt.title('nested for loop')

        plt.subplot(grid[n, 2])
        # plt.imshow(np.log(abs(fft_shift_img)) / np.max(np.max(np.log(abs(fft_shift_img)))), cmap="jet")
        plt.imshow(new_image2, cmap="jet")
        plt.colorbar()
        plt.title('fft')

    plt.subplot(grid[5,:])
    plt.scatter(np.arange(0,5),time_nf, label="nest for loop")
    plt.scatter(np.arange(0,5),time_fft, label="fft")
    plt.xlabel("img name:{}".format(name))
    plt.ylabel("time / s")
    plt.title("time for different size of img")
    plt.legend()
    plt.savefig('task2-2-2.png',bbox_inches='tight')

def calculate_function_map(img, v, w, alpha):
  func_map = np.zeros_like(img)
  for i in range (func_map.shape[0]):
    for j in range (func_map.shape[1]):
      func_map[i][j] = alpha * math.cos(v * i + w*j)
  return func_map

def calculate_fourier_array(img, v0, w0):
  fourier_array = np.zeros_like(img, dtype=float)
  for i in range (fourier_array.shape[0]):
    for j in range (fourier_array.shape[1]):
      fourier_array[i][j] = math.cos(v0*i + w0*j)
  return np.abs(fft.fft2(fourier_array))

def filter(img, v, w, threshold=100):
  fourier_array = calculate_fourier_array(img, v, w)
  img_fft = fft.fft2(img.copy())
  img_fft[fourier_array > threshold] = 0
  img_filtered = fft.ifft2(img_fft)
  return np.abs(img_filtered)

def task3_1():
  v = 0.1
  w = 0.2
  alpha = 15.0
  img_orignal = imread('./Images/cameraman.tif',as_gray=True)
  img_after_noise = img_orignal.copy() + calculate_function_map(img_orignal, v, w, alpha)
  img_after_noise_fft = fft.fft2(img_after_noise.copy())
  img_power_spectrum = fft.fftshift(img_after_noise_fft.copy())

  img_filtered = filter(img_after_noise, v, w)
  grid = plt.GridSpec(nrows=1, ncols=3, wspace=0.2, hspace=0.2)
  plt.figure(figsize = (21, 5))  

  plt.subplot(grid[0, 0])
  plt.imshow(img_after_noise,cmap="gray")
  plt.colorbar()
  plt.title('Add function Value')
  
  plt.subplot(grid[0, 1])
  plt.imshow(10*np.log10(abs(img_power_spectrum)**2), cmap="gray")
  plt.colorbar()
  plt.title("Power Spectrum")

  plt.subplot(grid[0, 2])
  plt.imshow(img_filtered, cmap="gray")
  plt.colorbar()
  plt.title('FTT2 Filter')
  plt.savefig("task3-1.png")

if __name__== "__main__":
  task2_1()
  task2_2()
  task2_2_2()
  task3_1()





