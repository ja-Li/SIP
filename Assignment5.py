from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage import feature
from scipy import signal, ndimage

def task1_1():
    img = imread('Week 6/hand.tiff', as_gray=True)

    plt.figure(dpi=150)
    plt.imshow(img, cmap="gray",aspect="auto")
    plt.colorbar()
    plt.title("origin")
    # plt.show()
    plt.savefig("task1-1-o.png", bbox_inches='tight')

    rows = 2
    cols = 4
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (20, 10), dpi=150)

    sigma = [1.,2.]
    low = [None, 50.,None, 50.]
    high = [None, None, 150., 150.]

    for row in range(rows):
        for col in range(cols):
            plt.subplot(grid[row, col])
            # print(sig, low_v, high_v)
            plt.imshow(feature.canny(img, sigma=sigma[row], low_threshold=low[col], high_threshold=high[col]), cmap="gray",aspect="auto")
            plt.colorbar()
            plt.title("sigma={}\n low_threshold={}, high_threshold={}".format(sigma[row], low[col], high[col]))
    plt.savefig("task1-1.png", bbox_inches='tight')

def task1_2():
    img = imread('Week 6/modelhouses.png', as_gray=True)

    plt.figure(dpi=150)
    plt.imshow(img, cmap="gray",aspect="auto")
    plt.colorbar()
    plt.title("origin")
    # plt.show()
    plt.savefig("task1-2-o.png", bbox_inches='tight')

    methods = ['k', 'eps']
    ks = [0.01, 0.1, 1, 2]
    epsilons = [1e-06, 1e-50, 0.01, 1]

    sigmas = [1.0, 3.0, 5.]

    rows = 4
    cols = 4
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (20, 20), dpi=200)

    for row in range(rows):
        for j, k in enumerate(ks):
            plt.subplot(grid[row, j])
            # print(sig, low_v, high_v)
            if row < 3:
                method = methods[0]
                res = feature.corner_harris(img, method=method, k=k, sigma=sigmas[row])
                plt.title("sigma={}\n method={}, k={}".format(sigmas[row], method, k))
            else:
                method = methods[1]
                res = feature.corner_harris(img, method=method, eps=epsilons[j], sigma=sigmas[0])
                plt.title("sigma={}\n method={}, eps={}".format(sigmas[0], method, epsilons[j]))
            plt.imshow(res, cmap="gray",aspect="auto")
            plt.colorbar()
    plt.savefig('task1-2.png',bbox_inches='tight')

def task1_3():
    img = imread('Week 6/modelhouses.png', as_gray=True)
    k = 0.2
    e = 1e-06
    sigma1 = 3
    sigma2 = 3
    harrisk_img = feature.corner_harris(img, method='k', k=k, sigma=sigma1)
    local_max_k = feature.corner_peaks(harrisk_img)


    harriseps_img = feature.corner_harris(img, method='eps', eps=e, sigma=sigma2)
    local_max_eps = feature.corner_peaks(harriseps_img)

    rows = 2
    cols = 2
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (15, 13), dpi=200)

    plt.subplot(grid[0,0])
    plt.imshow(harrisk_img, cmap="gray",aspect="auto")
    plt.title("Corner_harris transformed \n sigma={}\n method={}, k={}".format(sigma1, 'k', k))
    plt.colorbar()

    plt.subplot(grid[0,1])
    plt.imshow(img,cmap="gray",aspect="auto")
    plt.colorbar()
    plt.scatter(local_max_k[:,1],local_max_k[:,0], marker='.', color='r')
    # plt.imshow(bg_harrisk, cmap="gray",aspect="auto")
    plt.title('Highlighted local maxima in feature map (ie. "the red dots") \n sigma={}\n method={}, k={}'.format(sigma1, 'k', k))
    

    plt.subplot(grid[1,0])
    plt.imshow(harriseps_img, cmap="gray",aspect="auto")
    plt.title("Corner_harris transformed \n sigma={}\n method={}, eps={}".format(sigma2, 'eps', e))
    plt.colorbar()

    plt.subplot(grid[1,1])
    plt.imshow(img, cmap="gray",aspect="auto")
    plt.colorbar()
    plt.scatter(local_max_eps[:,1],local_max_eps[:,0], marker='.', color='y')
    plt.title('Highlighted local maxima in feature map (ie. "the yellow dots") \n sigma={}\n method={}, eps={}'.format(sigma2, 'eps', e))

    plt.savefig('task1-3.png', bbox_inches='tight')

def gauss1d_spatial(size, std):
    Z = std * np.sqrt(2 * np.pi)
    xs = np.arange(np.ceil(-size / 2) , np.ceil(size / 2))
    kernel = np.exp(-xs**2 / (2 * std **2)) / Z
    return kernel / kernel.sum()

def my_kernel2d(shape, std, func1d=gauss1d_spatial):
    return np.einsum('i,j -> ij', func1d(shape[0], std), func1d(shape[1], std))

def task2_1():
    size = (21,21)
    kernel = np.zeros(size)
    kernel[10][10] = 1
    std = 1
    tau = 2

    kernel_spatial_B = my_kernel2d(kernel.shape, std=1)
    B_img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(kernel_spatial_B)))
    kernel_spatial_Ck = my_kernel2d(kernel.shape, std=2)
    Convolved_tau_img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(kernel_spatial_Ck)))

    kernel_spatial_GG = my_kernel2d(kernel.shape, std=np.sqrt(5))
    GG_img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(kernel_spatial_GG)))
    
    I_img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fft2(B_img) * np.fft.fft2(Convolved_tau_img)))
    diff_img = GG_img - I_img
    
    rows = 2
    cols = 3
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (15, 9), dpi=200)

    imgs = [kernel, B_img, Convolved_tau_img, GG_img, I_img, diff_img]
    titles = ["Original img",
    "B(x,y)--the blob image \n"+r"$\sigma = 1$",
    "Convolved img \n" + r"$\tau = 2$",
    r"$G(x,y,\sigma)*G(x,y,\tau)$" +"\n" + r"$\sigma = {}\tau = {}$".format(1,2),
    r"$I(x,y,\tau)$" +"\n"+ r"$\tau = 2$",
    "Difference"]
    # for i in range(rows):
    for j, img in enumerate(imgs):
        if j < 3:
            plt.subplot(grid[0,j])
            # plt.imshow(imgs[j], cmap="gray", aspect="auto")
            # plt.title(titles[i])
            # plt.colorbar()
        else:
            plt.subplot(grid[1,j-3])
        plt.imshow(img.real, cmap="gray", aspect="auto")
        plt.title(titles[j])
        plt.axis('off')
        plt.colorbar()
    plt.savefig('task2-1.png',bbox_inches='tight')

def I_dxx(x, y, tau, sigma):
    return -tau**2 * (tau**2 + sigma**2 - x**2) * np.exp(-(x**2 + y**2) / (2 * (tau**2 + sigma**2))) / (2 * np.pi * (tau**2 + sigma**2)**3)

def I_dyy(x, y, tau, sigma):
    return -tau**2 * (tau**2 + sigma**2 - y**2) * np.exp(-(x**2 + y**2) / (2 * (tau**2 + sigma**2))) / (2 * np.pi * (tau**2 + sigma**2)**3)

def H(x, y, tau, sigma=1.0):
    return I_dxx(x, y, tau, sigma) + I_dyy(x, y, tau, sigma)

def task2_3_c():
    tau = np.linspace(0., 3., 100)
    fig, ax = plt.subplots(dpi=150)
    ax.plot(tau, H(0,0,tau, sigma=1))
    ax.set_xlabel(r"$\tau$")
    
    ax.set_ylabel(r"$H(0,0,\tau)$")
    ax.set_xlim(0,3.2)
    ax.set_ylim(-0.1,0)
    ax.xaxis.tick_top()
    fig.savefig('task2-3-c.png', bbox_inches='tight')
task2_3_c()

def d2_Gaussian_kernel(x_len, y_len, tau):
  """
  computes the second derivative of a descrete Gaussian in respect to x and y
  INPUT:
    - x_len, y_len: size of the kernel
    - tau: standartd derivation
  """
  x = np.arange(-np.ceil(x_len/2), np.ceil(x_len/2))
  y = np.arange(-np.ceil(y_len/2), np.ceil(y_len/2))
  xx, yy = np.meshgrid(x, y)
  d2gauss_x = 1/(2.*np.pi*tau**6) * (xx**2 - tau**2) * np.exp(-(xx**2 + yy**2) / (2. * tau**2))
  d2gauss_y = 1/(2.*np.pi*tau**6) * (yy**2 - tau**2) * np.exp(-(xx**2 + yy**2) / (2. * tau**2))
  #summe = np.sum(gauss)
  return d2gauss_x, d2gauss_y

def find_local_max_min(H_img, n = 150, k = 1):
    """
    finds the n highest local maxima in the image
    INPUT:
    - H_img: image
    - n: number of max values
    - k: minimal distance between the peeks
    OUTPUT:
    - maximal: max values
    - minimal: min values
    """
    local_max_coords = feature.peak_local_max(H_img, min_distance=k, exclude_border = True)
    local_max = []
    for m in range(len(local_max_coords)):
        local_max.append(H_img[local_max_coords[m, 0], local_max_coords[m, 1]])

    maximal = np.zeros((len(local_max), 3))
    maximal[:, 0:2] = local_max_coords
    maximal[:, 2] = local_max
    maximal = maximal[maximal[:,2].argsort()]

    local_min_coords = feature.peak_local_max(-1*H_img, min_distance=k, exclude_border = True)
    local_min = []
    for m in range(len(local_min_coords)):
        local_min.append(H_img[local_min_coords[m, 0], local_min_coords[m, 1]])

    minimal = np.zeros((len(local_min), 3))
    minimal[:, 0:2] = local_min_coords
    minimal[:, 2] = np.absolute(local_min)
    minimal = minimal[minimal[:,2].argsort()]
    return maximal[len(local_max_coords)-n:], minimal[len(local_min_coords)-n:]

def H_taus(I, taus):
    """
    computes the scale space applied to the image I for several tau values. It 
    returns the maximal H value for every tau at the pixel location x, y
    """
    y, x = I.shape
    H_img = np.zeros((y, x, len(taus)))
    dummy = 0
    #computing H for eac tau value
    for tau in taus:
        # compute the second derivative of the Gaussian blob image
        # x_conv = ndimage.gaussian_filter(I, sigma=tau, order=(2,0))
        # y_conv = ndimage.gaussian_filter(I, sigma=tau, order=(0,2))
        # # compute H as the sum of two convolutions 
        # H_img[:, :, dummy] = tau**2 * (x_conv + y_conv)
        # dummy += 1
        d2Gx, d2Gy = d2_Gaussian_kernel(tau*5, tau*5, tau)
        #compute H as the sum of two convolutions 
        x_conv = signal.convolve2d(I, d2Gx, mode='same')#, boundary='symm')
        y_conv = signal.convolve2d(I, d2Gy, mode='same')#, boundary='symm')
        H_img[:, :, dummy] = tau**2 * (x_conv + y_conv)
        dummy += 1
    #computing the max value
    H_img_max = np.max(H_img, axis=2)
    # index map of the corresponding max tau
    ind = np.argmax(H_img, axis=2)
    return H_img_max, ind

def task2_4():
    img = imread('Week 6/sunflower.tiff', as_gray=True)
    tau_arr = [3, 5, 11, 15]
    # tau_arr = [2,5,6]
    H_map_test, ind_tau = H_taus(img, tau_arr)
    n = 150  #max = 421500
    max_vals, min_vals = find_local_max_min(H_map_test, n = n, k=1)

    fig, ax = plt.subplots(figsize = (12, 9), dpi=150)
    bg = ax.imshow(img, cmap = 'gray')
    # ax.plot(max_vals[:n, 1], max_vals[:n, 0], marker = '.', s = .1, color = 'blue', alpha =1)
    ax.scatter(max_vals[:n, 1], max_vals[:n, 0], marker='.', color = 'yellow') #marker = 'x', '1'
    for j in range(n):
        circle_max = plt.Circle((max_vals[j, 1], max_vals[j, 0]), 
            1.5*tau_arr[ind_tau[int(max_vals[j, 0]), int(max_vals[j, 1])]], 
            color='y', 
            fill = False)
        ax.add_artist(circle_max)

    ax.scatter(min_vals[:n, 1], min_vals[:n, 0], marker='.', color = 'red') #marker = 'x', '1'
    for j in range(n):
        circle_min = plt.Circle((min_vals[j, 1], min_vals[j, 0]), 
            0.75*tau_arr[ind_tau[int(min_vals[j, 0]), int(min_vals[j, 1])]], 
            color='r', 
            fill = False)
        ax.add_artist(circle_min)
    ax.set_title("The first 150 maxima and 150 minima detected in the Sunflower image.")
    fig.colorbar(bg)
    fig.savefig('task2-4.png',bbox_inches='tight')

def vertical_soft_edge(img_size, sigma):
  x_index = np.arange(-img_size[0]/2, img_size[0]/2)
  x_map = np.tile(x_index,(img_size[1], 1))
  img = 1 / (2*np.pi*sigma**2)*np.exp(-x_map**2/2*sigma**2)
  return np.cumsum(img, axis=1)

def convolution(img, kernel):
  kernel_size = kernel.shape[0]
  center = int((kernel_size-1)/2)
  New_H = img.shape[0]
  New_W = img.shape[1]
  img_res = np.ones((New_H, New_W))
  img = np.pad(img, pad_width=center)
  for i in range(New_H):
    for j in range(New_W):
      # becuase we have padding in x so we need use the index x+center
        cur_windows = img[i:i+kernel_size, j:j+kernel_size]
        print(cur_windows.shape)
        img_res[i][j] = (cur_windows * kernel).sum() / abs(kernel).sum()
  return np.array(img_res)

def scale_normalized(x,y,tau,sigma):
  return (tau) * np.exp(-(x**2) / (sigma**2 + tau**2)) / (np.pi * (sigma**2 + tau**2))

def find_maxima(H_img, n=150, k=1):
    """
    finds the n highest local maxima in the image
    INPUT:
    - H_img: image
    - n: number of max values
    - k: minimal distance between the peeks
    OUTPUT:
    - maximal: max values
    """
    local_max_coords = feature.peak_local_max(H_img, min_distance=k, exclude_border = True)
    local_max = []
    for m in range(len(local_max_coords)):
        local_max.append(H_img[local_max_coords[m, 0], local_max_coords[m, 1]])

    maximal = np.zeros((len(local_max), 3))
    maximal[:, 0:2] = local_max_coords
    maximal[:, 2] = local_max
    return maximal

def scale_selection(I, taus):
    """
    computes the scale space applied to the image I for several tau values. It 
    returns the maximal H value for every tau at the pixel location x, y
    """
    y, x = I.shape
    H_img = np.zeros((y, x, len(taus)))
    dummy = 0
    #computing H for eac tau value
    for tau in taus:
        img_x = gaussian_filter(I, tau, order=[1,0])
        img_y = gaussian_filter(I, tau, order=[0,1])
        img_conv = tau * (img_x**2 + img_y**2)
        H_img[:, :, dummy] = img_conv
        dummy += 1
    #computing the max value
    H_img_max = np.max(H_img, axis=2)
    ind = np.argmax(H_img, axis=2)
    return(H_img_max, ind)
  
def task_3_1():
  img_size = [21,21]
  img = vertical_soft_edge(img_size, 1)
  grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.2)
  tau_list = np.array([[0, 0.1, 0.5], [1, 5, 10]])
  plt.figure(figsize = (16, 12))
  for i in range(2):
    if i == 0:
      plt.subplot(grid[i,0])
      plt.imshow(img, cmap="gray")
      plt.colorbar()
      plt.title("original img, $\\sigma=1$")
    else:
      
      gauss_img = gaussian_filter(img.copy(), tau_list[i,0])
      plt.subplot(grid[i,0])
      plt.imshow(gauss_img, cmap="gray")
      plt.colorbar()
      plt.title(f"$\\tau = $"+str(tau_list[i,0]))

    gauss_img = gaussian_filter(img.copy(), tau_list[i,1])
    plt.subplot(grid[i,1])
    plt.imshow(gauss_img, cmap="gray",)
    plt.colorbar()
    plt.title(f"$\\tau = $"+str(tau_list[i,1]))

    gauss_img = gaussian_filter(img.copy(), tau_list[i,2])
    plt.subplot(grid[i,2])
    plt.imshow(gauss_img, cmap="gray",)
    plt.colorbar()
    plt.title(f"$\\tau = $"+str(tau_list[i,2]))
  plt.savefig("task3-1", bbox_inches='tight')
  plt.close()

def task_3_2():
  sigma = 1
  tau = np.arange(0.0, 3.0, 0.01)
  x = 0
  y = 0
  J_nable = scale_normalized(x,y,tau,sigma)
  plt.figure(figsize = (18, 8))
  plt.plot(tau, J_nable)
  plt.title("function of $\\tau$, when $\\tau = \\sigma = 1.0$ is the maximum value")
  plt.savefig("task3-2-3", bbox_inches='tight')
  plt.close()

def task3_3():
  img = imread('./Week 6/hand.tiff', as_gray=True)
  n=200
  tau_arr = [3, 5, 9]
  edge_img, ind_tau = scale_selection(img.copy(), tau_arr)
  maxima = find_maxima(edge_img.copy() ,n)
  fig, ax = plt.subplots(figsize = (12, 9), dpi=150)
  bg = ax.imshow(img, cmap = 'gray')
  ax.scatter(maxima[:n, 1], maxima[:n, 0], color = 'red') #marker = 'x', '1'
  for j in range(n):
        circle_max = plt.Circle((maxima[j, 1], maxima[j, 0]), 
            tau_arr[ind_tau[int(maxima[j, 0]), int(maxima[j, 1])]], 
            color='r', 
            fill = False)
        ax.add_artist(circle_max)
  ax.set_title("original img")
  fig.colorbar(bg)
  fig.savefig("task3-3.png",bbox_inches='tight')
      
if __name__== "__main__":
    task1_1()
    task1_2()
    task1_3()
    task2_1()
    task2_3_c()
    task2_4()
    task_3_1()
    task_3_2()
    task3_3()