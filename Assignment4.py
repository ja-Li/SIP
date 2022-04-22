import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.morphology import binary_opening, binary_closing, disk, white_tophat, black_tophat, remove_small_objects, erosion, dilation
from skimage.segmentation import watershed
from skimage.transform import rescale,rotate,resize
from skimage import util, feature, measure
from skimage.filters import difference_of_gaussians, threshold_otsu, threshold_yen
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_hit_or_miss
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import convolve2d
from scipy import fft

def task1_1_1():
    img = np.array(imread('./Week 5/cells_binary.png', as_gray=True),dtype='uint8')
    bi_img = img > threshold_otsu(img)
    img_opening = binary_opening(bi_img, selem=disk(2))
    img_closing = binary_closing(bi_img, selem=disk(2))

    # print(bi_img[100])
    img_res = [bi_img, img_opening, img_closing]
    img_res_name = ['Binary_Orgin', "after opening operation", "after closing operation" ]

    # zoom_up
    zoom_up = 2
    row = 2
    cols = 3
    grid = plt.GridSpec(nrows=row, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (15, 8), dpi=150)
    for i in range(row):
        for j, name in enumerate(img_res_name):
            if i == 0:
                plt.subplot(grid[i, j])
                plt.imshow(img_res[j], cmap="gray",aspect="auto")
                plt.colorbar()
                plt.title(name) 
            else:
                ax = plt.subplot(grid[i, j])
                plt.imshow(img_res[j], cmap="gray",aspect="auto")
                plt.colorbar()
                plt.title(name+" zoomed") 
                axins = zoomed_inset_axes(ax, zoom_up, loc=1)
                axins.imshow(img_res[j], cmap="gray", aspect="auto")
                # sub region of the original image
                h1, h2, w1, w2 = 200, 300, 350, 500
                axins.set_ylim(h1, h2)
                axins.set_xlim(w1, w2)
                # close the axis number
                plt.setp(axins.get_xticklabels(), visible=False)
                plt.setp(axins.get_yticklabels(), visible=False)
                # draw a bbox of the region of the inset axes in the parent axes and
                # connecting lines between the bbox and the inset axes area
                mark_inset(ax, axins, loc1=2, loc2=4, fc="None", ec="r")
    plt.savefig("task1_1_1.png", bbox_inches='tight')


def task1_2():
    img = np.array(imread('./Week 5/blobs_inv.png', as_gray=True),dtype='uint8')
   
    bi_img = img > threshold_otsu(img)
    mask_vline = np.ones((5,1))
    mask_disc = disk(2)
    mask_corner = np.array([[0,0,0],[1,1,0],[1,1,0]])

    algo = [binary_hit_or_miss, white_tophat, black_tophat]
    algo_name = ['hit-or-miss','TopHat','BottomHat']
    masks = [mask_vline, mask_disc, mask_corner]
    masks_name = ['vertical line','disc','corner']
    
    row = 3
    cols = 3

    plt.figure()
    plt.imshow(img,cmap="gray",aspect="auto",vmin=0,vmax=1)
    plt.colorbar()
    plt.title('origin')
    plt.savefig('task1-2_origin.png',bbox_inches='tight')

    grid = plt.GridSpec(nrows=row, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (14, 12), dpi=150)

    for i, a_name in enumerate(algo):
        for j, mask in enumerate(masks):
            res_img = a_name(bi_img, mask)
            plt.subplot(grid[i,j])
            plt.imshow(res_img, cmap="gray",aspect="auto",vmin=0,vmax=1)
            plt.colorbar()
            plt.title(algo_name[i] + " with "+masks_name[j])
    plt.savefig('task1-2.png',bbox_inches='tight')
    # plt.imshow(mask_corner,cmap="gray",aspect="auto")


def task1_3():
    img = imread('./Week 5/digits_binary_inv.png', as_gray=True)  
    bi_img = img > threshold_otsu(img)

    mask_1st = bi_img[5:36][:,65:87]
    # mask_2= bi_img[286:309][:,70:86]
    mask_ad = resize(erosion(mask_1st), mask_1st.shape)

    masks = [mask_1st, mask_ad]
    masks_name = ['mask_1st_digit','mask_adapted']

    row = 2
    cols = 2
    zoom_up = 15
 
    grid = plt.GridSpec(nrows=row, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (12, 7), dpi=150)

    for j, mask in enumerate(masks):
        # if algo_name[i] == 'hit-or-miss':
        res_img = binary_hit_or_miss(bi_img, mask)
        pix = np.argwhere(res_img == True)
        print(pix)
        if masks_name[j] == 'mask_1st_digit':
            plt.subplot(grid[j,0])
            plt.imshow(mask_1st, cmap="gray", aspect="auto")
            plt.colorbar()
            plt.title('Mask')
            
            ax = plt.subplot(grid[j,1])
            plt.imshow(res_img, cmap="gray",aspect="auto",vmin=0,vmax=1)
            plt.colorbar()
            plt.title("With "+ masks_name[j])

            axins = zoomed_inset_axes(ax, zoom_up, loc=1)
            axins.imshow(res_img, cmap="gray", aspect="auto")
            h1, h2, w1, w2 = 15, 25, 70, 80
            axins.set_ylim(h1, h2)
            axins.set_xlim(w1, w2)
            
            # close the axis number
            plt.setp(axins.get_xticklabels(), visible=False)
            plt.setp(axins.get_yticklabels(), visible=False)
            axins.spines['left'].set_color('red')
            axins.spines['bottom'].set_color('red')
            axins.spines['top'].set_color('red')
            axins.spines['right'].set_color('red')

            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(ax, axins, loc1=2, loc2=4, fc="None", ec="r")
        else:
            plt.subplot(grid[j,0])
            plt.imshow(mask_ad, cmap="gray", aspect="auto")
            plt.colorbar()
            plt.title('Adapted Mask')
            
            ax = plt.subplot(grid[j,1])
            plt.imshow(res_img, cmap="gray",aspect="auto",vmin=0,vmax=1)
            plt.colorbar()
            plt.title("With "+ masks_name[j])

            # axins = zoomed_inset_axes(ax, zoom_up, loc=1)
            # axins.imshow(res_img, cmap="gray", aspect="auto")
            # h1, h2, w1, w2 = 15, 25, 70, 80
            # axins.set_ylim(h1, h2)
            # axins.set_xlim(w1, w2)
            
            # # close the axis number
            # plt.setp(axins.get_xticklabels(), visible=False)
            # plt.setp(axins.get_yticklabels(), visible=False)
            # axins.spines['left'].set_color('red')
            # axins.spines['bottom'].set_color('red')
            # axins.spines['top'].set_color('red')
            # axins.spines['right'].set_color('red')

            # # draw a bbox of the region of the inset axes in the parent axes and
            # # connecting lines between the bbox and the inset axes area
            # mark_inset(ax, axins, loc1=2, loc2=4, fc="None", ec="r")
    plt.savefig('task1_3.png', bbox_inches='tight')
    
    bi_show = bi_img.copy()
    img_t = binary_hit_or_miss(bi_show, mask_1st)
    img_t_dilation = dilation(img_t, disk(20))
    img_r = (img_t_dilation + util.invert(bi_show) * 255) > 255

    plt.figure(figsize = (6, 4),dpi=150)
    plt.imshow(img_r, cmap='jet')
    plt.colorbar()
    plt.title('colored cross')
    plt.savefig("task1-3-color-cross.png", bbox_inches='tight')



def task1_4():
    img = imread('./Week 5/money_bin.jpg',as_gray=True)
    
    bi_img = img < threshold_otsu(img)
    coins = binary_closing(bi_img,disk(2))
    
    #reference: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
    distance_im = ndi.distance_transform_edt(coins)
    print ('distance transform:', distance_im.shape, distance_im.dtype)
    
    coords = feature.peak_local_max(distance_im)
    mask = np.zeros(distance_im.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers_im, _ = ndi.label(mask)

    labelled_coins = remove_small_objects(watershed(distance_im, markers_im, mask=coins), min_size=10)

     # subtract 1 b/c background is labelled 0
    num_coins = len(np.unique(labelled_coins))-1 
    print ('number of coins: %i' % num_coins)

    properties = measure.regionprops(labelled_coins)
    coin_areas = [int(prop.area) for prop in properties]
    coin_centroids = [prop.centroid for prop in properties]

    properties = measure.regionprops(labelled_coins)

    coin_areas = np.array([prop.area for prop in properties])

    #print num_each_coin
    num_1kr = len(np.where(coin_areas < 3100)[0])
    num_2kr = len(np.where( (4100 < coin_areas) & (coin_areas < 4500))[0])
    num_5kr = len(np.where( (5800 < coin_areas))[0])
    num_20kr = len(np.where( (5250 < coin_areas) & (coin_areas < 5750))[0])
    num_50ore = len(np.where( (3150 < coin_areas) & (coin_areas < 3800))[0])

    rows = 1 
    cols = 4
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (20, 4), dpi=150)

    # plt.figure()
    plt.subplot(grid[0,0])
    plt.imshow(img, cmap='gray',aspect='auto')
    plt.title("Origin")
    plt.colorbar()

    plt.subplot(grid[0,1])
    plt.imshow(coins, cmap='gray', interpolation='none')
    plt.title('closed coins')
    plt.colorbar()

    plt.subplot(grid[0,2])
    mask_p = mask > 0
    mask_p = np.ma.masked_where(~mask_p, mask_p)        
    plt.imshow(distance_im, alpha=1, cmap='gray')
    plt.imshow(mask_p, alpha=1, cmap='gray')
    plt.title('labelled coins')
    plt.colorbar()

    plt.subplot(grid[0,3])
    plt.imshow(labelled_coins, cmap='jet', aspect='auto')
    plt.title('labelled coins')
    plt.colorbar()
    # plt.figure()
    # imshow_overlay(distance_im, markers_im, alpha=1, cmap='gray')
    # my_imshow(labelled_coins, 'labelled coins', cmap='jet')
    for lab in range(len(coin_areas)):
        plt.text(coin_centroids[lab][1]-30,coin_centroids[lab][0],coin_areas[lab])
    # plt.colorbar()

    plt.savefig("task1_4.png", bbox_inches='tight')
    
    print ('number of 1kr: %i' % num_1kr)
    print ('number of 2kr: %i' % num_2kr)
    print ('number of 5kr: %i' % num_5kr)
    print ('number of 20kr: %i' % num_20kr)
    print ('number of 50ore: %i' % num_50ore)

    print ('Total value in image: DKK %.2f' % (num_1kr*1+num_2kr*2+num_5kr*5+num_20kr*20+num_50ore*0.5))



def load_data():
    train_x = np.loadtxt("./Week 5/SIPdiatomsTrain.txt", delimiter= ',')
    train_y = np.loadtxt("./Week 5/SIPdiatomsTrain_classes.txt", delimiter= ',')
    test_x = np.loadtxt("./Week 5/SIPdiatomsTest.txt", delimiter= ',')
    test_y = np.loadtxt("./Week 5/SIPdiatomsTest_classes.txt", delimiter= ',')
    return train_x, train_y, test_x, test_y

def procrustes(target, input):
    # Translation, center 0
    diatom_x = target[0::2]
    diatom_y = target[1::2]

    diatom_c = np.array([diatom_x - np.mean(diatom_x), diatom_y - np.mean(diatom_y)])
    # print(diatom_c[0,:].shape)
    input_modified = np.zeros(input.shape)

    for i in range(len(input)):
        input_x = input[i][0::2]
        input_y = input[i][1::2]

        # Translation, center 0
        input_translated = np.array([input_x-np.mean(input_x), input_y-np.mean(input_y)])
        # print((input_translated @ diatom_c.T).shape)

        # Rotate
        U, S, V = np.linalg.svd(input_translated @ diatom_c.T)
        # print(U.shape, V.shape, input_translated[0,:].shape)
        rotated = (V.T @ U.T) @ input_translated
        # Scale
        s_input_x = rotated[0,:]
        s_input_y = rotated[1,:]

        topSum = 0
        botSum = 0
        for x in range(len(diatom_x)):
            y_n  = np.array([s_input_x[x],s_input_y[x]])
            x_n  = np.array([diatom_x[x],diatom_y[x]])
            topSum += x_n @ y_n
            botSum += y_n @ y_n

        scaleSum = topSum/botSum
        scaled = rotated * scaleSum

        input_modified[i, 0::2] = scaled[0, :]
        input_modified[i, 1::2] = scaled[1, :]

    return diatom_c, input_modified

def task2_1():
    tx, ty, ttx, tty = load_data()
    transformed_target, res = procrustes(tx[0],tx)
    transformed_target_2, res_test = procrustes(tx[1],ttx)
    rows = 2
    cols = 1
    grid = plt.GridSpec(nrows=rows, ncols=cols, wspace=0.2, hspace=0.2)
    plt.figure(figsize = (10, 12), dpi=150)
    plt.subplot(grid[0,0])
    # plt.scatter(tx[0,0::2],tx[0,1::2], label='target diatom')
    plt.scatter(transformed_target[0::2],transformed_target[1::2], label='target diatom 1')

    plt.subplot(grid[1,0])
    # plt.scatter(tx[0,0::2],tx[0,1::2], label='target diatom')
    plt.scatter(transformed_target_2[0::2],transformed_target_2[1::2], label='target diatom 2')
    
    num = np.random.randint(2.,4.)
    for i in np.arange(1,num):
        plt.subplot(grid[0,0])
        plt.scatter(tx[i,0::2],tx[i,1::2],label='Training example {}'.format(str(i)))
        plt.scatter(res[i,0::2],res[i,1::2],label='Aligned training example {}'.format(str(i)))
        
    plt.legend()
    plt.title("Illustration of a few examples before and after alignment, along with the target diatom")
    # plt.savefig('task2-1.png', bbox_inches='tight')

    for i in np.arange(1,num):
        plt.subplot(grid[1,0])
        plt.scatter(ttx[i,0::2],ttx[i,1::2],label='Testing example {}'.format(str(i)))
        plt.scatter(res_test[i,0::2],res_test[i,1::2],label='Aligned testing example {}'.format(str(i)))
    plt.legend()
    plt.title("Illustration of a few examples before and after alignment, along with the target diatom")
    plt.savefig('task2-1.png', bbox_inches='tight')


def task2_2():
    train_x, train_y, test_x, test_y = load_data()
    transformed_target, res_train = procrustes(train_x[0],train_x)
    transformed_target, res_test = procrustes(train_x[0], test_x)

    knn1 = KNeighborsClassifier()
    knn1.fit(train_x, train_y)
    knn1_predictions = knn1.predict(test_x)
    print(sum(knn1_predictions==test_y),len(test_y))
    acc = sum(knn1_predictions==test_y) / len(test_y)
    print("Accuracy without align: {}".format(acc))

    knn2 = KNeighborsClassifier()
    knn2.fit(res_train, train_y)
    knn2_predictions = knn2.predict(res_test)
    print(sum(knn2_predictions==test_y),len(test_y))
    acc_aligned = sum(knn2_predictions==test_y) / len(test_y)
    print("Accuracy with aligned: {}".format(acc_aligned))


def task3_4():
    img = np.ones([15,15])
    img = np.pad(img, pad_width=5, constant_values=(0,0))
    plt.figure(figsize = (10, 10))
    plt.imshow(img, cmap="gray")
    plt.title("Original img")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig('task_3-4')

def translation_linear_filter(img, x, y):
    kernel = np.zeros([2*abs(x)+1, 2*abs(y)+1])
    kernel[x + abs(x), y + abs(y)] = 1
    result = convolve2d(img, kernel, mode='same')
    return result

def task3_5():
    img = np.ones([15,15])
    img = np.pad(img, pad_width=5, constant_values=(0,0))

    img1 = translation_linear_filter(img, 3, 3)
    img2 = translation_linear_filter(img, -3, -3)
    img3 = translation_linear_filter(img, 7, 0)

    plt.figure(figsize = (15, 5))

    plt.subplot(1,4,1) 
    plt.imshow(img, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Original img")

    plt.subplot(1,4,2) 
    plt.imshow(img1, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = 3, y = 3 ")

    plt.subplot(1,4,3) 
    plt.imshow(img2, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = -3, y = -3  ")

    plt.subplot(1,4,4) 
    plt.imshow(img3, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = 7, y = 0 ")
    plt.tight_layout()
    plt.savefig('task_3-5')

def h_coord_translate_interpolation(img, x, y):
    img_x, img_y = img.shape
    result = np.zeros([img_x, img_y])
    H_translate = np.array([[1,0,x],[0,1,y],[0,0,1]]) 
    for i in range(img_x):
        for j in range(img_y):
          x_prime,y_prime,_ = H_translate @ np.array([i,j,1])
          x_prime=round(x_prime) % img_x
          y_prime=round(y_prime) % img_y
          result[x_prime, y_prime] =  img[i,j]
    return result

def task3_6():
    img = np.ones([15,15])
    img = np.pad(img, pad_width=5, constant_values=(0,0))
    x,y = 0.6,1.2
    img1 = h_coord_translate_interpolation(img, x, y)

    plt.figure(figsize = (10, 5))

    plt.subplot(1,2,1) 
    plt.imshow(img, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Original img")

    plt.subplot(1,2,2) 
    plt.imshow(img1, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("After Homogeneous transformation")
    plt.tight_layout()
    plt.savefig('task_3-6')

def translate_kernel(img, tx, ty):
  size = img.shape
  x, y = np.arange(size[1]), np.arange(size[0])
  yy, xx = np.meshgrid(x, y)
  kernel = np.exp(-2.j*np.pi*(xx *tx/size[0] + yy *ty/size[1])) 
  return kernel

def translate_fft(img, kernel):
  img_fft = fft.fftshift(fft.fft2(img))
  img_fft_d = img_fft*kernel
  img_d = fft.ifft2(fft.ifftshift(img_fft_d))
  return abs(img_d)

def task3_7():
    img = np.ones([15,15])
    img = np.pad(img, pad_width=5, constant_values=(0,0))
    kernel = translate_kernel(img, 3, 3)
    img1 = translate_fft(img, kernel)

    kernel = translate_kernel(img, -3, -3)
    img2 = translate_fft(img, kernel)

    kernel = translate_kernel(img, 7, 0)
    img3 = translate_fft(img, kernel)

    plt.figure(figsize = (15, 5))

    plt.subplot(1,4,1) 
    plt.imshow(img, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Original img")

    plt.subplot(1,4,2) 
    plt.imshow(img1, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = 3, y = 3 ")

    plt.subplot(1,4,3) 
    plt.imshow(img2, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = -3, y = -3  ")

    plt.subplot(1,4,4) 
    plt.imshow(img3, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Move x = 7, y = 0 ")
    plt.tight_layout()
    plt.savefig('task_3-7')

def task3_8():
    img = np.ones([15,15])
    img = np.pad(img, pad_width=5, constant_values=(0,0))
    kernel = translate_kernel(img, 2.1, 2.1)
    img1 = translate_fft(img, kernel)

    kernel = translate_kernel(img, 2.5, 2.5)
    img2 = translate_fft(img, kernel)

    kernel = translate_kernel(img, 2.9,2.9)
    img3 = translate_fft(img, kernel)

    plt.figure(figsize = (15, 5))

    plt.subplot(1,4,1) 
    plt.imshow(img, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Original img")

    plt.subplot(1,4,2) 
    plt.imshow(img1, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 2.1, y=2.1")
    plt.tight_layout()

    plt.subplot(1,4,3) 
    plt.imshow(img2, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 2.5, y=2.5")
    plt.tight_layout()

    plt.subplot(1,4,4) 
    plt.imshow(img3, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 2.9, y=2.9")
    plt.tight_layout()
    plt.savefig('task_3-8')
    plt.close()

    img = imread('./Week 5/basic_shapes.png',as_gray=True)
    img = np.pad(img, pad_width=50, constant_values=(0,0))
    kernel = translate_kernel(img, 10.5, 10.5)
    img1 = translate_fft(img, kernel)
    kernel = translate_kernel(img, 20.5, 20.5)
    img2 = translate_fft(img, kernel)
    kernel = translate_kernel(img, 30.5, 30.5)
    img3 = translate_fft(img, kernel)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,4,1) 
    plt.imshow(img, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Original img")

    plt.subplot(1,4,2) 
    plt.imshow(img1, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 10.5, y=10.5")
    plt.tight_layout()
    plt.subplot(1,4,3) 
    plt.imshow(img2, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 20.5, y=20.5")
    plt.tight_layout()
    plt.subplot(1,4,4) 
    plt.imshow(img3, cmap="gray")
    plt.axis('on')
    plt.colorbar()
    plt.title("Translation with x = 30.5, y=30.5")
    plt.tight_layout()
    plt.savefig('task_3-8-1')



if __name__== "__main__":
  task1_1_1()
  task1_2()
  task1_3()
  task1_4()
  task2_1()
  task2_2()
  task3_4()
  task3_5()
  task3_6()
  task3_7()
  task3_8()
