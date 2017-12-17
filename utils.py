import numpy as np
import scipy.misc

def get_images(filename, is_crop, fine_size, images_norm):
    img = scipy.misc.imread(filename, mode='RGB')
    if is_crop:
        size = img.shape
        start_h = int((size[0] - fine_size)/2)
        start_w = int((size[1] - fine_size)/2)
        img = img[start_h:start_h+fine_size, start_w:start_w+fine_size,:]
    img = np.array(img).astype(np.float32)
    if images_norm:
        img = (img-127.5)/127.5
    return img

def save_images(images, size, filename):
    return scipy.misc.imsave(filename, merge_images(images, size))

def merge_images(images, size):
    h,w = images.shape[1], images.shape[2]
    imgs = np.zeros((size[0]*h,size[1]*w, 3))
    
    for index, image in enumerate(images):
        i = index//size[1]
        j = index%size[0]
        imgs[i*h:i*h+h, j*w:j*w+w, :] = image

    return imgs
