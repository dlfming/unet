import cv2
import numpy as np
# make label iamge
def make_label(image):
    return iamge

# slide then cut image and label 
def slide_cut(image, label):
    pass

if __name__ == '__main__':
    image = cv2.imread('data/train/image/0.png', 1)
    label = cv2.imread('data/train/label/0.png', -1)

    label = np.expand_dims(label,-1)

    print(image.shape)
    print(label.shape)

    img = np.concatenate((image,label),axis=-1)
    img = cv2.resize(img, (5120, 512), interpolation=cv2.INTER_NEAREST)
    print(img.shape)

    for n in range(512,5632,512):
        temp = img[0:512,n-512:n]
        cv2.imshow('image', temp[:,:,:3])
        cv2.waitKey()
    # cv2.imshow('image', img[:,:,:3])
    # cv2.imshow('label', img[:,:,3:])
    # cv2.waitKey()
    cv2.destroyAllWindows()
    
