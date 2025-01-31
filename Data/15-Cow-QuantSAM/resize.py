import cv2

def resize(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':
    input_im = cv2.imread('input.png')
    seg_im = cv2.imread('seg.png')
    size = input_im.shape[:2][::-1]
    # size = (size[0] // 2, size[1] // 2)
    input_im = resize(input_im, size)
    seg_im = resize(seg_im, size)
    cv2.imwrite('input.png', input_im)
    cv2.imwrite('seg.png', seg_im)
    import numpy as np
    mask_im = np.ones_like(input_im) * 255
    cv2.imwrite('mask.png', mask_im)