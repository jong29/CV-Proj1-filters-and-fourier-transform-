from tkinter import Y
import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    # numpy implementation of fftshift: https://github.com/numpy/numpy/blob/main/numpy/fft/helper.py
    x_size = img.shape[0]
    y_size = img.shape[1]
    tmp_img = img.copy()
    cent_x = x_size//2
    cent_y = y_size//2

    for i in range(x_size):
        for j in range(y_size):
            x=i
            y=j
            if i < cent_x:
                x += cent_x
            else:
                x -= cent_x
            if j < cent_y:
                y += cent_y
            else:
                y -= cent_y
            tmp_img[x,y] = img[i,j]
    return tmp_img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    x_size = img.shape[0]
    y_size = img.shape[1]
    tmp_img = img.copy()
    cent_x = x_size//2
    cent_y = y_size//2

    for i in range(x_size):
        for j in range(y_size):
            x=i
            y=j
            if i < cent_x:
                x += cent_x
            else:
                x -= cent_x
            if j < cent_y:
                y += cent_y
            else:
                y -= cent_y
            tmp_img[x,y] = img[i,j]
    return tmp_img

def fm_spectrum(img):
    '''
    This function should get the frequenccy magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    
    frimg = np.fft.fft2(img)
    frimg = fftshift(frimg)
    fmspec = 20*np.log(np.abs(frimg))
    # fmspec = np.asarray(fmspec)
    
    return fmspec


def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    frimg = np.fft.fft2(img)
    fmspec = fftshift(frimg)

    h = img.shape[0]
    w = img.shape[1]
    lpf = np.zeros((h, w), np.uint8)

    for i in range(h):
        for j in range(w):
            if (i - (h)/2)**2 + (j - (w)/2)**2 <= r**2:
                lpf[i, j] = 1

    conv=lpf*fmspec

    rvshift = ifftshift(conv)
    rvfour = np.fft.ifft2(rvshift)
    rvfour = np.abs(rvfour)

    return np.asarray(rvfour, np.uint8)

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    frimg = np.fft.fft2(img)
    fmspec = fftshift(frimg)

    h = img.shape[0]
    w = img.shape[1]

    hpf = np.zeros((h, w), np.uint8)

    for i in range(h):
        for j in range(w):
            if (i - (h)/2)**2 + (j - (w)/2)**2 >= r**2:
                hpf[i, j] = 1

    conv=hpf*fmspec

    rvshift = ifftshift(conv)
    rvfour = np.fft.ifft2(rvshift)

    temp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            temp[i,j] = rvfour[i,j].real
    
    return temp

def denoise1(img, r1=5, r2=10):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''

    frimg = np.fft.fft2(img)
    fmspec = fftshift(frimg)

    h = img.shape[0]
    w = img.shape[1]
    brf = np.zeros((h, w), np.uint8)

    for i in range(h):
        for j in range(w):
            coor = (i - (h)/2)**2 + (j - (w)/2)**2
            if coor>=r1**2 and coor<=r2**2:
                brf[i, j] = 1

    conv=brf*fmspec

    rvshift = ifftshift(conv)
    rvfour = np.fft.ifft2(rvshift)
    rvfour = np.abs(rvfour)

    return rvfour

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    return img

#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    return img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    return img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()