import cv2
import matplotlib
import numpy as npimport


# %matplotlib inline
def cv_show(name, image):
    cv2.imshow(name, image)
    # 等待时间,毫秒级
    cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('cat.jpg')
    cv2.IMREAD_GRAYSCALE
    cv2.destroyAllWindows()

    vc = cv2.VideoCapture('laughing.mp4')
    # 检查是否打开
    if vc.isOpened():
        isOpen, frame = vc.read()
    else:
        isOpen = False
    while isOpen:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(10) & 0xFF == 27:
                break
    vc.release()
    cv2.destroyAllWindows()

    # 截取部分图像数据
    img = cv2.imread('dog.jpeg')
    dog = img[0:200, 0:200]
    cv_show('dog', dog)

    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                    borderType=cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

    import matplotlib.pyplot as plt

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')
    plt.show()

    img_cat = cv2.imread('cat.jpg')
    img_cat2 = img_cat + 10
    img_cat[:5, :0]
    img_cat2[:5, :0]
    (img_cat2 + img_cat)[:5, :0]

    cv2.add(img_cat2, img_cat)[:5, :, 0]

    img_cat = cv2.imread('dog.jpeg')
    img_wechat = cv2.imread('wechat.jpeg')

    img_wechat = cv2.resize(img_wechat, 1440, 1080, 3)
    img_wechat.shape

    res = cv2.addWeighted(img_cat, 0.4, img_wechat, 0.6)
    plt.imshow(res)

ret, thresh1 = cv2.threshold(img_dog, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_dog, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_dog, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_dog, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_dog, 127, 255, cv2.THRESH_TOZERO_INV)
ret, thresh5 = cv2.threshold(srz, thresh, maxval, type)

img = sp_noise(img, 0.2)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]


def plt_show(titles, images):
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


import random


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


img = cv2.imread('elon.png')
img_with_salt = sp_noise(img, 0.2)
# 方框滤波
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# 高斯滤波
gauss = cv2.GaussianBlur(img, (5, 5), 1)
# 均值滤波
blur = cv2.blur(img, (3, 3))
# 中值滤波
median = cv2.medianBlur(img, 5)

titles = ['original', 'salt', 'blur', 'box', 'gauss', 'median']
images = [img, img_with_salt, blur, box, gauss, median]

plt_show(titles, images)

cv_show(blur)

box = cv2.boxFilter(img, -1, (3, 3), normalize=True)

gauss = cv2.GaussianBlur(img, (5, 5), 1)

# void ColorSalt(Mat& image, int n)//本函数加入彩色盐噪声
# {
#         srand((unsigned)time(NULL));
#         for (int k = 0; k<n; k++)//将图像中n个像素随机置零
#         {
#                 int i = rand() % image.cols;
#                 int j = rand() % image.rows; RAND_MAX
#                 //将图像颜色随机改变
#                 image.at<Vec3b>(j, i)[0] = 250;
#                 image.at<Vec3b>(j, i)[1] = 150;
#                 image.at<Vec3b>(j, i)[2] = 250;
#         }
#
#         }


kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

cv2.dilate(img, iterations=1)

cv2.imshow('erosion', erosion)

img = cv2.imread('erode.png')
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
rect = cv2.morphologyEx(img, cv2.MORPH_RECT, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelX = cv2.convertScaleAbs(sobelX)

sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelY = cv2.convertScaleAbs(sobelY)

sobelXY = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
sobelXY_union = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelXY_union_abs = cv2.convertScaleAbs(sobelXY_union)

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx_abs = cv2.convertScaleAbs(scharrx)
scharry_abs = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

res = np.hstack((scharrxy))

img = cv2.imread('elon.png', cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

up = cv2.pyrUp(img)
down = cv2.pyrDown(img)

down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
original_down_up = img - down_up

img = cv2.imread('dindang.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_cp = img.copy()
res = cv2.drawContours(img_cp, contours, -1, (0, 0, 255), 2)
res1 = cv2.drawContours(img_cp, contours, 0, (0, 0, 255), 2)

area = cv2.contourArea(cnt)
arc_len = cv2.arcLength(cnt, True)
print(area)
print(arc_len)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[1]
img_cp = img.copy()
res = cv2.drawContours(img_cp, contours, -1, (0, 0, 255), 2)

epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
img_cp = img.copy()
res = cv2.drawContours(img_cp, [approx], -1, (0, 0, 255), 2)

cnt = contours[2]
draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)  # 参数为输入图像  轮廓  填充颜色  线宽 -1表示填充所有轮廓  参数为输出图像
cv_show(res, 'after_drawContours')
# 轮廓近似
epsilon = 0.08 * cv2.arcLength(cnt, True)  # 近似轮廓的精度  参数为轮廓  参数为是否闭合 精度越小越精确
approx = cv2.approxPolyDP(cnt, epsilon, True)  # 近似轮廓  参数为轮廓  精度  参数为是否闭合
print('轮廓近似后的面积为', cv2.contourArea(approx))  # 近似后的面积
print('轮廓近似后的周长为', cv2.arcLength(approx, True))
draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)  # 参数为输入图像  轮廓  填充颜色  线宽 -1表示填充所有轮廓  参数为输出图像
cv_show(res, 'after_drawContours')

x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.TM_CCOEFF_NORMED

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

img = cv2.imread('liangxiao.jpeg')
template = cv2.imread('lx_face.png')

for meth in methods:
    img2 = img.copy()
    # 匹配方法的真值
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 如果是平方差(TM_SQDIFF)匹配或归一化平方差(TM_SQDIFF_NORMED) 取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    # 隐藏坐标轴
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

img = cv2.imread('elon.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('elon_face.png', 0)
h, w = template.shape[:2]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于80%的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256)
plt.show()

img = cv2.imread('elon.png')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim(0, 256)

# 创建mask
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255
# 与操作
masked_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

img = cv2.imread('elon.png')
plt.hist(img.ravel(), 256)
plt.show()

equ = cv2.equalizeHist(img_as_float(img))

plt.hist(equ.ravel(), 256)
plt.show()

Mat
Image(512, 512, CV_32FC1);
Image = imread("C:\\MGC.jpg", CV_LOAD_IMAGE_GRAYSCALE);

import cv2
from skimage import img_as_float

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
image2 = img_as_float(image1)
cv2.imshow('IMAGE1', image1)
cv2.imshow('IMAGE2', image2)
while (1):
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
\

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)

import numpy as np

img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表达的形式
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude_spectrum')
plt.xticks([]), plt.yticks([])
plt.show()

img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# 高通滤波
mask_high = np.ones((rows, cols, 2), np.uint8)
mask_high[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
fshift_high = dft_shift * mask_high
f_ishift_high = np.fft.ifftshift(fshift_high)
img_back_high = cv2.idft(f_ishift_high)
img_back_high = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])
# IDFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

img = cv2.imread('approx.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)

kp, des = sift.compute(gray, kp)

img1 = cv2.imread('cat.jpg', 0)
img2 = cv2.imread('cat.jpg', 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img1, None)
# crossCheck表示两个特征点要相互匹配,例如A中的第i个特征点与B中的第j个特征点最近的,并且B中的第j个特征点到A中的第i个特征点也是最近的
bf = cv2.BFMatcher(crossCheck=True)

# 1对1匹配
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# k对最佳匹配
bf = cv2.BFMatcher();
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

