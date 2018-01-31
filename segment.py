import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import math
from otsu import OtsuThresholdMethod
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk,ball

class canny_op:
    def sobel_filter(self, img):
        # detect x,y edge
        y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])
        x = np.array([[-1,  0,  1],
                      [-2,  0,  2],
                      [-1,  0,  1]])
        Gy = signal.convolve(img, y) # detect _ edge
        Gx = signal.convolve(img, x) # detect | edge
        return Gy, Gx

    def get_grad_magnitude_and_angle(self, img):
        Gy, Gx = self.sobel_filter(img)
        grad_mag = np.hypot(Gx, Gy) # equal to sqrt(Gx**2 + Gy**2)
        angle = np.arctan2(Gy, Gx) # some neg angle
        angle += 2 * math.pi
        angle %= (2 * math.pi) #ensure angle between 0 to pi
        return grad_mag, angle

    def get_dir(self, angle):
        angle %= math.pi # we only need to compute 0 to pi
        pi = math.pi
        #detect 8 dir , so cutting 2pi into 16 pieces
        dir_NE = (angle > 2 * pi / 16) & (angle < 6 * pi / 16)
        dir_N = (angle >= 6 * pi / 16) & (angle <=10 * pi / 16)
        dir_NW = (angle > 10 * pi / 16) & (angle < 14 * pi / 16)
        dir_W = (angle >= 14 * pi / 16) | (angle <= 2 * pi / 16)
        return [dir_NE, dir_NW, dir_W, dir_N]

    def canny_nms(self, mag, angle):
        shape = mag.shape
        higher, lower = np.zeros(shape), np.zeros(shape)
        toLeft, toRight = np.zeros(shape), np.zeros(shape)
        downLeft, upRight = np.zeros(shape), np.zeros(shape)
        upLeft, downRight = np.zeros(shape), np.zeros(shape)
        # ------ vertical ------- #
        higher[:-1, :] = mag[1:, :]  # shift rows up
        lower[1:, :] = mag[:-1, :]  # shift rows down
        # ------ horizontal ------- #
        toLeft[:, :-1] = mag[:, 1:]  # shift rows left
        toRight[:, 1:] = mag[:, :-1]  # shift rows right
        # ------ diagForward ------- #  /
        downLeft[1:, :-1] = mag[:-1, 1:]
        upRight[:-1, 1:] = mag[1:, :-1]
        # ------ diagBackward ------- #  \
        downRight[1:, 1:] = mag[:-1, :-1]
        upLeft[:-1, :-1] = mag[1:, 1:]
        # -------------------------------
        diagFphi, diagBphi, horizPhi, vertPhi = self.get_dir(angle)
        thinVert = vertPhi & (mag > higher) & (mag >= lower)
        thinHoriz = horizPhi & (mag > toLeft) & (mag >= toRight)
        thinDiagF = diagFphi & (mag > downRight) & (mag >= upLeft)
        thinDiagB = diagBphi & (mag > downLeft) & (mag >= upRight)
        return [thinDiagF, thinDiagB, thinHoriz, thinVert]

    def smooth_image(self, im):
        gaussian = [2,  4,  5,  4, 2,
                    4,  9, 12,  9, 4,
                    5, 12, 15, 12, 5,
                    2,  4,  5,  4, 2,
                    4,  9, 12,  9, 4]
        gaussian = 1.0 / sum(gaussian) * np.reshape(gaussian, (5,5))
        return signal.convolve(im, gaussian, mode='same')

    def normalize_magnitude(self, mag):
        """ scales magnitude matrix back to 0 - 255 values """
        offset = mag - mag.min()  # offset mag so that minimum value is always 0
        if offset.dtype == np.uint8:
            raise
        normalized = offset * 255 / offset.max()  # now.. if this image isn't float, you're screwed
        return offset * 255 / offset.max()

    def get_combined_thinned_image(self, mag, phi):
        thinDiagF, thinDiagB, thinVert, thinHoriz = self.canny_nms(mag, phi)
        normalMag = self.normalize_magnitude(mag)
        thinNormalMag = np.array(normalMag * (thinDiagF + thinDiagB + thinVert + thinHoriz), dtype=np.uint8)  # convert to uint8 image format.
        return thinNormalMag

    def edge_tracking(self, weak, strong):
        """ hysteresis edge tracking: keeps weak pixels that are direct neighbors to strong pixels. Improves line detection.
        :param weak: an image thresholded by the lower threshold, such that it includes all weak and strong pixels
        :param strong: an image thresholded by the higher threshold, such that it includes only strong pixels
         """
        #weakOnly = weak - strong
        blurKernel = np.ones((3,3)) / 9
        strongSmeared = signal.convolve(strong, blurKernel, mode='same') > 0
        strongWithWeakNeighbors = weak & strongSmeared  # this is your normal result. trying for more will be expensive

        return strongWithWeakNeighbors

    def double_threshold(self, im):
        """ obtain two thresholds for determining weak and strong pixels. return two images, weak and strong,
        where strong contains only strong pixels, and weak contains both weak and strong
        """
        plt.hist(im.ravel(), 256, [1, 256])
        #plt.xlim([0, 256])
        plt.show()
        cv2.imwrite('123.jpg',im)
        otsu = OtsuThresholdMethod(im, 4)  # speedup of 4 keeps things pretty accurate but much faster
        #_, lowThresh, highThresh, tooHigh = otsu.calculate_n_thresholds(4)
        lowThresh, highThresh = otsu.calculate_2_thresholds()
        weakLines = im > lowThresh
        strongLines = im > highThresh
        return weakLines, strongLines

    def bound_cut(self, im):
        """ cutting edge picture equal to input image """
        #shape = im.shape
        cut_edgesFinal = np.copy(im[1:-1, 1:-1])
        cut_edgesFinal[0, :], cut_edgesFinal[:, 0], cut_edgesFinal[-1, :], cut_edgesFinal[:, -1] = False, False, False, False
        return cut_edgesFinal

    def find_edges(self, im):
        """ returns boolean array represting lines. to convert to image just use edges * 255 """
        if im.ndim > 2 and im.shape[-1] > 1:  # aka if we have a full color picture
            im = im[:, :, 0]  # sorry, we can only deal with one channel. I hope you loaded it as greyscale!
        smoothed = self.smooth_image(im)
        mag, phi = self.get_grad_magnitude_and_angle(smoothed)
        thinNormalMag = self.get_combined_thinned_image(mag, phi)
        weak, strong = self.double_threshold(thinNormalMag)
        cannyEdges = self.edge_tracking(weak, strong)
        cut_canntEdges = self.bound_cut(cannyEdges)
        cut_weak = self.bound_cut(weak)
        return cannyEdges, weak,cut_canntEdges, cut_weak


if __name__ == '__main__':
    #preprocessing & edge track
    f = 'input1.jpg'
    img1_gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #selem = disk(2)
    selem = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    #img1_gray = opening(img1_gray, selem)
    #img1_gray = closing(img1_gray, selem)
    #cv2.imwrite("smooth.jpg",img1_gray)

    canny = canny_op()
    edgesFinal, uncanny,cut_edgesFinal,cut_uncanny = canny.find_edges(img1_gray)
    img1_color = cv2.imread(f)
    # try do eliminate noise using morphology
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    cut_uncanny = cut_uncanny*1
    cut_uncanny = cut_uncanny.astype(np.uint8)
    #cut_uncanny1 = cv2.morphologyEx(cut_uncanny, cv2.MORPH_CLOSE, kernel1)
    cut_uncanny = cv2.morphologyEx(cut_uncanny1, cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    #cut_uncanny2 = cv2.morphologyEx(cut_uncanny, cv2.MORPH_CLOSE, kernel2)
    cut_uncanny2 = cv2.morphologyEx(cut_uncanny2, cv2.MORPH_OPEN, kernel2)
    cut_uncanny = cut_uncanny1 | cut_uncanny2
    #drawing red contour
    img1_strong = np.zeros(img1_color.shape)
    img1_strong[:, :, 0] = np.where(cut_edgesFinal == True, 0, img1_color[:, :, 0])
    img1_strong[:, :, 1] = np.where(cut_edgesFinal == True, 0, img1_color[:, :, 1])
    img1_strong[:, :, 2] = np.where(cut_edgesFinal == True, 255, img1_color[:, :, 2])
    img1_weak = np.zeros(img1_color.shape)
    img1_weak[:, :, 0] = np.where(cut_uncanny == 1, 0, img1_color[:, :, 0])
    img1_weak[:, :, 1] = np.where(cut_uncanny == 1, 0, img1_color[:, :, 1])
    img1_weak[:, :, 2] = np.where(cut_uncanny == 1, 255, img1_color[:, :, 2])

    cv2.imwrite('out1.jpg', img1_weak)

    cv2.imwrite('out1_strong.jpg', img1_strong)
    cv2.imwrite('out1_weak.jpg', img1_weak)
    cv2.imwrite('strong_edge.jpg', edgesFinal * 255)
    cv2.imwrite('weak_edge.jpg', cut_uncanny * 255)


