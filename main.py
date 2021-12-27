import cv2
from matplotlib import pyplot as plt


def compute_fast_det(img, is_nms=True, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()  # FastFeatureDetector()

    #     # find and draw the keypoints
    if not is_nms:
        fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)

    kp = fast.detect(img, None)
    cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv2.drawKeypoints(gray, kp, gray)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def compute_fast_det1(filename, is_nms=True, thresh=10):
    img = cv2.imread(filename)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()  # FastFeatureDetector()

    # find and draw the keypoints
    if not is_nms:
        fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)

    kp = fast.detect(img, None)
    cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    return img


def main():
    compute_fast_det(cv2.imread("target1.jpeg"))


if __name__ == '__main__':
    main()
