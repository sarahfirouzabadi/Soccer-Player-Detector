import cv2
import numpy as np

video = cv2.VideoCapture('dataset/cam1_second_half_start.H264')  # reading video
success, frame = video.read()

pts = [[41, 209], [142, 167], [867, 777], [876, 776], [697, 255], [691, 255], [662, 161], [642, 111], [1135, 116],
       [1226, 88], [17, 151]]
pts = np.array(pts, dtype=np.float32)
dst = [[164, 273], [165, 150], [522, 695], [527, 695], [527, 446], [522, 446], [527, 255], [527, 4], [884, 149],
       [1040, 8], [9, 8]]
dst = np.array(dst, dtype=np.float32)
M, _ = cv2.findHomography(pts, dst)

backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=3000)

pitch = cv2.imread('dataset/2D_field.png', cv2.IMREAD_COLOR)

while success:
    success, frame = video.read()
    fgMask = backSub.apply(frame)
    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(video.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    w, h = frame.shape[0], frame.shape[1]
    warped = cv2.warpPerspective(frame, M, (1050, 700))

    kernel = np.ones((11, 11), np.uint8)
    closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    pitch_copy = np.copy(pitch)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        hull = cv2.convexHull(c)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(hull)
        x, y, w, h = cv2.boundingRect(c)

        pts = np.array([[x + w / 2, y + h / 2]], np.float32).reshape(-1, 1, 2).astype(np.float32)
        pt = cv2.perspectiveTransform(pts, M)

        aspectRatio = boxW / float(boxH)
        hullArea = cv2.contourArea(hull) / (closing.shape[0] * closing.shape[1])
        keep_area = hullArea > 0.0001

        keepAspectRatio = aspectRatio < 0.9 and keep_area
        if keepAspectRatio:
            pt = np.squeeze(pt)
            cv2.circle(pitch_copy, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 255), thickness=-1)
            pass

    cv2.imshow('Bird-eye', warped)
    cv2.imshow('Perspective pitch', pitch_copy)
    key = cv2.waitKey(4)
    if key == ord('q'):
        break
