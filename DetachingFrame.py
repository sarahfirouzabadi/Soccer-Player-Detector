import cv2

video = cv2.VideoCapture('dataset/secondhalf-start.H264')  # reading video
success, frame = video.read()

cv2.imshow("frame", frame)
cv2.waitKey(0)
