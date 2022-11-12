import cv2
import numpy as np

vid_cam0 = cv2.VideoCapture('dataset/Cam0.h264')
vid_cam1 = cv2.VideoCapture('dataset/Cam1.h264')
vid_cam2 = cv2.VideoCapture('dataset/Cam2.h264')
success_cam0, frame_cam0 = vid_cam0.read()
success_cam1, frame_cam1 = vid_cam1.read()
success_cam2, frame_cam2 = vid_cam2.read()

pts_cam0 = [[1280, 366], [1279, 299], [1037, 959], [476, 297], [901, 192], [51, 303]]
pts_cam0 = np.array(pts_cam0, dtype=np.float32)
dst_cam0 = [[470, 421], [459, 289], [487, 695], [165, 553], [165, 149], [4, 695]]
dst_cam0 = np.array(dst_cam0, dtype=np.float32)
M_cam0, _0 = cv2.findHomography(pts_cam0, dst_cam0)

pts_cam1 = [[41, 209], [142, 167], [867, 777], [876, 776], [697, 255], [691, 255], [662, 161], [642, 111], [1135, 116],
            [1226, 88], [17, 151]]
pts_cam1 = np.array(pts_cam1, dtype=np.float32)
dst_cam1 = [[164, 273], [165, 150], [522, 695], [527, 695], [527, 446], [522, 446], [527, 255], [527, 4], [884, 149],
            [1040, 8], [9, 8]]
dst_cam1 = np.array(dst_cam1, dtype=np.float32)
M_cam1, _1 = cv2.findHomography(pts_cam1, dst_cam1)

pts_cam2 = [[404, 895], [397, 902], [31, 424], [24, 426], [319, 959], [673, 221], [453, 210], [1251, 272], [879, 271],
            [803, 209]]
pts_cam2 = np.array(pts_cam2, dtype=np.float32)
dst_cam2 = [[527, 695], [522, 695], [527, 446], [522, 446], [469, 695], [941, 349], [884, 149], [1045, 695], [884, 553],
            [1049, 386]]
dst_cam2 = np.array(dst_cam2, dtype=np.float32)
M_cam2, _2 = cv2.findHomography(pts_cam2, dst_cam2)

backSub_cam0 = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=4000)
backSub_cam1 = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=4000)
backSub_cam2 = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=4000)

pitch = cv2.imread('dataset/2D_field.png', cv2.IMREAD_COLOR)

count_team1 = 0
count_team2 = 0
count_referee = 0
while success_cam1:
    success_cam1, frame_cam1 = vid_cam1.read()
    success_cam2, frame_cam2 = vid_cam2.read()
    success_cam0, frame_cam0 = vid_cam0.read()
    # cv2.imwrite('frame0.png',frame_cam0);

    fgMask_cam0 = backSub_cam1.apply(frame_cam0)
    kernel = np.ones((5, 5), np.uint8)
    closing_cam0 = cv2.morphologyEx(fgMask_cam0, cv2.MORPH_CLOSE, kernel)
    contours_cam0 = cv2.findContours(closing_cam0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    fgMask_cam1 = backSub_cam1.apply(frame_cam1)
    closing_cam1 = cv2.morphologyEx(fgMask_cam1, cv2.MORPH_CLOSE, kernel)
    contours_cam1 = cv2.findContours(closing_cam1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    fgMask_cam2 = backSub_cam2.apply(frame_cam2)
    closing_cam2 = cv2.morphologyEx(fgMask_cam2, cv2.MORPH_CLOSE, kernel)
    contours_cam2 = cv2.findContours(closing_cam2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    w, h = frame_cam1.shape[0], frame_cam1.shape[1]
    warped_cam0 = cv2.warpPerspective(frame_cam0, M_cam0, (1050, 700));
    warped_cam1 = cv2.warpPerspective(frame_cam1, M_cam1, (1050, 700));

    # a = np.where(warped_cam1 == 0, warped_cam2, warped_cam1);
    # cv2.imshow('p',warped_cam1);
    # cv2.waitKey();

    pitch_copy = np.copy(pitch)

    prev_points = []

    for c in contours_cam1:
        hull = cv2.convexHull(c)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(hull)
        x, y, w, h = cv2.boundingRect(c)

        pts = np.array([[x + w / 2, y + h / 2]], np.float32).reshape(-1, 1, 2).astype(np.float32)
        pt = cv2.perspectiveTransform(pts, M_cam1)
        prev_points.append([pt[0, 0, 0], pt[0, 0, 1]])

        aspectRatio = boxW / float(boxH)
        hullArea = cv2.contourArea(hull) / (closing_cam1.shape[0] * closing_cam1.shape[1])
        keep_area = hullArea > 0.00005

        keepAspectRatio = aspectRatio < 1.0
        if keepAspectRatio and keep_area:
            pt = np.squeeze(pt)
            cv2.circle(pitch_copy, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 255), thickness=-1)
            # cv2.rectangle(frame_cam1,(int(x),int(y)),(int(x+w),int(y+h)),color=(255,255,255),thickness=1);

            # cv2.waitKey();

            # crop from frame_cam1
            # c = frame_cam1[y:y+h,x:x+w];
            # #cv2.rectangle(frame_cam1,(x,y),(x+w,y+h),(255,255,255),2);
            # if(c.shape[0] !=0 and c.shape[1] !=0):
            #     cv2.imshow('player',c);
            #     cv2.imshow('pitch',frame_cam1);
            #     k = cv2.waitKey();
            #     if k == 49:
            #         cv2.imwrite(f'dataset/1/{count_team1}.png',c);
            #         count_team1+=1;
            #     elif k == 50:
            #         cv2.imwrite(f'dataset/2/{count_team2}.png',c);
            #         count_team2+=1;
            #     elif k == 51:
            #         cv2.imwrite(f'dataset/3/{count_referee}.png',c);
            #         count_referee+=1;
            #     cv2.destroyWindow('player');
            pass

    for c in contours_cam2:
        hull = cv2.convexHull(c)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(hull)
        x, y, w, h = cv2.boundingRect(c)

        pts = np.array([[x + w / 2, y + h / 2]], np.float32).reshape(-1, 1, 2).astype(np.float32)
        pt = cv2.perspectiveTransform(pts, M_cam2)

        aspectRatio = boxW / float(boxH)
        hullArea = cv2.contourArea(hull) / (closing_cam1.shape[0] * closing_cam1.shape[1])
        keep_area = hullArea > 0.00007

        keepAspectRatio = aspectRatio < 1.0
        if keepAspectRatio and keep_area:
            pt = np.squeeze(pt)
            minDist = 1000000000
            for p in prev_points:
                d = np.squeeze(np.abs(pt[0] - p[0]) + np.abs(pt[1] - p[1]))
                if d < minDist:
                    minDist = d
            if minDist > 50:
                cv2.circle(pitch_copy, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 255), thickness=-1)
                # cv2.rectangle(frame_cam2,(int(x),int(y)),(int(x+w),int(y+h)),color=(255,255,255),thickness=1);
                prev_points.append(pt)

            key = cv2.waitKey(5)

            # #  crop frame from frame_cam0
            # c = frame_cam2[y:y + h, x:x + w]
            # cv2.rectangle(frame_cam2, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # if c.shape[0] != 0 and c.shape[1] != 0:
            #     cv2.imshow('player', c)
            #     cv2.imshow('pitch', frame_cam2)
            #     k = cv2.waitKey()
            #     if k == 49:
            #         cv2.imwrite(f'dataset/1/{count_team1}.png', c)
            #         count_team1 += 1
            #     elif k == 50:
            #         cv2.imwrite(f'dataset/2/{count_team2}.png', c)
            #         count_team2 += 1
            #     elif k == 51:
            #         cv2.imwrite(f'dataset/3/{count_referee}.png', c)
            #         count_referee += 1
            #     cv2.destroyWindow('player')
            pass

    for c in contours_cam0:
        hull = cv2.convexHull(c)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(hull)
        x, y, w, h = cv2.boundingRect(c)

        pts = np.array([[x + w / 2, y + h / 2]], np.float32).reshape(-1, 1, 2).astype(np.float32)
        pt = cv2.perspectiveTransform(pts, M_cam0)

        aspectRatio = boxW / float(boxH)
        hullArea = cv2.contourArea(hull) / (closing_cam1.shape[0] * closing_cam1.shape[1])
        keep_area = hullArea > 0.00005

        keepAspectRatio = aspectRatio < 1.0
        if keepAspectRatio and keep_area:
            pt = np.squeeze(pt)
            minDist = 1000000000
            for p in prev_points:
                d = np.squeeze(np.abs(pt[0] - p[0]) + np.abs(pt[1] - p[1]))
                if d < minDist:
                    minDist = d
            if minDist > 50:
                cv2.circle(pitch_copy, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 255), thickness=-1)
                cv2.rectangle(frame_cam0, (int(x), int(y)), (int(x + w), int(y + h)), color=(255, 255, 255),
                              thickness=1)
                prev_points.append(pt)

            # cv2.waitKey();

            # # crop from frame
            # c = frame_cam0[y:y+h,x:x+w];
            # #cv2.rectangle(frame_cam0,(x,y),(x+w,y+h),(255,255,255),2);
            # if(c.shape[0] !=0 and c.shape[1] !=0):
            #     cv2.imshow('player',c);
            #     cv2.imshow('pitch',frame_cam0);
            #     k = cv2.waitKey();
            #     if k == 49:
            #         cv2.imwrite(f'dataset/1/{count_team1}.png',c);
            #         count_team1+=1;
            #     elif k == 50:
            #         cv2.imwrite(f'dataset/2/{count_team2}.png',c);
            #         count_team2+=1;
            #     elif k == 51:
            #         cv2.imwrite(f'dataset/3/{count_referee}.png',c);
            #         count_referee+=1;
            #     cv2.destroyWindow('player');
            pass

    cv2.imshow('Bird-eye', frame_cam0)
    cv2.imshow('pitch', pitch_copy)
    key = cv2.waitKey(20)
    if key == "q":
        break
