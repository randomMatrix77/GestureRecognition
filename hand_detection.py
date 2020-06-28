import cv2
import mouse
import numpy as np
import vgg_model
import time

low1 = np.array([5, 59, 202])
high1 = np.array([11, 104, 255])

# low2 = np.array([17, 9, 240])
# high2 = np.array([25, 33, 255])

labels = ['Palm', 'Fist', 'Other']

# buffer = deque([np.zeros(3) for i in range(3)])
# buffer_x = deque([0, 0, 0, 0, 0])
# buffer_y = deque([0, 0, 0, 0, 0])

prev_cX = 0
prev_cY = 0

def move_mouse(locX, locY):
    global prev_cX, prev_cY

    if locX != -1 and locY != -1:

        mouse.move(int((locX - prev_cX)*4), -int((locY - prev_cY) / 50), absolute = False)
        # int((locY - prev_cY) / 20)

        prev_cX = locX
        prev_cY = locY

def mouse_press(pred):

    if pred == 2:
        mouse.click(mouse.RIGHT)
    if pred == 1:
        mouse.click(mouse.LEFT)

def gesture_detector(img):

    cropped = cv2.resize(img, (224, 224))
    cropped = np.expand_dims(cropped, 0)
    pred = vgg_model.runtime_model(cropped)

    # buffer.append(np.eye(3)[pred])
    # buffer.popleft()
    #
    # pred = np.argmax(np.sum(np.array(buffer), 0))

    return pred

def draw_contours(detected_contours, original_image):

    for c in detected_contours:
        if len(c) > 300:

            cv2.polylines(original_image, [c], True, (255, 255, 255))
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.line(original_image, (cX - 10, cY), (cX + 10, cY), (255, 255, 255), 5)
                cv2.line(original_image, (cX, cY - 10), (cX, cY + 10), (255, 255, 255), 5)
                return cX, cY
            else:
                return -1, -1

def contour_detect(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contour1 = cv2.inRange(hsv, low1, high1)
    # contour2 = cv2.inRange(hsv, low2, high2)

    # contour = cv2.bitwise_or(contour1, contour2)
    # contour = cv2.morphologyEx(contour1, cv2.MORPH_CLOSE, kernel)
    im, detected_contours, heir = cv2.findContours(contour1[5:360, 5:375], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        cX, cY = draw_contours(detected_contours, contour1)
        cv2.putText(contour1, 'Center:' + str(cX) + ' ' + str(cY), (200, 450), cv2.FONT_ITALIC, 1,
                    (255, 255, 255))
        move_mouse(cX, cY)

    except:
        pass

    cv2.rectangle(contour1, (0, 0), (365, 380), (255, 255, 255), 2)

    return contour1

cap = cv2.VideoCapture(0)

print('Starting in...')
for i in reversed(range(15)):

    print(i, end=' ')
    time.sleep(1)
print('\n')

while(True):
    ret, frame = cap.read()
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw = bw[5:355, 5:355]
    # bw = cv2.flip(bw, 1)
    contour = contour_detect(frame)

    try:
        pred = gesture_detector(bw/255)
        cv2.putText(bw, 'Detected Gesture : ' + str(labels[pred]), (5, 300), cv2.FONT_ITALIC, 0.8,
                    (255, 255, 255))
        # print('Detected Gesture : {}'.format(labels[pred]))
        mouse_press(pred)

    except:
        pass

    cv2.imshow("gesture", bw)
    cv2.imshow("contour", contour)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
