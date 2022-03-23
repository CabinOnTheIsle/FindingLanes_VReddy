import cv2
import numpy as np

picture = r'[]/Resources/solidYellowLeft.jpg'     # Enter the file path of the Image here
video = r'[]/Resources/solidWhiteRight.mp4'       # Enter the file path of the Video here

img = cv2.imread(picture)
lane_image = np.copy(img)


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
    canny_img = cv2.Canny(blur, 100, 200)
    dilate = cv2.dilate(canny_img, kernel=None, iterations=1)
    return dilate


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)  # Blank Matrix like image
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:  # Since Average lines is a one dimension array we can iterate by 4 vars
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    img = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return img


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        param = np.polyfit((x1, x2), (y1, y2), 1)
        slope = param[0]
        intercept = param[1]
        if slope < 0:  # Lines on the left are negative due to y = mx+c (Gradiant is negative)
            left_fit.append((slope, intercept))  # sorts out all the lines that are left fit
        else:
            right_fit.append((slope, intercept))  # sorts out all the lines that are right fit
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coords(image, left_fit_avg)
    right_line = make_coords(image, right_fit_avg)
    return np.array([left_line, right_line])


def make_coords(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.00001, 1
    y1 = image.shape[0]
    y2 = int(y1 * (3.5 / 5))  # Start from bottom and go until 3/5ths of the way
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

## NOTE: Here the desired program can be commented out. Unncomment the section to receive the output.

## SECTION 1 - IMAGE PROCESSING
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
height = img.shape[0]
width = img.shape[1]
region_of_interest_vertices = [(0, height), (width / 2, height / 1.794), (width, height)]
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(cropped_image,
                        rho=2, theta=np.pi / 180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40, maxLineGap=25)
avg_lines = average_slope_intercept(lane_image, lines)
line_image = draw_lines(lane_image, avg_lines)

cv2.imshow("Lane Detected Image", line_image)
cv2.waitKey(0)
'''

## SECTION 2 - VIDEO PROCESSING
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
height = img.shape[0]
width = img.shape[1]
cap = cv2.VideoCapture(video)
process_save = cv2.VideoWriter('processedVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

while cap.isOpened():
    ret, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]
    region_of_interest_vertices = [(0, height), (width / 2, height / 1.794), (width, height)]
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2, theta=np.pi / 180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40, maxLineGap=25)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = draw_lines(frame, averaged_lines)

    process_save.write(line_image)

    cv2.imshow("result", line_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

cap.release()
process_save.release()
cv2.destroyAllWindows()

