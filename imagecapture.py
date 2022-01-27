import cv2
import os

camera = cv2.VideoCapture(6) #6 is the port for my external camera
# Show error if camera doesnt show up
if not camera.isOpened():
    raise Exception("Could not open video device")
# Set picture Frame. High quality is 1280 by 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set up window
cv2.namedWindow('Color')
img_counter = 0 #counter for # of images we want to save
file_path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/Photos' # folder to save photos to

# Read in picture to a frame
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error. Unable to capture Frame")
        break
    # Show the camera view on this frame
    cv2.imshow("color", frame)
    key = cv2.waitKey(1)
    if key == ord('p'):
        #Capture image
        #Create name for new image
        img_name = "my_photo_{}.jpg".format(img_counter)
        # store image in correct folder
        cv2.imwrite(os.path.join(file_path, img_name), frame)
        #increment counter for new images
        img_counter += 1
    elif key == ord('q'):
        # time to quit
        print("closing window")
        break
camera.release()
cv2.destroyAllWindows()