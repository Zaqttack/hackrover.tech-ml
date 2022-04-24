#!/usr/bin/python
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

def paint_joints(image_capture, hands_model, drawing_model, output="show"):
    # cap = cv2.imread('photo.jpg', cv2.IMREAD_UNCHANGED)

    #You can pass `max_num_hands` argument here as well if you want to detect more that one hand
    with hands_model.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            
            image = cv2.cvtColor(image_capture, cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False
            
            results = hands.process(image)
            
            image.flags.writeable = True
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
           
            # Rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    drawing_model.draw_landmarks(image, hand,
                                              hands_model.HAND_CONNECTIONS)

    image_name = "{}.jpg".format(uuid.uuid1())
    if output == "write":
        output_name = os.path.join("output-images", image_name)
        cv2.imwrite(output_name, image)
    elif output == "show":
        cv2.imshow(image_name, image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


class DrawingCamera:
    def __init__(self):
        self.drawing_model = mp.solutions.drawing_utils
        self.hands_model = mp.solutions.hands

    def take_pic_and_trace(self, output="show"):
        camera = cv2.VideoCapture(0)
        success, image_capture = camera.read()

        if not success:
            return

        self.paint_joints(image_capture, output=output)

        del(camera)

    def paint_joints(self, image_capture, output="show"):
        paint_joints(image_capture, self.hands_model, self.drawing_model, output="show")



if __name__ == "__main__":
    take_picture_and_paint()
