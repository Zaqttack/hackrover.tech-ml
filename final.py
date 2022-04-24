# from rover_cnn import read_model_history, MODEL_PATH, HISTORY_PATH
from rover_cnn import RoverCNN
from rover_tracer import DrawingCamera

import cv2


def main():
    # model, history = read_model_history(model_path=MODEL_PATH,
    #                                     history_path=HISTORY_PATH)
    rover_cnn = RoverCNN()
    drawing_camera = DrawingCamera()

    
    drawing_camera.take_pic_and_trace()

    example_image = rover_cnn.x_train[0]
    cv2.imshow("HackUNT", example_image)
    cv2.waitKey(0)
    print("The predicted hand symbol is", rover_cnn.predict(example_image))
    rover_cnn.show_graphs()

    
    # image_capture = image_capture.reshape(1920,1080,1)
    # image_capture = image_capture[1920//2-14:1920//2+13, 1080//2-14:1080//2+13]
    # print(image_capture.shape)


    # cv2.imshow("balls", image_capture)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    # del camera
    return

    print(rover_cnn.predict(image_capture))

    return

    while True:
        success, image_capture = camera.read()
        
        if not success:
            continue


if __name__ == "__main__":
    main()
