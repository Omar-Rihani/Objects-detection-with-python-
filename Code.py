import cv2

# Function to perform object detection from an image file
def detect_objects_from_image():
    # Load class names
    class_names = []
    class_file_path = 'coco.names'

    with open(class_file_path, 'rt') as file:
        class_names = file.read().rstrip('\n').split('\n')

    # Load the model configuration and weights
    config_file_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_file_path = 'frozen_inference_graph.pb'

    # Initialize the DNN model
    dnn_model = cv2.dnn_DetectionModel(weights_file_path, config_file_path)
    dnn_model.setInputSize(320, 230)  # Set input size
    dnn_model.setInputScale(1.0 / 127.5)  # Scale input
    dnn_model.setInputMean((127.5, 127.5, 127.5))  # Set mean values
    dnn_model.setInputSwapRB(True)  # Swap the B and R channels

    # Perform detection
    class_ids, confidences, bounding_boxes = dnn_model.detect(image, confThreshold=0.5)
    print(class_ids, bounding_boxes)

    # Draw bounding boxes and labels
    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bounding_boxes):
        cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
        cv2.putText(image, class_names[class_id - 1], (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

    cv2.imshow('Output', image)  # Display the output image
    cv2.waitKey(0)  # Wait for a key press

# Function to perform object detection from camera input
def detect_objects_from_camera():
    camera = cv2.VideoCapture(0)  # Open camera

    camera.set(3, 740)  # Set camera width
    camera.set(4, 580)  # Set camera height

    # Load class names
    class_names = []
    class_file_path = 'coco.names'

    with open(class_file_path, 'rt') as file:
        class_names = file.read().rstrip('\n').split('\n')

    # Load the model configuration and weights
    config_file_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_file_path = 'frozen_inference_graph.pb'

    # Initialize the DNN model
    dnn_model = cv2.dnn_DetectionModel(weights_file_path, config_file_path)
    dnn_model.setInputSize(320, 230)  # Set input size
    dnn_model.setInputScale(1.0 / 127.5)  # Scale input
    dnn_model.setInputMean((127.5, 127.5, 127.5))  # Set mean values
    dnn_model.setInputSwapRB(True)  # Swap the B and R channels

    while True:
        success, frame = camera.read()  # Capture frame from camera
        class_ids, confidences, bounding_boxes = dnn_model.detect(frame, confThreshold=0.5)
        print(class_ids, bounding_boxes)

        if len(class_ids) != 0:
            # Draw bounding boxes and labels
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bounding_boxes):
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, class_names[class_id - 1], (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow('Output', frame)  # Display the output frame
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('y'):  # Press 'y' to quit
            break

    camera.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Call the desired function: detect_objects_from_image() for image or detect_objects_from_camera() for video
# detect_objects_from_image()
detect_objects_from_camera()
