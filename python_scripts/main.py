import cv2
import numpy as np
import tensorflow as tf



class_id_to_name = {
    1: 'cardboard',
    2: 'glass',
    3: 'metal',
    4: 'paper',
    5: 'plastic',
}

# Path to the .tflite model file
model_path = '../assets/models/detect.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess frame
def preprocess_frame(frame):
    # Resize frame to model input size
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    # Convert frame to float32 and normalize (if required by your model)
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0
    return input_data

def postprocess_results(frame, output_data, threshold=0.5):
    # Retrieve outputs for the number of detections, detection boxes, detection classes, and detection scores
    num_detections = int(output_data[2][0])  # 'number of detections' is typically the 3rd output
    detection_boxes = output_data[1][0][:num_detections]  # 'location'
    detection_classes = output_data[3][0][:num_detections]  # 'category'
    detection_scores = output_data[0][0][:num_detections]  # 'score'

    # Scale box coordinates to frame dimensions.
    height, width, _ = frame.shape
    
    for i in range(num_detections):
        # Skip detections with a score below the threshold.
        if detection_scores[i] < threshold:
            # print(f'Skipping detection {i} with score {detection_scores[i]:.2f}')
            continue

        # Get bounding box coordinates.
        box = detection_boxes[i]
        
        print("Boxes: ", box)   
        ymin, xmin, ymax, xmax = box

        # Convert to absolute coordinates.
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)

        # Draw the bounding box on the frame.
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)

        # Get class ID and score for display.
        class_id = int(detection_classes[i])
        score = detection_scores[i]

        # Map class ID to class name
        class_name = class_id_to_name.get(class_id, "Unknown")

        print(f'Detected class: {class_name} with score {score:.2f}')

        # Draw label and score below the bounding box.
        label = f'{class_name}: {score:.2f}'
        cv2.putText(frame, label, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



# Open a handle to the default webcam
path = "./inputs/output_video.mp4"
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Before your loop, get all output details
    output_details = interpreter.get_output_details()

    # Then, in your loop after invoking the interpreter
    results = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]

    # Now call the postprocess_results function with the frame and results
    # Postprocess the results
    postprocess_results(frame, results, threshold=0.7)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame, )

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()