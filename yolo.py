import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class VehicleDetection {
    private MultiLayerNetwork model;

    public VehicleDetection(String modelPath) throws IOException {
        // Load the trained model from file
        model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
    }

    public boolean detectVehicle(File imageFile) throws IOException {
        // Load the image as a BufferedImage
        BufferedImage image = ImageIO.read(imageFile);

        // Preprocess the image
        INDArray input = preprocessImage(image);

        // Make a prediction using the model
        INDArray output = model.output(input);

        // Extract the vehicle probability from the output
        double vehicleProb = output.getDouble(0);

        // Return true if the probability is above a threshold
        return vehicleProb > 0.5;
    }

    private INDArray preprocessImage(BufferedImage image) {
        // Resize the image to the expected input size of the model
        BufferedImage resizedImage = resizeImage(image, 416, 416);

        // Convert the resized image to a 3D array of RGB values
        int[] pixels = resizedImage.getRGB(0, 0, resizedImage.getWidth(), resizedImage.getHeight(), null, 0, resizedImage.getWidth());
        INDArray imageArray = Nd4j.create(pixels, new int[] {1, resizedImage.getHeight(), resizedImage.getWidth(), 3}, 'c');

        // Normalize the pixel values
        imageArray.divi(255.0);

        return imageArray;
    }

    private BufferedImage resizeImage(BufferedImage image, int newWidth, int newHeight) {
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        resizedImage.createGraphics().drawImage(image.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH), 0, 0, null);
        return resizedImage;
    }
}

import cv2

# Load pre-trained YOLOv3 model
model_weights = "yolov3.weights"
model_config = "yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Set threshold and confidence level
conf_threshold = 0.5
nms_threshold = 0.4

# Load input image
img = cv2.imread("road_camera.jpg")

# Get image dimensions and create 4D blob
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input and output nodes for the model
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Process each output layer
boxes = []
confidences = []
class_ids = []
for output in layer_outputs:
    for detection in output:
        # Extract class ID, confidence level, and bounding box coordinates
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2
            # Add bounding box coordinates, confidence level, and class ID to lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to eliminate redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes around detected vehicles
for i in indices:
    i = i[0]
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display output image
cv2.imshow("Vehicle Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
