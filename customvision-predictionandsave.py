from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Replace with valid values
ENDPOINT = "YOUR ENDPOINT"
prediction_key = "YOUR prediction_key"

#Authenticate the client
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

publish_iteration_name = "YOUR  Iteration"
projectId = "YOUR projectId"

x=0

# Open the sample image and get back the prediction results.
with open("3.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        projectId, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
            ": {0:.2f}%".format(prediction.probability * 100))
        if x == 0:
            image = Image.open("3.jpg")
            plt.imshow(image)
            plt.axis("off")
            _ = plt.title(prediction.tag_name +
                ": {0:.2f}%".format(prediction.probability * 100), size="x-large", y=-0.1)
            #plt.show()
            plt.savefig("filename3.jpg")
            x=x+1
