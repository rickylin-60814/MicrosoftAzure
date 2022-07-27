from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


ENDPOINT = "輸入端點位置"
prediction_key = "輸入預測金鑰匙"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key}) # 設定預測憑證
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials) # 選擇預測器和設定端點跟憑證

publish_iteration_name = "選擇要預測的模型編號"
projectId = "輸入項目id"

x = 0 #次數

# with open用2進制(rb)讀取檔案
with open("3.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        projectId, publish_iteration_name, image_contents.read())#調用模型

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
            ": {0:.2f}%".format(prediction.probability * 100))#format 格式化，": {0:.2f}%" 保留小數後兩位
        if x == 0:
            image = Image.open("3.jpg")#開啟圖片
            plt.imshow(image)
            plt.axis("off")
            _ = plt.title(prediction.tag_name +
                ": {0:.2f}%".format(prediction.probability * 100), size="x-large", y=-0.1)
            #plt.show()
            plt.savefig("filename3.jpg")
            x=x+1