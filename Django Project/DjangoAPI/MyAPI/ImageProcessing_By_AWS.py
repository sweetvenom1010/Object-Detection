import boto3
import cv2




def ObjectDetection(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
    print(imgH,imgW)

    response = Service.detect_labels(Image = {"Bytes":image})
    for objects in response["Labels"]:
        if objects['Instances']:
            objectName = objects["Name"]
            for boxs in objects["Instances"]:
                box = boxs["BoundingBox"]
                x = int(imgW * box["Left"])
                y = int(imgH * box["Top"])
                w = int(imgW * box["Width"])
                h = int(imgH * box["Height"])
                print(x,y,w,h)
                MyImage = cv2.rectangle(MyImage,(x,y),(x+w, y+h), (0,255,0), 2)
                MyImage = cv2.putText(MyImage,objectName, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,0,255), 2)
    cv2.imwrite("detected.jpg", MyImage)


def CelebritiesDetection(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
    print(imgH,imgW)

    response = Service.recognize_celebrities(Image = {"Bytes":image})
    for objects in response["CelebrityFaces"]:
        CelName = objects["Name"]
        Face = objects["Face"]
        box = Face["BoundingBox"]
        x = int(imgW * box["Left"])
        y = int(imgH * box["Top"])
        w = int(imgW * box["Width"])
        h = int(imgH * box["Height"])
        print(x,y,w,h)
        MyImage = cv2.rectangle(MyImage,(x,y),(x+w, y+h), (0,255,0), 2)
        MyImage = cv2.putText(MyImage,CelName, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,0,255), 2)
    cv2.imwrite("detected.jpg", MyImage)



image = "A:\Django Project\DjangoAPI\MyUploads\image5.jpg"
CelebritiesDetection(image, "Object Detection")
