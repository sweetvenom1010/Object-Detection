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
    print(response)
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
        MyImage = cv2.putText(MyImage, CelName, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
    cv2.imwrite("detected.jpg", MyImage)



def facialAnalysis(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
    print(imgH,imgW)
    response = Service.detect_faces(Image = {"Bytes":image}, Attributes=['ALL'])
    print(response)
    for faceDetail in response['FaceDetails']:
        Face = faceDetail
        box = Face["BoundingBox"]
        x = int(imgW * box["Left"])
        y = int(imgH * box["Top"])
        w = int(imgW * box["Width"])
        h = int(imgH * box["Height"])
        print(x, y, w, h)
        MyImage = cv2.rectangle(MyImage, (x-30, y-65), (x + w, y + h), (0, 255, 0), 2)
        if faceDetail['Landmarks']:
            for landmarks in faceDetail['Landmarks']:
                print(str(landmarks['Type']) + "\n")

        age_range = faceDetail['AgeRange']
        print(" Age Range \nLOW    HIGH" )
        age_limit = "Age:" + str(age_range['Low']) + "-" + str(age_range['High'])
        print(str(age_range['Low']) + "\t\t" + str(age_range['High']))
        print('\n')

        gender = faceDetail['Gender']
        print("Gender:" + str(gender['Value']))
        gender = str(gender['Value'])
        print('\n')
        MyImage = cv2.putText(MyImage, gender, (x-30, y-65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
        MyImage = cv2.putText(MyImage, age_limit, (x-120, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)

        print('Emotions: \t Confidence\n')
        for emotion in faceDetail['Emotions']:
            print(str(emotion['Type']) + '\t\t' + str(emotion['Confidence']))
            percentage = str(emotion['Confidence'])
            percentage = percentage[:2] + "%"
            emote = (str(emotion['Type']) + "-" + percentage)
            MyImage = cv2.putText(MyImage, emote, (x - 120, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
            print('\n')
            break
    cv2.imwrite("detected.jpg", MyImage)

image = "A:\Django Project\DjangoAPI\MyUploads\image5.jpg"
facialAnalysis(image)
#CelebritiesDetection(image)