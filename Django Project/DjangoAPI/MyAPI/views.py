from django.shortcuts import render, redirect
from MyAPI.models import MyFile
from django.conf import settings
import boto3
import cv2
from rest_framework.serializers import ModelSerializer
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, renderer_classes, parser_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import MultiPartParser, FormParser

def ObjectDetection(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
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
    cv2.imwrite(imagePath, MyImage)


def CelebritiesDetection(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
    response = Service.recognize_celebrities(Image = {"Bytes":image})
    for objects in response["CelebrityFaces"]:
        CelName = objects["Name"]
        Face = objects["Face"]
        box = Face["BoundingBox"]
        x = int(imgW * box["Left"])
        y = int(imgH * box["Top"])
        w = int(imgW * box["Width"])
        h = int(imgH * box["Height"])
        MyImage = cv2.rectangle(MyImage,(x,y),(x+w, y+h), (0,255,0), 2)
        MyImage = cv2.putText(MyImage,CelName, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90,0,255), 2)
    cv2.imwrite(imagePath, MyImage)

def facialAnalysis(imagePath):
    session = boto3.Session(profile_name="default")
    Service = session.client("rekognition")
    image = open(imagePath,"rb").read()
    imgH, imgW = cv2.imread(imagePath).shape[:2]
    MyImage = cv2.imread(imagePath)
    response = Service.detect_faces(Image = {"Bytes":image}, Attributes=['ALL'])
    for faceDetail in response['FaceDetails']:
        Face = faceDetail
        box = Face["BoundingBox"]
        x = int(imgW * box["Left"])
        y = int(imgH * box["Top"])
        w = int(imgW * box["Width"])
        h = int(imgH * box["Height"])
        MyImage = cv2.rectangle(MyImage, (x-30, y-65), (x + w, y + h), (0, 255, 0), 2)
        age_range = faceDetail['AgeRange']
        age_limit = "Age:" + str(age_range['Low']) + "-" + str(age_range['High'])
        gender = faceDetail['Gender']
        gender = str(gender['Value'])
        MyImage = cv2.putText(MyImage, gender, (x-30, y-65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
        MyImage = cv2.putText(MyImage, age_limit, (x-120, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
        for emotion in faceDetail['Emotions']:
            percentage = str(emotion['Confidence'])
            percentage = percentage[:2] + "%"
            emote = (str(emotion['Type']) + "-" + percentage)
            MyImage = cv2.putText(MyImage, emote, (x - 120, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 0, 255), 2)
            break
    cv2.imwrite(imagePath, MyImage)




class FileSer(ModelSerializer):
    class Meta:
        model = MyFile
        fields = "__all__"


@api_view(["GET", "POST"])
@renderer_classes([JSONRenderer])
@parser_classes([MultiPartParser , FormParser])
def home(request):
    print("User request method is", request.method)
    if request.method == "POST" :

        service = request.POST["service"]
        Ser = FileSer(data = request.data)
        if Ser.is_valid():
            Ser.save()

            LastFile = MyFile.objects.get(id = Ser.data["id"])
            print(LastFile.image.url)
        print(Ser)

        path = str(settings.MEDIA_ROOT) + "/" + LastFile.image.name
        if service == "Celebrity Detection":
            CelebritiesDetection(path)
        if service == "Face Analysis":
            facialAnalysis(path)
        if service == "Object Detection":
            ObjectDetection(path)
        url = "http://127.0.0.1:8000" + LastFile.image.url
        print(path,url)
        Msg = {"Url":url}
        return Response(data = Msg, status=status.HTTP_200_OK)

    return render(request, "index.html")
