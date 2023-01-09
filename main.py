import cv2 as cv

classNames=[]
classFile="coco.names"
with open(classFile,"rt") as file:
    classNames=file.read().rstrip("\n").split("\n")

weightPath="frozen_inference_graph.pb"
configPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

threshold_value=0.5

net=cv.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5,127.5))
net.setInputSwapRB(True)

def rescaleFrame(frame,scale_x=1.2,scale_y=1.2):
    breadth=int(frame.shape[1]*scale_x)
    length=int(frame.shape[0]*scale_y)
    dimensions=(breadth,length)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

capture=cv.VideoCapture(0)

while True:
    isTrue,frame=capture.read()
    changed_frame=rescaleFrame(frame)
    classIds,confidence,bbox=net.detect(changed_frame,confThreshold=threshold_value)
    print(classIds,bbox)

    if len(classIds)!=0:
        for classId,conf,box in zip(classIds.flatten(),confidence.flatten(),bbox):
            cv.rectangle(changed_frame,box,color=(0,255,0),thickness=2)
            cv.putText(changed_frame,classNames[classId-1].upper(),(box[0]+10,box[1]+20),fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,255,0),thickness=1)
            cv.putText(changed_frame,str(round(conf*100,2)),(box[0]+200,box[1]+20),fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,255,0),thickness=1)
    
    cv.imshow("Camera Capture",changed_frame)
    if cv.waitKey(20) & 0xFF==ord("d"):
        break