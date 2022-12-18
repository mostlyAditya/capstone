from pymongo import MongoClient
try:
    conn = MongoClient()
    print("Connected successfully!!!")
except:  
    print("Could not connect to MongoDB")
  
# database
db = conn.database
# Created or Switched to collection names: my_gfg_collection
collection = db.model_data

import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
def get_iou(bb1, bb2):
    
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    #print(iou)
    return iou
def try_overlapping(chairs,persons):
    if len(persons) == 0:
        return [0]*len(chairs)
    occupied = [0]*len(chairs)
    for i,row_chairs in chairs.iterrows():
        x1,x2,y1,y2 = row_chairs['xmin'] , row_chairs['xmax'] , row_chairs['ymin'], row_chairs['ymax']
        bb1 = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}
        for j,row in persons.iterrows():
            x1,x2,y1,y2 = row['xmin'] , row['xmax'] , row['ymin'], row['ymax']
            bb2 = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}
            percentage = get_iou(bb1,bb2)
            if percentage > 0.3:
                occupied[i] = 1
    return occupied
#Model
def main():
    model = torch.hub.load('mostlyAditya/yolov5',  'yolov5s',device = 'cpu')  # local repo
    # Images
    img = 'yolov5/data/videos/seating_data.mp4'
    img = cv2.VideoCapture(img)
    found = False
    chairs = []
    second = 0
    width = int(img.get(3))
    height = int(img.get(4))
    #vid = cv2.VideoWriter('save_video.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(width, height))

    while img.isOpened():
        ret, frame = img.read()
        for i in range(23):
            img.read()
        # Make detections 
        results = None
        try:
            results = model(frame)
        except AttributeError:
            print("Completed")
            break
        #results.save()
        #vid.write(np.squeeze(results.render()))
        cv2.imshow('YOLO', np.squeeze(results.render()))
        #results.print()  
        #results.show()  # or .show()
        #results = results.xyxy[0]  # img1 predictions (tensor)
        boxes = results.pandas().xyxy[0]
        if not found:
            chairs = boxes[boxes["class"] == 56].sort_values('xmin')
            found = True
        persons = boxes[boxes["class"] == 0].sort_values('xmin')
        #chairs.to_csv('file.csv', mode='a', index=False, header=False)
        #persons.to_csv('file.csv', mode='a', index=False, header=False)
        
        seats = try_overlapping(chairs,persons)
        occupied = {
                    'second':second,
                    'seats':seats
                    }
        collection.insert_one(occupied)
        print(occupied)
        print(second)
        second += 1
        #print(boxes)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    img.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

#img = cv2.imread('yolov5/data/videos/VID_20220826_133818.mp4')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
# Inference
'''
results = model(img, size=328)  # includes NMS

# Results
results.print()  
#results.show()  # or .show()

#results = results.xyxy[0]  # img1 predictions (tensor)
boxes = results.pandas().xyxy[0]
print(boxes)
'''