import numpy as np
import os
import cv2
import cvzone
import statistics
import math
import argparse
import ImageIterator

from sort import*
import ultralytics
from ultralytics import YOLO
from collections import Counter


class YOLOv8_ObjectTracker:
    
    def __init__(self, model_file='yolov8n.pt', labels=None, classes=None, conf=0.25, iou=0.45, track_max_age=55, track_min_hits=15, track_iou_threshold=0.3):
        self.classes = classes
        self.conf = conf
        self.iou = iou

        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None
        
        # if no labels are provided then use default COCO names 
        if not labels:
            self.labels = self.model.names
        else:
            self.labels = labels
            
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
        
        
    def images_to_video(self, image_folder, save_path, size = (1000, 500)):
        img_array = []
        
        for image in os.listdir(image_folder):
            img = cv2.imread(os.path.join(image_folder, image))
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        # Save the video in specified path
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        return out

    def find_mode(self, data_list):
        data_count = dict(Counter(data_list))
        max_count = max(data_count.values())

        mode_value = [num for num, freq in data_count.items() if freq == max_count]

        return mode_value[0]

    def predict_img(self, img, verbose=True):
        results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=verbose)

        # Save the original image and the results for further analysis if needed
        self.orig_img = img
        self.results = results[0]

        # Return the detection results
        return results[0]
    
    def annotate_frames_find_direction(self, show_cls = True, show_conf = True):
        img = self.orig_img
        
        for box in self.results.boxes:
            textstring = ""

             # Extract object class and confidence score
            score = box.conf.item() * 100
            class_id = int(box.cls.item())

            x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
            
            if show_cls:
                textstring += f"{self.labels[class_id]}"
            if show_conf:
                textstring += f" {score:,.2f}%"
                
            # Draw bounding box, a centroid and label on the image
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            center_coordinates = ((x1 + x2)//2, (y1 + y2) // 2)

            img = cv2.circle(img, center_coordinates, 5, (0, 255, 0), -1)

        return img
        
    def track_images(self, folder_path, save_dir, verbose=False, **display_args):

        cap = ImageIterator.ImageIterator(folder_path)
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        tracker = Sort(max_age = self.track_max_age, min_hits= self.track_min_hits , 
                            iou_threshold = self.track_iou_threshold)
        
        frame_count = 0

        # Tracking implementation
        currentArray = np.empty((0, 5))
        ids = {}
        crowd_avg_direction = []
        for frame in cap:
            detections = np.empty((0, 5))

            # Run object detection on the frame and calculate FPS
            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results == None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())

                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)

                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))
            
            resultsTracker = tracker.update(detections)
            #print(resultsTracker)

            directions = []
            tracked_detections = []
            
            for result in resultsTracker:
                #print(type(result))

                # Get the tracker results
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                #print(result)

                # Display current objects IDs
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                id_txt = f"ID: {str(id)}"

                # Appending the detections to a list
                tracked_detections.append([str(frame_count), str(id), str(x1), str(y1), str(w), str(h), '-1', '-1', '-1'])
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 90, y1), (0, 0, 255), -1)
                cv2.putText(frame, id_txt, (int(x1 + 10), y1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                
                if frame_count % 5 == 0:
                    if id not in ids:
                        ids[id] = [(cx,cy),(0,0)]
                        #ids[id] = np.subtract((cx, cy), (0, 0))
                    else:
                        ids[id] = [(cx, cy), np.subtract((cx, cy), (ids[id][0][0], ids[id][0][1]))]
                        #print(np.subtract((cx, cy), (ids[id][0], ids[id][1])))

            print(ids)
            # Direction tracking
            for key in ids:
                 #centers = (cx, cy)
                 #change = tuple(np.subtract(centers, change))
                 diff = ids[key][1]
                 print('Diff', diff)
                 if diff[0]==0 and diff[1]==0:
                     continue
                 elif abs(diff[0]) >= abs(diff[1]):
                     if diff[0] > 0:
                         directions.append('East')  # East
                     else:
                         directions.append('West')  # West
                 else:
                     if diff[1] > 0:
                         directions.append('South')  # South
                     else:
                         directions.append('North')  # North
            print('\n')
            print('dir', directions)
            with open('tracked.txt', 'a') as f:
                for results in tracked_detections:
                    f.write(','.join(results))
                    f.write('\n')
            f.close()
            if directions:

                crowd_direct = self.find_mode(directions)
            else:
                # Handle the case where 'directions' is empty
                crowd_direct = None
            crowd_avg_direction.append(crowd_direct)
        
            frame = self.annotate_frames_find_direction(**display_args)
            cvzone.putTextRect(frame, f"CROWD DIRECTION : {crowd_direct}",
                                        [280, 80], thickness=3, scale=3, border=2)
            # Display Counting results
            count_txt = f"TOTAL COUNT : {len(resultsTracker)}"
            cvzone.putTextRect(frame, count_txt, [290, 34], thickness=3, scale=3, border=2)
            cvzone.putTextRect(frame, f"FPS : {fps:,.2f}", [100, 850],
                               thickness=3, scale=3, border=2)
            
            # Save annotated images to save_dir
            cv2.imwrite(os.path.join(save_dir, f"yolov8_frame_{frame_count:04d}.jpg"), frame)
            frame_count += 1
            
            if verbose:
                cv2.imshow('frame',frame)
                
                k = cv2.waitKey(1)
                if k == 27:
                    break

        cv2.destroyAllWindows()    

        print(statistics.mode(crowd_avg_direction)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=str, help='path to image folder')
    parser.add_argument('-s', '--save_dir', type=str, help='folder to save the results')
    parser.add_argument('-v', "--video_path", type=str, help="Path to the output video")
    parser.add_argument('-m', "--model_file", type=str, help="The Yolov8 model file or file path to be used")
    parser.add_argument('-mh', "--track_min_hits", type=int, help="The minimum number of hits to be used")
    args = parser.parse_args()

    with open("classes (1).txt") as f:
        classnames = f.read().splitlines()

    model = YOLOv8_ObjectTracker(labels=classnames, classes=[0], model_file=args.model_file, track_min_hits=args.track_min_hits)
    model.track_images(folder_path=args.image_folder, save_dir=args.save_dir)
    model.images_to_video(image_folder=args.save_dir, save_path=args.video_path)