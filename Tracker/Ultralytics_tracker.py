import cv2
import numpy as np
from ultralytics import YOLO
import os

import cvzone
import statistics
import ImageIterator
import time


class UltralyticsTracker:
    def __init__(self, model_file='yolov8n.pt',
                 classes=None, conf=0.25, iou=0.45, tracker='botsort.yaml'):
        self.classes = classes
        self.conf = conf
        self.iou = iou

        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None

        self.tracker = tracker

        # Function to convert images to videos

    def images_to_video(self, image_folder, save_path, size=(1000, 500)):
        img_array = []

        for image in os.listdir(image_folder):
            img = cv2.imread(os.path.join(image_folder, image))
            # img = cv2.resize(img, size)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        # Save the video in specified path
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        return out

    def track_images(self, folder_path, save_dir, frame_size=(1000, 500), verbose=True):
        cap = ImageIterator.ImageIterator(folder_path)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        cv2.namedWindow('Tracking_results', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracking_results', 1000, 500)

        frame_count = 0
        ids = {}
        crowd_avg_direction = []
        for frame in cap:
            # Calculate FPS
            beg = time.time()

            # print(fps)
            results = self.model.track(frame, persist=True, tracker=self.tracker,
                                       classes=self.classes, conf=self.conf, iou=self.iou)
            fps = 1 / (time.time() - beg)
            print(fps)
            annotated_frame = results[0].plot(conf=False, font_size=0.25, labels=False)

            directions = []
            tracked_detections = []

            for r in results[0]:
                track_id = int(r.boxes.id.item())
                cx, cy, w, h = np.squeeze(r.boxes.xywh.numpy())

                tracked_detections.append(
                    [str(frame_count), str(track_id), str(cx), str(cy), str(w), str(h), '-1', '-1', '-1'])

                if frame_count % 5 == 0:
                    if id not in ids:
                        ids[id] = [(cx, cy), (0, 0)]
                        # ids[id] = np.subtract((cx, cy), (0, 0))
                    else:
                        ids[id] = [(cx, cy), np.subtract((cx, cy), (ids[id][0][0], ids[id][0][1]))]
            # print(ids)
            # Direction tracking
            for key in ids:
                # centers = (cx, cy)
                # change = tuple(np.subtract(centers, change))
                diff = ids[key][1]
                # print(diff)
                if diff[0] == 0 and diff[1] == 0:
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
                        directions.append('North')

            print('\n')

            with open('tracked.txt', 'a') as f:
                for result in tracked_detections:
                    f.write(','.join(result))
                    f.write('\n')
            f.close()
            if directions:
                crowd_direct = statistics.mode(directions)
            else:
                # Handle the case where 'directions' is empty
                crowd_direct = None
            crowd_avg_direction.append(crowd_direct)

            cvzone.putTextRect(annotated_frame, f"CROWD DIRECTION : {crowd_direct}",
                               [280, 80], thickness=3, scale=3, border=2)

            # Display Counting results
            detect_count = len(results[0].boxes.id.numpy())
            # print(f'Detection count: {detect_count}')
            count_txt = f"TOTAL COUNT : {detect_count}"
            cvzone.putTextRect(annotated_frame, count_txt, [290, 34], thickness=3, scale=3, border=2)
            cvzone.putTextRect(annotated_frame, f"FPS : {fps:,.2f}", [100, 850],
                               thickness=3, scale=3, border=2)

            # Save annotated images to save_dir
            cv2.imwrite(os.path.join(save_dir, f"yolov8_frame_{frame_count:04d}.jpg"), annotated_frame)
            frame_count += 1

            if verbose:
                cv2.imshow('Tracking_results', annotated_frame)

                k = cv2.waitKey(1)
                if k == 27:
                    break

        cv2.destroyAllWindows()

        print(statistics.mode(crowd_avg_direction))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', type=str, help='Path to input images folder')
    parser.add_argument('-s', '--save_dir', type=str, help='Path to output folder')
    parser.add_argument('-v', '--output_video', type=str, help='Path to output video')
    parser.add_argument('-t', '--tracker', type=str, choices=['botsort.yaml', 'bytetrack.yaml'],
                        help='Choose a tracker')
    parser.add_argument('-m', '--model_file', type=str, help='YOLO model to be used')

    args = parser.parse_args()

    model = UltralyticsTracker(model_file=args.model_file, classes=[0], tracker=args.tracker)
    model.track_images(folder_path=args.input_folder, save_dir=args.save_dir)
    model.images_to_video(image_folder=args.save_dir, save_path=args.output_video)
