#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand Gesture Recognition with SAHI-like Fallback maintaining Original Resolution

Original gesture preprocessing restored to maintain classification orientation.
"""
import csv
import copy
import argparse
import time
from collections import Counter, deque
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Constants
HISTORY_LEN = 16
# Load labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', 'r', encoding='utf-8-sig') as f:
    KP_LABELS = [r[0] for r in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', 'r', encoding='utf-8-sig') as f:
    PH_LABELS = [r[0] for r in csv.reader(f)]


def get_args():
    parser = argparse.ArgumentParser(description="SAHI-like Hand Gesture Recognition")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--use-static-image-mode", action="store_true")
    parser.add_argument("--tile-size", type=int, default=500, help="Base tile size for SAHI search")
    parser.add_argument("--tile-stride", type=int, default=250, help="Tile stride for SAHI search")
    parser.add_argument("--fallback-time-sec", type=float, default=1.0, help="Seconds of no detection before SAHI fallback")
    return parser.parse_args()


def main():
    args = get_args()
    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence)

    kp_clf = KeyPointClassifier()
    ph_clf = PointHistoryClassifier()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    fps_calc = CvFpsCalc(buffer_len=10)
    point_hist = deque(maxlen=HISTORY_LEN)
    gesture_hist = deque(maxlen=HISTORY_LEN)
    last_detect_time = time.time()

    mode = 0
    def select_mode(key):
        nonlocal mode
        n=-1
        if 48<=key<=57: n=key-48
        if key==ord('n'): mode=0
        if key==ord('k'): mode=1
        if key==ord('h'): mode=2
        return n

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv.flip(frame,1)
        fps = fps_calc.get()
        key = cv.waitKey(10)
        if key==27: break
        num = select_mode(key)

        found, debug = detect_and_classify(frame, hands, kp_clf, ph_clf,
                                           point_hist, gesture_hist,
                                           num, fps, mode)
        now = time.time()
        if found:
            last_detect_time = now
        elif now-last_detect_time>args.fallback_time_sec:
            found, debug = sahi_fallback(frame, hands, kp_clf, ph_clf,
                                          point_hist, gesture_hist,
                                          num, fps, mode,
                                          args.tile_size, args.tile_stride)
            if found: last_detect_time = now

        cv.imshow('Hand Gesture Recognition', debug)
    cap.release(); cv.destroyAllWindows()


def detect_and_classify(frame, hands, kp_clf, ph_clf,
                        point_hist, gesture_hist,
                        num, fps, mode):
    img=frame; debug=img.copy()
    rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img.flags.writeable=False
    res=hands.process(rgb)
    img.flags.writeable=True

    found=False
    if res.multi_hand_landmarks:
        found=True
        lms=res.multi_hand_landmarks[0]; hd=res.multi_handedness[0]
        brect=calc_bounding_rect(img,lms)
        lm_list=calc_landmark_list(img,lms)
        # original preprocessing
        plm=pre_process_landmark(lm_list)
        pph=pre_process_point_history(img,point_hist)
        logging_csv(num,mode,plm,pph)
        sid=kp_clf(plm)
        if sid==2: point_hist.append(lm_list[8])
        else:       point_hist.append([0,0])
        fid=0
        if len(pph)==HISTORY_LEN*2: fid=ph_clf(pph)
        gesture_hist.append(fid)
        most=Counter(gesture_hist).most_common(1)[0][0]
        debug=draw_bounding_rect(True,debug,brect)
        debug=draw_landmarks(debug,lm_list)
        debug=draw_info_text(debug,brect,hd, KP_LABELS[sid],PH_LABELS[most])
    else:
        point_hist.append([0,0])
    debug=draw_point_history(debug,point_hist)
    debug=draw_info(debug,fps,mode,num)
    return found,debug


def sahi_fallback(frame,hands,kp_clf,ph_clf,
                  point_hist,gesture_hist,
                  num,fps,mode,
                  base_size,stride):
    fh,fw=frame.shape[:2]; aspect=fw/fh
    tile_h=base_size; tile_w=int(base_size*aspect)
    debug=frame.copy(); found=False
    for y in range(0,fh-tile_h+1,stride):
        for x in range(0,fw-tile_w+1,stride):
            tile=frame[y:y+tile_h,x:x+tile_w]
            rgb=cv.cvtColor(tile,cv.COLOR_BGR2RGB)
            res=hands.process(rgb)
            if res.multi_hand_landmarks:
                found=True
                lms=res.multi_hand_landmarks[0]; hd=res.multi_handedness[0]
                brect=calc_bounding_rect(tile,lms)
                lm_list=calc_landmark_list(tile,lms)
                plm=pre_process_landmark(lm_list)
                pph=pre_process_point_history(tile,point_hist)
                logging_csv(num,mode,plm,pph)
                sid=kp_clf(plm)
                if sid==2: point_hist.append(lm_list[8])
                else:       point_hist.append([0,0])
                fid=0
                if len(pph)==HISTORY_LEN*2: fid=ph_clf(pph)
                gesture_hist.append(fid)
                most=Counter(gesture_hist).most_common(1)[0][0]
                x0,y0=x,y
                gr=[x0+brect[0],y0+brect[1],x0+brect[2],y0+brect[3]]
                debug=draw_bounding_rect(True,debug,gr)
                pts=[(pt[0]+x0,pt[1]+y0) for pt in lm_list]
                debug=draw_landmarks(debug,pts)
                debug=draw_info_text(debug,gr,hd, KP_LABELS[sid],PH_LABELS[most])
                break
        if found: break
    if not found: point_hist.append([0,0])
    debug=draw_point_history(debug,point_hist)
    debug=draw_info(debug,fps,mode,num)
    return found,debug

# Helper functions

def calc_bounding_rect(image,landmarks):
    h,w=image.shape[:2]
    pts=np.array([[int(lm.x*w),int(lm.y*h)] for lm in landmarks.landmark])
    x,y,ww,hh=cv.boundingRect(pts)
    return [x,y,x+ww,y+hh]

def calc_landmark_list(image,landmarks):
    h,w=image.shape[:2]
    return [[min(int(lm.x*w),w-1),min(int(lm.y*h),h-1)] for lm in landmarks.landmark]

# ORIGINAL preprocessing functions restored:

def pre_process_landmark(landmark_list):
    temp_landmark_list=copy.deepcopy(landmark_list)
    base_x,base_y=0,0
    for i,pt in enumerate(temp_landmark_list):
        if i==0:
            base_x,base_y=pt[0],pt[1]
        temp_landmark_list[i][0]=pt[0]-base_x
        temp_landmark_list[i][1]=pt[1]-base_y
    temp_one_d=list(itertools.chain.from_iterable(temp_landmark_list))
    max_v=max(map(abs,temp_one_d)) or 1
    return [v/max_v for v in temp_one_d]


def pre_process_point_history(image,point_history):
    w,h=image.shape[1],image.shape[0]
    temp=copy.deepcopy(point_history)
    base_x,base_y=0,0
    for i,pt in enumerate(temp):
        if i==0: base_x,base_y=pt[0],pt[1]
        temp[i][0]=(pt[0]-base_x)/w
        temp[i][1]=(pt[1]-base_y)/h
    return list(itertools.chain.from_iterable(temp))


def logging_csv(num,mode,lm,ph):
    if mode==1 and 0<=num<=9:
        with open('model/keypoint_classifier/keypoint.csv','a',newline='') as f:
            csv.writer(f).writerow([num,*lm])
    if mode==2 and 0<=num<=9:
        with open('model/point_history_classifier/point_history.csv','a',newline='') as f:
            csv.writer(f).writerow([num,*ph])

# Drawing placeholders

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__=='__main__': main()
