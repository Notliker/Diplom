#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import numpy as np
import cv2 as cv
import mediapipe as mp
from collections import deque
from model import KeyPointClassifier
from model import PointHistoryClassifier
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureClassifier:
    def __init__(self, cfg):
        """
        Инициализация классификатора жестов.

        Args:
            cfg (dict): Конфигурация с путями к меткам и настройками MediaPipe.
        """
        logger.info("Инициализация GestureClassifier")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg['hands']['max_num_hands'],
            min_detection_confidence=cfg['hands']['min_detection_confidence'],
            min_tracking_confidence=cfg['hands']['min_tracking_confidence']
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=cfg['pose']['model_complexity'],
            min_detection_confidence=cfg['pose']['min_detection_confidence'],
            min_tracking_confidence=cfg['pose']['min_tracking_confidence']
        )
        # передаём явные пути к .tflite и .csv из конфига
        self.keypoint_classifier = KeyPointClassifier(
            model_path=cfg['keypoint_classifier']['model_path']
        )
        self.point_history_classifier = PointHistoryClassifier(
            model_path=cfg['point_history_classifier']['model_path']
        )
        # метки по-прежнему читаются из cfg
        self.keypoint_labels = self._load_labels(
            cfg['keypoint_classifier']['label_path']
        )
        self.point_history_labels = self._load_labels(
            cfg['point_history_classifier']['label_path']
        )
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        logger.info("GestureClassifier успешно инициализирован")

    def _load_labels(self, label_path):
        """Загрузка меток классов из CSV файла."""
        try:
            with open(label_path, encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                labels = [row[1] if len(row) > 1 else row[0] for row in reader]
                logger.info(f"Загружены метки из {label_path}: {labels}")
                return labels
        except FileNotFoundError:
            logger.error(f"Файл меток не найден: {label_path}")
            raise

    def _calc_bounding_rect(self, image, landmarks):
        """Вычисление ограничивающего прямоугольника для руки."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def _calc_landmark_list(self, image, landmarks):
        """Вычисление списка координат ключевых точек."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def _pre_process_landmark(self, landmark_list, brect):
        """Предобработка ключевых точек для классификации."""
        temp_landmark_list = copy.deepcopy(landmark_list)
        brect_width = brect[2] - brect[0]
        brect_height = brect[3] - brect[1]

        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for index, landmark_point in enumerate(temp_landmark_list):
            temp_landmark_list[index][0] = (landmark_point[0] - base_x) / brect_width if brect_width != 0 else 0
            temp_landmark_list[index][1] = (landmark_point[1] - base_y) / brect_height if brect_height != 0 else 0

        return list(np.array(temp_landmark_list).flatten())

    def _pre_process_point_history(self, image, point_history):
        """Предобработка истории точек для классификации."""
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)

        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (point[0] - base_x) / image_width if image_width != 0 else 0
            temp_point_history[index][1] = (point[1] - base_y) / image_height if image_height != 0 else 0

        return list(np.array(temp_point_history).flatten())

    def _process_pose_and_crop(self, image):
        """Обработка позы и обрезка/масштабирование изображения."""
        logger.debug("Обработка позы и обрезка изображения")
        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        frame_h, frame_w, _ = image.shape

        if results_pose.pose_landmarks:
            logger.debug("Обнаружена поза")
            orig_min_x = min(int(l.x * frame_w) for l in results_pose.pose_landmarks.landmark)
            orig_max_x = max(int(l.x * frame_w) for l in results_pose.pose_landmarks.landmark)
            orig_min_y = min(int(l.y * frame_h) for l in results_pose.pose_landmarks.landmark)
            orig_max_y = max(int(l.y * frame_h) for l in results_pose.pose_landmarks.landmark)
            person_height_without_padding = orig_max_y - orig_min_y

            padding = 50
            min_x = max(0, orig_min_x - padding)
            max_x = min(frame_w, orig_max_x + padding)
            min_y = max(0, orig_min_y - padding)
            max_y = min(frame_h, orig_max_y + padding)

            person_width = max_x - min_x
            person_height = max_y - min_y
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            scaling_factor = min(500 / person_height_without_padding, 3)

            cropped_x_min = max(0, center_x - person_width // 2 - padding)
            cropped_x_max = min(frame_w, center_x + person_width // 2 + padding)
            cropped_y_min = max(0, center_y - person_height // 2 - padding)
            cropped_y_max = min(frame_h, center_y + person_height // 2 + padding)

            cropped_image = image[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max]

            person_height = cropped_image.shape[0]
            if person_height > 0:
                scaling_factor = 500 / person_height
                scaled_width = int(cropped_image.shape[1] * scaling_factor)
                scaled_height = int(cropped_image.shape[0] * scaling_factor)
                logger.debug(f"Изображение обрезано и масштабировано до {scaled_width}x{scaled_height}")
                return cv.resize(cropped_image, (scaled_width, scaled_height), interpolation=cv.INTER_LANCZOS4)
        logger.debug("Поза не обнаружена, возвращается исходное изображение")
        return cv.resize(image, (1280, 720), interpolation=cv.INTER_LANCZOS4)

    def classify_gesture(self, image):
        """Классификация жеста на основе входного изображения."""
        logger.debug("Начало классификации жеста")
        processed_image = self._process_pose_and_crop(image)
        image_rgb = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            logger.debug("Обнаружены руки")
            hand_landmarks = results.multi_hand_landmarks[0]
            brect = self._calc_bounding_rect(processed_image, hand_landmarks)
            landmark_list = self._calc_landmark_list(processed_image, hand_landmarks)
            pre_processed_landmark_list = self._pre_process_landmark(landmark_list, brect)

            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
            logger.debug(f"ID жеста: {hand_sign_id}")

            if hand_sign_id == 2:
                self.point_history.append(landmark_list[8])
            else:
                self.point_history.append([0, 0])

            pre_processed_point_history_list = self._pre_process_point_history(processed_image, self.point_history)
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (self.history_length * 2):
                finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                self.finger_gesture_history.append(finger_gesture_id)
                logger.debug(f"ID жеста пальца: {finger_gesture_id}")

            gesture_label = self.keypoint_labels[hand_sign_id] if hand_sign_id < len(self.keypoint_labels) else None
            logger.info(f"Распознан жест: {gesture_label}")
            return gesture_label
        else:
            logger.debug("Руки не обнаружены")
            self.point_history.append([0, 0])
            return None

    def release(self):
        """Освобождение ресурсов."""
        logger.info("Освобождение ресурсов GestureClassifier")
        self.hands.close()
        self.pose.close()