#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import numpy as np
import cv2 as cv
import mediapipe as mp
from collections import deque, Counter
from model import KeyPointClassifier, PointHistoryClassifier
import logging
import time
import itertools

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
        self.keypoint_classifier = KeyPointClassifier(
            model_path=cfg['keypoint_classifier']['model_path']
        )
        self.point_history_classifier = PointHistoryClassifier(
            model_path=cfg['point_history_classifier']['model_path']
        )
        self.keypoint_labels = self._load_labels(
            cfg['keypoint_classifier']['label_path']
        )
        self.point_history_labels = self._load_labels(
            cfg['point_history_classifier']['label_path']
        )

        # SAHI fallback параметры
        self.fallback_sec = cfg.get('fallback_time_sec', 1.0)
        self.tile_size = cfg.get('tile_size', 500)
        self.tile_stride = cfg.get('tile_stride', 250)
        self.last_detect_time = time.time()

        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        logger.info("GestureClassifier успешно инициализирован")

    def _load_labels(self, label_path):
        """Загрузка меток классов из CSV файла."""
        try:
            with open(label_path, encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                labels = [row[0] for row in reader]
                logger.info(f"Загружены метки из {label_path}: {labels}")
                return labels
        except FileNotFoundError:
            logger.error(f"Файл меток не найден: {label_path}")
            raise

    def _calc_bounding_rect(self, image, landmarks):
        """Вычисление ограничивающего прямоугольника для руки."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.array([(lm.x * image_width, lm.y * image_height) for lm in landmarks.landmark])
        x, y, w, h = cv.boundingRect(landmark_array.astype(np.float32))
        return [x, y, x + w, y + h]

    def _calc_landmark_list(self, image, landmarks):
        """Вычисление списка координат ключевых точек."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_points = []
        for lm in landmarks.landmark:
            landmark_points.append([int(lm.x * image_width), int(lm.y * image_height)])
        return landmark_points

    def _pre_process_landmark(self, landmark_list):
        """Оригинальная предобработка ключевых точек для классификации."""
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list

    def _pre_process_point_history(self, image, point_history):
        """Предобработка истории точек для классификации."""
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def classify_gesture(self, image):
        """Классификация жеста на основе входного изображения."""
        now = time.time()
        logger.debug("Начало классификации жеста")

        processed_image = image  # Используем изображение напрямую без масштабирования
        image_rgb = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        hand_sign_label = None
        point_history_label = None

        if results.multi_hand_landmarks:
            self.last_detect_time = now
            logger.debug("Обнаружены руки")

            hand_landmarks = results.multi_hand_landmarks[0]
            brect = self._calc_bounding_rect(processed_image, hand_landmarks)
            landmark_list = self._calc_landmark_list(processed_image, hand_landmarks)
            pre_processed_landmark_list = self._pre_process_landmark(landmark_list)

            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Указательный палец
                self.point_history.append(landmark_list[8])
            else:
                self.point_history.append([0, 0])

            pre_processed_point_history_list = self._pre_process_point_history(processed_image, self.point_history)
            if len(pre_processed_point_history_list) == self.history_length * 2:
                finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common(1)[0][0]
                point_history_label = self.point_history_labels[most_common_fg_id]

            hand_sign_label = self.keypoint_labels[hand_sign_id]

        else:
            logger.debug("Руки не обнаружены")
            self.point_history.append([0, 0])

            if now - self.last_detect_time > self.fallback_sec:
                logger.info("Активирован SAHI fallback")
                found, hand_sign_id, finger_gesture_id = self._sahi_fallback(image)
                if found:
                    hand_sign_label = self.keypoint_labels[hand_sign_id]
                    if finger_gesture_id is not None:
                        point_history_label = self.point_history_labels[finger_gesture_id]
                    self.last_detect_time = now

        logger.info(f"Распознанные жесты - Поза: {hand_sign_label}, Движение пальца: {point_history_label}")
        return hand_sign_label, point_history_label

    def _sahi_fallback(self, image):
        """SAHI fallback для поиска рук в окнах изображения."""
        fh, fw = image.shape[:2]
        tile_h = self.tile_size
        tile_w = int(self.tile_size * (fw / fh))  # Сохраняем соотношение сторон
        stride = self.tile_stride
        found = False
        hand_sign_id = None
        finger_gesture_id = None

        for y in range(0, fh - tile_h + 1, stride):
            for x in range(0, fw - tile_w + 1, stride):
                tile = image[y:y + tile_h, x:x + tile_w]
                rgb = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    found = True
                    hand_landmarks = res.multi_hand_landmarks[0]
                    landmark_list = self._calc_landmark_list(tile, hand_landmarks)
                    pre_processed_landmark_list = self._pre_process_landmark(landmark_list)

                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:
                        self.point_history.append(landmark_list[8])
                    else:
                        self.point_history.append([0, 0])

                    pre_processed_point_history_list = self._pre_process_point_history(tile, self.point_history)
                    if len(pre_processed_point_history_list) == self.history_length * 2:
                        finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                        self.finger_gesture_history.append(finger_gesture_id)
                    break
            if found:
                break
        return found, hand_sign_id, finger_gesture_id

    def release(self):
        """Освобождение ресурсов."""
        logger.info("Освобождение ресурсов GestureClassifier")
        self.hands.close()