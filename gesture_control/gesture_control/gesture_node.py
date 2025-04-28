#!/usr/bin/env python3
import os
import sys
import yaml
import logging
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import CommandBool, CommandTOL
from mavros_msgs.msg import State

from addict import Dict
from ament_index_python.packages import get_package_share_directory

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gesture_node')

def get_cfg() -> Dict:
    pkg_share = get_package_share_directory('gesture_control')
    config_path = os.path.join(pkg_share, 'params', 'params.yaml')
    with open(config_path, 'rb') as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    return Dict(raw['gesture_classifier'])

class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.cfg = get_cfg()
        
        # Добавим путь к скрипту
        script_dir = os.path.dirname(self.cfg.script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Импортируем и создаём классификатор
        from app_for_ros2 import GestureClassifier
        self.classifier = GestureClassifier(self.cfg)

        self.bridge = CvBridge()
        self.current_command = (0.0, 0.0, 0.0, 0.0)  # x, y, z, yaw
        self.current_state = State()
        self.offboard_attempts = 0
        self.max_offboard_attempts = 5
        self.disarm_count = 0
        self.disarm_threshold = 15  # Требуется 15 последовательных распознаваний Disarm

        # Подписка на состояние дрона
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            self.cfg.topic.image_input,
            self.image_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            self.cfg.topic.cmd_vel_output,
            10
        )

        self.arm_client = self.create_client(CommandBool, self.cfg.topic.arming_srv)
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(' Ожидание MAVROS /cmd/arming...')

        from mavros_msgs.srv import SetMode
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(' Ожидание MAVROS /set_mode...')

        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(' Ожидание MAVROS /cmd/takeoff...')

        self.gesture_commands = {
            'Up':       (0.0, 0.0, 1.0, 0.0),
            'Down':     (0.0, 0.0, -1.0, 0.0),
            'Left':     (0.0, 1.0, 0.0, 0.0),
            'Right':    (0.0, -1.0, 0.0, 0.0),
            'Forward':  (1.0, 0.0, 0.0, 0.0),
            'Backward': (-1.0, 0.0, 0.0, 0.0),
        }

        self.speed = self.cfg.get('speed', 0.5)

        # Таймер для команд и проверки режима
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(f' GestureNode инициализирован, подписан на {self.cfg.topic.image_input}')

    def state_callback(self, msg):
        self.current_state = msg
        self.get_logger().info(f'Текущий режим: {self.current_state.mode}, Armed: {self.current_state.armed}')

    def set_mode(self, mode_name: str):
        from mavros_msgs.srv import SetMode
        req = SetMode.Request()
        req.custom_mode = mode_name
        future = self.set_mode_client.call_async(req)
        self.get_logger().info(f'Попытка установить режим {mode_name}')
        future.add_done_callback(lambda f: self.mode_response_cb(f, mode_name))

    def mode_response_cb(self, future, mode_name):
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info(f'✅ Режим {mode_name} установлен')
                if mode_name == 'OFFBOARD':
                    self.offboard_attempts = 0  # Сбрасываем счётчик попыток
            else:
                self.get_logger().warn(f'⚠️ Не удалось установить режим {mode_name}')
                if mode_name == 'OFFBOARD' and self.offboard_attempts < self.max_offboard_attempts:
                    self.offboard_attempts += 1
                    self.get_logger().info(f'Повторная попытка установить OFFBOARD ({self.offboard_attempts}/{self.max_offboard_attempts})')
                    self.set_mode('OFFBOARD')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при установке режима {mode_name}: {e}')

    def timer_callback(self):
        x, y, z, yaw = self.current_command
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = x
        twist.twist.linear.y = y
        twist.twist.linear.z = z
        twist.twist.angular.z = yaw
        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(f'Команда отправлена: x={x}, y={y}, z={z}, yaw={yaw}')

        # Проверяем, если дрон в режиме AUTO.LOITER после арминга
        if self.current_state.armed and self.current_state.mode != 'OFFBOARD' and self.offboard_attempts < self.max_offboard_attempts:
            self.get_logger().info(f'Дрон в режиме {self.current_state.mode}, попытка переключить в OFFBOARD')
            self.set_mode('OFFBOARD')

    def arm_response_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('🛩️ Arm успешен')
                # Начинаем публиковать команды скорости для OFFBOARD
                self.current_command = (0.0, 0.0, 0.0, 0.0)  # Нулевые команды для стабильности
                # Устанавливаем режим OFFBOARD
                self.set_mode('OFFBOARD')
                # Отправляем команду на взлёт
                self.send_takeoff_command()
            else:
                self.get_logger().warn(f'⚠️ Arm отказ: result={result.result}')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при обработке Arm: {e}')

    def disarm_response_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('🛬 Disarm успешен')
                self.disarm_count = 0  # Сбрасываем счётчик
            else:
                self.get_logger().warn(f'⚠️ Disarm отказ: result={result.result}')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при обработке Disarm: {e}')

    def send_takeoff_command(self):
        req = CommandTOL.Request()
        req.altitude = 2.0  # Высота взлёта в метрах
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        future = self.takeoff_client.call_async(req)
        future.add_done_callback(self.takeoff_response_cb)

    def takeoff_response_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('🛫 Взлёт успешен')
            else:
                self.get_logger().warn(f'⚠️ Взлёт отказ: result={result.result}')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при обработке взлёта: {e}')

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img = cv2.flip(img, 1)  # Переворачиваем изображение по горизонтали
            gesture, move_gesture = self.classifier.classify_gesture(img)
            self.get_logger().info(f' Распознан жест: {gesture}')

            if gesture == 'Arm':
                self.disarm_count = 0  # Сбрасываем счётчик Disarm
                req = CommandBool.Request()
                req.value = True
                future = self.arm_client.call_async(req)
                future.add_done_callback(self.arm_response_cb)

            elif gesture == 'Disarm':
                self.disarm_count += 1
                self.get_logger().info(f'Обнаружен Disarm, счётчик: {self.disarm_count}/{self.disarm_threshold}')
                if self.disarm_count >= self.disarm_threshold:
                    req = CommandBool.Request()
                    req.value = False
                    future = self.arm_client.call_async(req)
                    future.add_done_callback(self.disarm_response_cb)
                else:
                    self.current_command = (0.0, 0.0, 0.0, 0.0)  # Останавливаем движение

            elif gesture in self.gesture_commands:
                self.disarm_count = 0  # Сбрасываем счётчик Disarm
                x, y, z, yaw = self.gesture_commands[gesture]
                if gesture not in ['Up', 'Down']:
                    z = 0.0
                self.current_command = (x * self.speed, y * self.speed, z * self.speed, yaw)
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = 'base_link'
                twist.twist.linear.x = x * self.speed
                twist.twist.linear.y = y * self.speed
                twist.twist.linear.z = z * self.speed
                twist.twist.angular.z = yaw
                self.cmd_vel_pub.publish(twist)
                self.get_logger().info(f' Команда: x={x}, y={y}, z={z}, yaw={yaw}')
            else:
                self.disarm_count = 0  # Сбрасываем счётчик Disarm
                self.get_logger().debug('Жест не распознан или не предусмотрен')
                self.current_command = (0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            self.get_logger().error(f' Ошибка обработки изображения: {e}')

    def destroy_node(self):
        self.classifier.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()