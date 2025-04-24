#!/usr/bin/env python3
import os
import sys
import yaml
import logging

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import CommandBool

from addict import Dict
from ament_index_python.packages import get_package_share_directory

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gesture_node')

def get_cfg() -> Dict:
    pkg_share = get_package_share_directory('gesture_control')
    config_path = os.path.join(pkg_share, 'params', 'params.yaml')
    with open(config_path, 'rb') as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)  # <--- Вот тут создаётся raw
    return Dict(raw['gesture_classifier'])  # <--- А тут ты уже его используешь


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

        self.gesture_commands = {
            'Up':       (0.0, 0.0, 1.0, 0.0),
            'Down':     (0.0, 0.0, -1.0, 0.0),
            'Left':     (0.0, 1.0, 0.0, 0.0),
            'Right':    (0.0, -1.0, 0.0, 0.0),
            'Forward':  (1.0, 0.0, 0.0, 0.0),
            'Backward': (-1.0, 0.0, 0.0, 0.0),
        }

        self.speed = self.cfg.get('speed', 0.5)

        # Таймер, который будет публиковать команды
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(f' GestureNode инициализирован, подписан на {self.cfg.topic.image_input}')


    def set_mode(self, mode_name: str):
        from mavros_msgs.srv import SetMode
        req = SetMode.Request()
        req.custom_mode = mode_name
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(lambda f: self.mode_response_cb(f, mode_name))


    def mode_response_cb(self, future, mode_name):
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info(f'✅ Режим {mode_name} установлен')
            else:
                self.get_logger().warn(f'⚠️ Не удалось установить режим {mode_name}')
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


    def set_offboard_mode(self):
        req = SetMode.Request()
        req.custom_mode = 'OFFBOARD'
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(self.offboard_response_cb)


    def offboard_response_cb(self, future):
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info('🛠️ Режим OFFBOARD установлен')
            else:
                self.get_logger().warn('⚠️ Не удалось установить режим OFFBOARD')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при установке режима: {e}')



    def arm_response_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('🛩️ Arm успешен')
                # Сначала — POSCTL
                self.set_mode('POSCTL')
                # Ждём чуть-чуть и потом OFFBOARD
                self.create_timer(1.0, lambda: self.set_mode('OFFBOARD'))
            else:
                self.get_logger().warn(f'⚠️ Arm отказ: result={result.result}')
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка при обработке Arm: {e}')


            
    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gesture = self.classifier.classify_gesture(img)
            self.get_logger().info(f' Распознан жест: {gesture}')

            if gesture == 'Arm':
                req = CommandBool.Request()
                req.value = True
                future = self.arm_client.call_async(req)
                future.add_done_callback(self.arm_response_cb)

            elif gesture == 'Disarm':
                req = CommandBool.Request()
                req.value = False
                future = self.arm_client.call_async(req)
                future.add_done_callback(self.arm_response_cb)

            elif gesture in self.gesture_commands:
                x, y, z, yaw = self.gesture_commands[gesture]
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
                self.get_logger().debug('Жест не распознан или не предусмотрен')
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