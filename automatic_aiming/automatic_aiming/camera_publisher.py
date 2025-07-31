import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
        else:
            self.timer = self.create_timer(0.01, self.publish_image)  # Hz 发布频率

    def publish_image(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(msg)
            self.get_logger().debug('发布图像')
        else:
            self.get_logger().warn('无法捕获图像')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    camera_publisher.get_logger().info('摄像头发布节点已启动')
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()