"""
    请先开sender，再开receiver
"""


import rclpy                                      
from rclpy.node   import Node 
import threading
import serial         
from std_msgs.msg import String                  


class UartSender(Node):
    def __init__(self, name, serial_port, baudrate, topic_stack=50):
        super().__init__(name)

        self.name = name
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.topic_stack = topic_stack

        self.ser = serial.Serial(self.serial_port, self.baudrate)
        self.serial_lock = threading.Lock()  # 串口锁
        self.sub = self.create_subscription(String, name + "_topic", self.send_uart_data, topic_stack)    # 订阅空话题会自己创建话题
        self.get_logger().info(f"创建{name}_topic订阅方成功！")

    def send_uart_data(self, msg):
        self.get_logger().info(f"request to send: {msg.data}")
        if self.ser.isOpen():
            try:
                # 将字符串转换为字节（使用UTF-8编码）
                byte_data = msg.data.encode('utf-8')
                with self.serial_lock:  # 获取锁，确保独占串口
                    send_count = self.ser.write(byte_data)
                    if send_count == len(byte_data):
                        return True
                    else:
                        self.get_logger().warn(f"仅发送了{send_count}/{len(byte_data)}字节")
                        return False
            except Exception as e:
                self.get_logger().error(f"发送数据失败: {e}")
                return False
        else:
            self.get_logger().error("串口未打开")
            return False
            

def main(args=None):
    rclpy.init(args=args)
    uartSender = UartSender("uart1_sender", "/dev/ttyS1", 115200)
    uartSender.get_logger().info(f"uart1_sender @ {uartSender.serial_port} Buad: {uartSender.baudrate} init success")
    rclpy.spin(uartSender)
    uartSender.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# ----------------------------------------------
# 其他节点 发送信息 函数
def uart1_sender_init(self, topic_stack=10):
    '''
        在其他话题的初始化中使用
        创建一个发布方 ---- 发布方功能：发送数据给串口 
        功能：发送数据到 uart_sender_topic
    '''
    self.pub_uart1_sender = self.create_publisher(String, 'uart1_sender_topic', topic_stack)
    self.get_logger().info(f'创建uart1_sender_topic发布方成功！')  

def uart1_sender_send(self, data):
    '''
        在非串口话题中使用
        发送数据到话题 uart_sender_topic
    '''
    msg = String()
    msg.data = data
    self.pub_uart1_sender.publish(msg)
    self.get_logger().info(f'发送数据:{msg}到 uart1')  