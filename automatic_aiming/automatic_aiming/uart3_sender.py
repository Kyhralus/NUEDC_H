"""
    请先开sender，再开receiver
"""


import rclpy                                      
from rclpy.node   import Node 
import serial         
from std_msgs.msg import String                  


class UartSender(Node):
    def __init__(self, name, serial_port, baudrate, topic_stack=50):
        super().__init__(name)

        self.name = name
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.topic_stack = topic_stack

        # 存储最近3条消息，用于去重
        self.message_history = []
        self.max_history_size = 3

        self.ser = serial.Serial(self.serial_port, self.baudrate)
        self.sub = self.create_subscription(String, name + "_topic", self.send_uart_data, topic_stack)    # 订阅空话题会自己创建话题
        self.get_logger().info(f"创建{name}_topic订阅方成功！")

    def is_duplicate_message(self, new_message):
        """检查新消息是否与最近的两条消息相同"""
        if len(self.message_history) >= 2:
            # 检查最近的两条消息是否都与新消息相同
            return (self.message_history[-1] == new_message and 
                    self.message_history[-2] == new_message)
        return False
    
    def add_to_history(self, message):
        """将消息添加到历史记录中"""
        self.message_history.append(message)
        # 保持历史记录不超过最大长度
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)
    
    def send_uart_data(self, msg):
        # 检查是否为重复消息（连续两条相同消息）
        if self.is_duplicate_message(msg.data):
            self.get_logger().info(f"检测到连续重复消息，跳过发送: {msg.data}")
            return True
        
        # 添加到历史记录
        self.add_to_history(msg.data)
        
        self.get_logger().info(f"request to send: {msg.data}")
        if self.ser.isOpen():
            try:
                # 将字符串转换为字节（使用UTF-8编码）
                byte_data = msg.data.encode('utf-8')
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
    uartSender = UartSender("uart3_sender", "/dev/ttyS3", 115200)
    uartSender.get_logger().info(f"uart3_sender @ {uartSender.serial_port} Buad: {uartSender.baudrate} init success")
    rclpy.spin(uartSender)
    uartSender.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# ----------------------------------------------
# 其他节点 发送信息 函数
def uart3_sender_init(self, topic_stack=10):
    '''
        在其他话题的初始化中使用
        创建一个发布方 ---- 发布方功能：发送数据给串口 
        功能：发送数据到 uart_sender_topic
    '''
    self.pub_uart3_sender = self.create_publisher(String, 'uart3_sender_topic', topic_stack)
    self.get_logger().info(f'创建uart3_sender_topic发布方成功！')  

def uart3_sender_send(self, data):
    '''
        在非串口话题中使用
        发送数据到话题 uart_sender_topic
    '''
    msg = String()
    msg.data = data
    self.pub_uart3_sender.publish(msg)
    self.get_logger().info(f'发送数据:{msg}到 uart3')  