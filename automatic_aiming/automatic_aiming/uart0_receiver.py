import rclpy                                      
from rclpy.node   import Node 
import serial
from std_msgs.msg import String


class UartReceiver(Node):

    def __init__(self, name, serial_port, baudrate, topic_stack=10):
        super().__init__(name)
        self.name = name                    # 串口名称
        self.serial_port = serial_port      # 串口设备
        self.baudrate = baudrate            # 串口波特率
        self.topic_stack = topic_stack      # t

        self.pub = self.create_publisher(String, name + "_topic",  topic_stack)    # 创建发布方，发布数据到 uart0_receiver_topic 
        self.get_logger().info(f"创建{name}_topic发布方成功")
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate)   # 创建串口实例
            if not self.ser.isOpen():
                self.ser.open()
                self.get_logger().info(f"串口 {self.serial_port} 打开成功")
            else:
                self.get_logger().info(f"串口 {self.serial_port} 已打开")
        except Exception as e:
            self.get_logger().error(f"无法打开串口 {self.serial_port}: {str(e)}")
            return

        # 使用定时器定期检查串口数据
        self.timer = self.create_timer(0.01, self.check_serial_data)  # 10ms 检查一次

    def check_serial_data(self):
        try:
            if self.ser.isOpen():
                serial_data_length = self.ser.inWaiting()	
                if serial_data_length:
                    serial_data = self.ser.read(serial_data_length)
                    # 使用 decode 方法转换字节数据为字符串
                    serial_data = serial_data.decode('utf-8', errors='ignore')
                    self.publish(serial_data)

        except Exception as e:
            self.get_logger().error(str(e))

    def publish(self, data):
        msg = String()                                            
        msg.data = data                                    
        self.pub.publish(msg)                                    
        self.get_logger().info(f'接收数据发布到 uart0_receiver_topic | 数据：{data}')


def main(args=None):
    rclpy.init(args=args)
    uartReceiver = UartReceiver("uart0_receiver", "/dev/ttyS0", 115200)
    uartReceiver.get_logger().info(f"uart0_receiver @ {uartReceiver.serial_port} Buad: {uartReceiver.baudrate} init success")
    rclpy.spin(uartReceiver)
    uartReceiver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# ----------------------------------------------
# 其他节点 接收信息 回调函数
    def uart0_receiver_init(self, topic_stack=10):
        '''
            在非串口节点中使用
            订阅 uart0_receiver_topic 话题，接收相关数据进行处理
            【需搭配定时器进行使用】
        '''
        self.sub_uart0_receiver = self.create_subscription(String, "uart0_receiver_topic",self.uart0_receiver_callback, topic_stack)
        # 有数据发布在 uart0_receiver_topic 话题，即调用uart0_receiver_callback

    def uart0_receiver_callback(self, msg):
        '''
            在非串口节点中使用
            订阅 uart0_receiver_topic 话题，接收相关数据进行处理
            
        '''
        self.get_logger().info(f"接收到数据{msg}")
        