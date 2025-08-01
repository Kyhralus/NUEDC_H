import wiringpi
from wiringpi import GPIO  
# 初始化GPIO
laser_switch_pin = 5  # 激光安全开关引脚编号
wiringpi.wiringPiSetup()
# 设置激光安全开关引脚为输入模式，默认上拉
wiringpi.pinMode(laser_switch_pin, GPIO.INPUT)
wiringpi.pullUpDnControl(laser_switch_pin, GPIO.PUD_UP)  # 上拉电阻
while wiringpi.digitalRead(laser_switch_pin):  # 读取激光安全开关状态
    print("高")
print("低")