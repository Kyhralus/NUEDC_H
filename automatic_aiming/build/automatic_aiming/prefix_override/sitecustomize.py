import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/orangepi/ros2_workspace/NUEDC_H/automatic_aiming/install/automatic_aiming'
