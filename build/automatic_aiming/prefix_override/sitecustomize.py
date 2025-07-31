import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/orangepi/ros2_workspace/NUEDC_H/install/automatic_aiming'
