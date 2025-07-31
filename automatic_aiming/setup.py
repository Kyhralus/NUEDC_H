from setuptools import find_packages, setup
import glob

package_name = 'automatic_aiming'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装所有 launch 文件
        ('share/' + package_name + '/launch', glob.glob('launch/*launch.py')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='alineyiee@shu.edu.cn',
    description='NUEDC problem H',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = automatic_aiming.camera_publisher:main',
            'main_controller = automatic_aiming.main_controller:main',

            # 目标检测节点
            'target_detect = automatic_aiming.target_detect:main',
            # 云台控制器节点
            'gimbal_controller = automatic_aiming.gimbal_controller:main',
            # 单片机通信 uart 节点
            'uart1_receiver = automatic_aiming.uart1_receiver:main',
            'uart1_sender = automatic_aiming.uart1_sender:main',
            # 语音模块 uart 节点
            'uart3_receiver = automatic_aiming.uart3_receiver:main',
            'uart3_sender = automatic_aiming.uart3_sender:main',
            'test_node = automatic_aiming.test_node:main',
            'speakr_node = automatic_aiming.speakr_node:main',
            # 备用 uart 节点
            'uart0_receiver = automatic_aiming.uart0_receiver:main',
            'uart0_sender = automatic_aiming.uart0_sender:main',
        ],
    },
)

