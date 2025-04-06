import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mpl_toolkits.mplot3d import Axes3D


class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/rotated_pose',
            self.pose_callback,
            10
        )
        self.positions_x = []
        self.positions_y = []
        self.positions_z = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()
    
    def pose_callback(self, msg):
        self.positions_x.append(msg.pose.position.x)
        self.positions_y.append(msg.pose.position.y)
        self.positions_z.append(msg.pose.position.z)
        self.update_plot()
    
    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.positions_x, self.positions_y, self.positions_z, marker='o', linestyle='-')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.1)

def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = PoseSubscriber()
    rclpy.spin(pose_subscriber)
    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
