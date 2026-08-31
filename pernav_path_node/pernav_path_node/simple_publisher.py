import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    def __init__(self) -> None:
        super().__init__('simple_publisher_node')
        self.publisher_ = self.create_publisher(String, 'pernav/chatter', 10)
        self.timer_ = self.create_timer(1.0, self.timer_callback)
        self.count_ = 0
        self.get_logger().info('Simple publisher started on topic pernav/chatter')

    def timer_callback(self) -> None:
        msg = String()
        msg.data = f'hello from pernav_path_node #{self.count_}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.count_ += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimplePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
