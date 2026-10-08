"""Read-only RViz preview of one saved snapshot. Never captures or moves."""
import json

from irb120_perception.press_targets import press_targets, build_press_markers


def preview(episode, filename='contacts.json'):
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from visualization_msgs.msg import MarkerArray
    stage = next((s for s in reversed(episode.load()['stages'])
                  if s['kind'] == 'perceive' and s['status'] == 'ok'), None)
    if stage is None:
        raise ValueError('No successful snapshot in this episode')
    path = episode.path / stage['dir'] / filename
    contacts = json.loads(path.read_text())
    targets = press_targets(contacts['press'])
    rclpy.init()
    node = rclpy.create_node('saved_press_preview')
    publisher = node.create_publisher(MarkerArray, '/contact_point_selector/markers',
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def publish():
        publisher.publish(build_press_markers(targets, 'base_link', node.get_clock().now().to_msg(),
                          episode.path.name+'/'+stage['dir']+'/'+filename))
    node.create_timer(.5, publish)
    publish()
    node.get_logger().info(f'Read-only preview: {path}\n'
        f'Contact {targets["contact"]}; pre-press ball {targets["pre_press"]} (base_link). '
        'RViz: /contact_point_selector/markers. Ctrl-C stops preview. '
        'Arc reuses this snapshot only when perceive is the last stage; otherwise it captures a new one.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
