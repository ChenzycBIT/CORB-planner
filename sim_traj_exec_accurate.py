import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import random
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from traj_utils.msg import Bspline
from nav_msgs.msg import Odometry
import copy
import time




def stage3_bdescpline_base(u):
    if u > 2 and u <= 4:
        u = 4 - u
    if u >= 0 and u <= 1:
        y = u ** 3 / 6
    elif u <= 2:
        y = (-3 * (u ** 3) + 12 * (u ** 2) - 12 * u + 4) / 6
    else:
        raise ValueError('input must be in the range of [0, 4]')
    return y

def stage2_bdescpline_base(u):
    if u >= 0 and u <= 1:
        y = u ** 2 / 2
    elif u <= 2:
        y = -(u ** 2) + 3 * u - 1.5
    elif u <= 3:
        y = (3 - u) ** 2 / 2
    else:
        raise ValueError('input must be in the range of [0, 3]')
    return y


def main():
    drone_id = int(input('input drone id!'))
    rospy.init_node('traj' + str(drone_id) + '_exec')
    pose_publisher = rospy.Publisher('/drone_' + str(drone_id) + '/pos', Odometry, queue_size=1)
    visual_publisher = rospy.Publisher('/drone_' + str(drone_id) + '_odom_visualization/robot', Marker, queue_size=1)
    global_map = rospy.wait_for_message('/map_generator/global_cloud', PointCloud2)
    global_map = np.frombuffer(global_map.data, dtype=np.float32)
    cloud = np.reshape(global_map, [int(global_map.shape[0] / 4), 4])  
    cloud = cloud[:, :3] * 4
    cloud = np.floor(cloud).astype(np.int32)  
    cloud = ((cloud[:, 0] + 300) * 14400 + (cloud[:, 1] + 300) * 24 + np.clip(cloud[:, 2], 0, 23)).astype(np.int32)
    grid_map1 = np.zeros([600 * 600 * 24])
    grid_map1[cloud] = 1
    grid_map1 = grid_map1.reshape([600, 600, 24]).astype(np.int16)
    grid_map = np.zeros([600, 600, 48])
    for i in range(24):
        grid_map[:, :, 2 * i] = grid_map1[:, :, i]
        grid_map[:, :, 2 * i + 1] = grid_map[:, :, i]
    while(True):
        drone_pos = np.array([random.uniform(-25, 25), -30, random.uniform(1, 3)])
        drone_grid_pos = (np.floor(drone_pos * 4) + np.array([300, 300, 0])).astype(np.int16)
        if np.max(grid_map[drone_grid_pos[0]-2: drone_grid_pos[0]+3,
                           drone_grid_pos[1]-2: drone_grid_pos[1]+3]) == 0:
            break
    max_speed = 10.0
    max_acc = 20.0
    #max_jerk = 10

    bspline_points = np.array([])
    target_time = 0
    knots = np.array([])
    exec_knots = np.array([])
    mode = 'hover'
    current_u = int(1)

    visual_origin = Marker()
    visual_origin.header.frame_id = 'world'
    visual_origin.ns = 'mesh'
    visual_origin.type = 10
    visual_origin.pose.orientation.w = 1
    visual_origin.scale.x = 1
    visual_origin.scale.y = 1
    visual_origin.scale.z = 1
    visual_origin.color.a = 1
    visual_origin.frame_locked = False
    visual_origin.mesh_resource = "package://odom_visualization/meshes/hummingbird.mesh"
    visual_origin.mesh_use_embedded_materials = False
    current_time = time.time()
    while(True):
        target_time += 0.03
        try:
            bspline = rospy.wait_for_message('/drone_' + str(drone_id) + '_planning/bspline', Bspline, timeout=0.02)
            if len(bspline.pos_pts) == 0:
                while (True):
                    drone_pos = np.array([random.uniform(-10, 10), -30, random.uniform(1, 4)])
                    drone_grid_pos = (np.floor(drone_pos * 4) + np.array([300, 300, 0])).astype(np.int16)
                    if np.max(grid_map[drone_grid_pos[0] - 2: drone_grid_pos[0] + 3,
                              drone_grid_pos[1] - 2: drone_grid_pos[1] + 3]) == 0:
                        break
                mode = 'hover'
                print('trajectory done!')
                target_time = 0
            else:
                knots = np.array(bspline.knots)
                bspline_points = np.array([[i.x, i.y, i.z] for i in bspline.pos_pts])
                mode = 'exec'
                speed_bspline = (bspline_points[1:] - bspline_points[:-1]) / knots[:, np.newaxis]
                acc_bspline = (speed_bspline[1:] - speed_bspline[:-1]) / (knots[1:])[:, np.newaxis]
                jerk_bspline = (acc_bspline[1:] - acc_bspline[:-1]) / (knots[2:])[:, np.newaxis]
                exec_knots = [0]
                for i in range(2, len(knots)):
                    exec_knots.append(exec_knots[-1] + knots[i])
                current_u = int(1)
                print('get trajectory')
                target_time = min(bspline.yaw_dt, 0.1)
                while(True):
                    if target_time > exec_knots[current_u]:
                        current_u += 1
                    else:
                        break


        except Exception as e:
            #print(e)
            pass

        if mode == 'exec':
            print(time.time())
            try:
                if target_time > exec_knots[current_u]:
                    current_u += 1
                t = target_time - exec_knots[current_u - 1]
                position_controls = bspline_points[current_u - 1: current_u + 3]
                speed_controls = speed_bspline[current_u - 1: current_u + 2]
                acc_controls = acc_bspline[current_u - 1: current_u + 1]
                jerk = jerk_bspline[current_u - 1]
                pos_weights = np.array([[stage3_bdescpline_base(3 + t / 10),
                                            stage3_bdescpline_base(2 + t / 10),
                                            stage3_bdescpline_base(1 + t / 10),
                                            stage3_bdescpline_base(t / 10)]])
                drone_pos = np.matmul(pos_weights, position_controls)[0]
                drone_grid_pos = (np.floor(drone_pos * 4) + np.array([300, 300, 0])).astype(np.int16)
                if np.max(grid_map[drone_grid_pos[0], drone_grid_pos[1], drone_grid_pos[2]]) == 1:
                    mode = 'hover'
                else:
                    '''
                    if jerk > max_jerk:
                        mode = 'hover'
                    else:
                    '''
                    acc_weights = np.array([[1 - t, t]])
                    acc = np.matmul(acc_weights, acc_controls)[0]
                    if np.sum(acc[:2] ** 2) ** 0.5 > max_acc:
                        mode = 'hover'
                    else:
                        speed_weights = np.array([[stage2_bdescpline_base(2 + t / 10),
                                                   stage2_bdescpline_base(1 + t / 10),
                                                   stage2_bdescpline_base(t / 10)]])
                        speed = np.matmul(speed_weights, speed_controls)[0]
                        if np.sum(speed[:2]** 2) ** 0.5 > max_speed:
                            mode = 'hover'

            except:
                mode = 'hover'
                print('hover!')
        if mode == 'hover':
            speed = np.zeros(3)
            acc = np.zeros(3)

        position = Odometry()
        visual_position = copy.deepcopy(visual_origin)
        position.header.frame_id = mode
        position.pose.pose.position.x = drone_pos[0]
        position.pose.pose.position.y = drone_pos[1]
        position.pose.pose.position.z = drone_pos[2]
        position.pose.pose.orientation.w = 1
        position.twist.twist.linear.x = speed[0]
        position.twist.twist.linear.y = speed[1]
        position.twist.twist.linear.z = speed[2]
        position.twist.twist.angular.x = acc[0]
        position.twist.twist.angular.y = acc[1]
        position.twist.twist.angular.z = acc[2]
        pose_publisher.publish(position)

        visual_position.pose.position.x =drone_pos[0]
        visual_position.pose.position.y = drone_pos[1]
        visual_position.pose.position.z = drone_pos[2]
        visual_publisher.publish(visual_position)
        while(True):
            if time.time() - current_time > 0.08:
                current_time = time.time()
                break



if __name__ == '__main__':
    main()

















