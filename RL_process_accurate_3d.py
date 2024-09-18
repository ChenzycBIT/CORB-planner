# -*- coding:utf-8 -*-
import numpy as np
from Astar_3Dsearch import *
import copy
import math

map_resolution = 1
init_size = [61, 61, 31]
expand_size = [63, 63, 33]
map_center = [31, 31, 16]
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

def cloud_process(pos, target, cloud):
    cloud = np.frombuffer(cloud, dtype=np.float32)
    cloud = np.reshape(cloud, [int(cloud.shape[0] / 4), 4])
    cloud = cloud[:, :3] * 3
    cloud = np.floor(cloud).astype(np.int32)
    grid_pos = np.floor(pos * 3).astype(np.int32)
    cloud = cloud - np.array([grid_pos])
    x1bound = cloud[:, 0] >= -30
    x2bound = cloud[:, 0] <= 30
    y1bound = cloud[:, 1] >= -30
    y2bound = cloud[:, 1] <= 30
    z1bound = cloud[:, 2] >= -15
    z2bound = cloud[:, 2] <= 15
    cloud_bound = x1bound * x2bound * y1bound * y2bound * z1bound * z2bound#点云范围限制

    cloud_index = np.where(cloud_bound == 1)
    cloud = cloud[cloud_index]
    cloud = ((cloud[:, 0] + 30) * 1891 + (cloud[:, 1] + 30) * 31 + (cloud[:, 2] + 15)).astype(np.int32)
    grid_map = np.zeros([61 * 61 * 31])
    grid_map[cloud] = 1
    grid_map = grid_map.reshape([61, 61, 31])
    grid_map[:, :, 2:] = np.clip(grid_map[:, :, 2:] + grid_map[:, :, 1: -1] + grid_map[:, :, :-2], 0, 1).astype(
        np.int32)  # occupancy grid map 生成

    grid_map = np.concatenate([np.zeros([1, 61, 31]), grid_map, np.zeros([1, 61, 31])], axis=0)
    grid_map = np.concatenate([np.zeros([63, 1, 31]), grid_map, np.zeros([63, 1, 31])], axis=1)
    grid_map = np.concatenate([np.zeros([63, 63, 1]), grid_map, np.zeros([63, 63, 1])], axis=2)#地图填充一圈0
    if grid_pos[2] <= 0:
        raise ValueError('drone too low!')
    try:
        if grid_pos[2] < 18:
            grid_map[:, :, :18 - grid_pos[2]] = 1
    except:
        pass
    if grid_pos[2] >= 36:
        raise ValueError('drone too high!')
    try:
        if grid_pos[2] > 22:
            grid_map[:, :, 36 - grid_pos[2]:] = 1
    except:
        pass
    if target[2] < 1:
        target[2] = 1
    grid_target = np.floor(target * 3).astype(np.int16)
    relative_target = grid_target - grid_pos
    try:
        if grid_map[relative_target[0] + 31, relative_target[1] + 31, relative_target[2] + 16] != 0:
            print('obstacles appeared at the target! we assume that there are nothing')
            grid_map[relative_target[0] + 31, relative_target[1] + 31, relative_target[2] + 16] = 0
    except:
        pass
    return grid_map

def Astar_process(start, end, map):
    if start[2] == end[2]:  
        expansion = 8
        box_xmin = max(0, min(start[0], end[0]) - expansion)
        box_xmax = min(41, max(start[0], end[0]) + expansion + 1)
        box_ymin = max(0, min(start[1], end[1]) - expansion)
        box_ymax = min(41, max(start[1], end[1]) + expansion + 1)

        xmin_plus = int(max(0, expansion - min(start[0], end[0])))
        ymin_plus = int(max(0, expansion - min(start[1], end[1])))
        xmax_plus = int(max(0, max(start[0], end[0]) + expansion - 40))
        ymax_plus = int(max(0, max(start[1], end[1]) + expansion - 40))

        Astar_map = map[box_xmin: box_xmax, box_ymin: box_ymax, start[2]]
        Astar_map = np.concatenate([np.zeros([xmin_plus, Astar_map.shape[1]]),
                                    Astar_map,
                                    np.zeros([xmax_plus, Astar_map.shape[1]])], axis=0)
        Astar_map = np.concatenate([np.zeros([Astar_map.shape[0], ymin_plus]),
                                    Astar_map,
                                    np.zeros([Astar_map.shape[0], ymax_plus])], axis=1)

        try:
            Astar_trajectory_base = np.array([[min(start[0], end[0]) - expansion,
                                               min(start[1], end[1]) - expansion]]).astype(np.int16)
            base_start = start[:2] - Astar_trajectory_base[0]
            base_end = end[:2] - Astar_trajectory_base[0]
            Astar_trajectory_list = fastsearch_2d(base_start, base_end, Astar_map)
            Astar_trajectory_list += Astar_trajectory_base
            Astar_trajectory_list = np.concatenate(
                [Astar_trajectory_list, np.ones([Astar_trajectory_list.shape[0], 1]) * start[2]], axis=1)
            return Astar_trajectory_list
        except Exception as e:
            print(e)
            pass
    elif np.abs(start[2] - end[2]) <= 3:
        expansion = 8
        box_xmin = max(0, min(start[0], end[0]) - expansion)
        box_xmax = min(41, max(start[0], end[0]) + expansion + 1)
        box_ymin = max(0, min(start[1], end[1]) - expansion)
        box_ymax = min(41, max(start[1], end[1]) + expansion + 1)

        Astar_map = map[box_xmin: box_xmax, box_ymin: box_ymax, min(start[2], end[2]): max(start[2], end[2])]
        Astar_map = np.sum(Astar_map, axis=2)
        Astar_map = np.clip(Astar_map, 0, 1).astype(np.int16)
        xmin_plus = int(max(0, expansion - min(start[0], end[0])))
        ymin_plus = int(max(0, expansion - min(start[1], end[1])))
        xmax_plus = int(max(0, max(start[0], end[0]) + expansion - 40))
        ymax_plus = int(max(0, max(start[1], end[1]) + expansion - 40))
        Astar_map = np.concatenate([np.zeros([xmin_plus, Astar_map.shape[1]]),
                                    Astar_map,
                                    np.zeros([xmax_plus, Astar_map.shape[1]])], axis=0)
        Astar_map = np.concatenate([np.zeros([Astar_map.shape[0], ymin_plus]),
                                    Astar_map,
                                    np.zeros([Astar_map.shape[0], ymax_plus])], axis=1)
        try:
            Astar_trajectory_base = np.array([[min(start[0], end[0]) - expansion,
                                               min(start[1], end[1]) - expansion]]).astype(np.int16)
            base_start = start[:2] - Astar_trajectory_base[0]
            base_end = end[:2] - Astar_trajectory_base[0]
            Astar_trajectory_list = fastsearch_2d(base_start, base_end, Astar_map)
            Astar_trajectory_list += Astar_trajectory_base
            z_list = (np.array([i for i in range(Astar_trajectory_list.shape[0])]) * (end[2] - start[2]) /
                      Astar_trajectory_list.shape[0] + start[2])[:, np.newaxis]
            Astar_trajectory_list = np.concatenate([Astar_trajectory_list, z_list], axis=1)
            return Astar_trajectory_list
        except Exception as e:
            print(e)
            pass
    raise ValueError('3d time too long')

def Astar_process_3d(start, end, map):
    expansion = 8
    box_xmin = max(0, min(start[0], end[0]) - expansion)
    box_xmax = min(41, max(start[0], end[0]) + expansion + 1)
    box_ymin = max(0, min(start[1], end[1]) - expansion)
    box_ymax = min(41, max(start[1], end[1]) + expansion + 1)
    box_zmin = max(0, min(start[2], end[2]) - 2)
    box_zmax = min(21, max(start[2], end[2]) + 3)

    Astar_map = map[box_xmin: box_xmax, box_ymin: box_ymax, box_zmin: box_zmax]
    Astar_map = np.clip(Astar_map, 0, 1).astype(np.int16)
    xmin_plus = int(max(0, expansion - min(start[0], end[0])))
    ymin_plus = int(max(0, expansion - min(start[1], end[1])))
    xmax_plus = int(max(0, max(start[0], end[0]) + expansion - 40))
    ymax_plus = int(max(0, max(start[1], end[1]) + expansion - 40))
    Astar_map = np.concatenate([np.zeros([xmin_plus, Astar_map.shape[1], Astar_map.shape[2]]),
                                Astar_map,
                                np.zeros([xmax_plus, Astar_map.shape[1], Astar_map.shape[2]])], axis=0)
    Astar_map = np.concatenate([np.zeros([Astar_map.shape[0], ymin_plus, Astar_map.shape[2]]),
                                Astar_map,
                                np.zeros([Astar_map.shape[0], ymax_plus, Astar_map.shape[2]])], axis=1)
    try:
        Astar_trajectory_base = np.array([[min(start[0], end[0]) - expansion,
                                           min(start[1], end[1]) - expansion,
                                           box_zmin]]).astype(np.int16)
        base_start = start - Astar_trajectory_base[0]
        base_end = end - Astar_trajectory_base[0]
        Astar_trajectory_list = fastsearch_3d(base_start, base_end, Astar_map)
        Astar_trajectory_list += Astar_trajectory_base
        return Astar_trajectory_list
    except Exception as e:
        print(e)
        pass


def out_obstacle(grid_map):
    path = []
    x = 100
    for i in range(1, 5):
        if grid_map[31 + i, 31, 16] == 0:
            x = i
            break
    x_ = 100
    for i in range(1, 5):
        if grid_map[31 - i, 31, 16] == 0:
            x_ = i
            break
    y = 100
    for i in range(1, 5):
        if grid_map[31, 31 + i ,16] == 0:
            y = i
            break
    y_ = 100
    for i in range(1, 5):
        if grid_map[31, 31 - i, 16] == 0:
            y_ = i
    if np.min(np.array([x, x_, y, y_])) == 100:
        for i in range(1, 5):
            if grid_map[31, 31, 16 + i] == 0:
                for j in range(i):
                    path.append(np.array([31, 31, 16 + i]))
                return path
        raise ValueError('drone out of obstacle failed!')
    else:
        if x <= x_ and x <= y and x <= y_:
            for j in range(x):
                path.append(np.array([31 + j, 31, 16]))
            return path
        elif x_ <= y and x_ <= y_:
            for j in range(x_):
                path.append(np.array([31 - j, 31, 16]))
            return path
        elif y <= y_:
            for j in range(y):
                path.append(np.array([31, 31 + j, 16]))
            return path
        else:
            for j in range(y_):
                path.append(np.array([21, 21 - j, 13]))
            return path


def find_path(pos, target, vel, grid_map):
    if (target[2] - pos[2] > 0.8):
        target[2] = pos[2] + 0.8
    elif (target[2] - pos[2] < -0.8):
        target[2] = pos[2] - 0.8
    astar_sign = 0
    grid_map[0] = 0
    grid_map[-1] = 0
    grid_map[:, 0] = 0
    grid_map[:, -1] = 0
    path = np.array([[20, 20, 10]]).astype(np.float64)
    obstacles = [0]
    pos1 = pos * 4 
    pos1[2] *= 2
    pos_base = pos1 % 1 
    path[0] += pos_base
    grid_map[19: 22, 20, 10] = 0

    target = target * 4
    target[2] *= 2
    relative_target = (target - pos1) 
    grid_target = path[0] + relative_target
    target_direction = grid_target - path[-1]

    grid_targetz = relative_target[2] + 10  
    #(1)
    path_resolution = (np.sum(target_direction[:2] ** 2) ** 0.5).astype(np.int16)  
    if path_resolution < 2:
        raise ValueError('success! reached the target')
    target_direction = target_direction[:2] / path_resolution
    vel_z_bound = int(np.floor(np.abs(vel[2] / 0.5)))
    z_move = - vel_z_bound
    if (vel[2] > 0) and (np.floor(grid_targetz) > np.floor(path[-1, 2])):
        z_move = 1
    elif (vel[2] < 0) and (np.floor(grid_targetz) < np.floor(path[-1, 2])):
        z_move = 1
    for grids in range(path_resolution):
        path_position = np.zeros(3)
        path_position[:2] = path[-1, :2] + target_direction
        if np.floor(grid_targetz) > np.floor(path[-1, 2]):
            if z_move > 0.5:
                path_position[-1] = path[-1, 2] + 1
                z_move = 0
            else:
                z_move += 1
                path_position[-1] = path[-1, 2]
        elif np.floor(grid_targetz) < np.floor(path[-1, 2]):
            if z_move > 0.5:
                path_position[-1] = path[-1, 2] - 1
                z_move = 0
            else:
                z_move += 1
                path_position[-1] = path[-1, 2]
        else:
            path_position[-1] = path[-1, 2]
        position_floor = np.floor(path_position).astype(np.int16)
        try:
            if grid_map[position_floor[0], position_floor[1], position_floor[2]] == 0:
                obstacles.append(0)
            else:
                obstacles.append(1)
        except:
            break
        path = np.append(path, path_position[np.newaxis], axis=0)
        if len(path) >= 30:  
            break
    obstacles = np.array(obstacles)
    obstacles1 = copy.deepcopy(obstacles)
    for i in range(obstacles.shape[0]):
        if obstacles[i] == 1:
            obstacles1[max(1, i - 4): min(obstacles.shape[0] - 1, i + 4)] = 1
    obstacles = obstacles1.astype(np.int16)
    #print(2)
    obstacles[0] = 0
    obstacles[-1] = 0
    obstacle_range = []
    for i in range(1, len(obstacles)):
        if obstacles[i] == 1:
            if obstacles[i - 1] == 0:
                obstacle_range.append([i - 1])
        else:
            try:
                if len(obstacle_range[-1]) == 1:
                    obstacle_range[-1].append(i)
            except:
                pass
    if len(obstacle_range) == 0:
        if len(obstacles) < 30:
            path = np.concatenate([path, np.repeat(path[-1:], 30 - len(obstacles), axis=0)], axis=0)
    else:
        if len(obstacle_range) > 1:
            for i in reversed(range(1, len(obstacle_range))):
                if obstacle_range[i][0] - obstacle_range[i-1][1] <= 3:
                    obstacle_range[i-1][1] = obstacle_range[i][1]
                    obstacle_range.pop(i)
        if len(obstacle_range[-1]) == 1:
            obstacle_range[-1].append(29)
        a_paths = []
        for slice in obstacle_range:
            ini_position = np.floor(path[slice[0]]).astype(np.int16)
            target_position = np.floor(path[slice[1]]).astype(np.int16)
            a_path = Astar_process_3d(ini_position, target_position, grid_map) + 0.5
            astar_sign = 1
            a_paths.append(a_path)
        path_slices = []
        slice_base = 0
        for slice in obstacle_range:
            path_slices.append(path[slice_base: slice[0]])
            slice_base = slice[1] + 1
        all_path = []
        for i in range(len(a_paths)):
            all_path.append(path_slices[i])
            all_path.append(a_paths[i])
        if slice_base < len(path):
            all_path.append(path[slice_base: len(path)])
        path = np.concatenate(all_path, axis=0)
    path_distances = path[1:] - path[: -1]
    path_distances = np.sum(path_distances ** 2, axis=1) ** 0.5
    eq_path = [path[0]]
    path_index = 0.0001
    #print(3)
    while (True):
        index_append = 1 / path_distances[int(np.floor(path_index))]
        if (path_index % 1) + index_append > 1:
            distance_remain = 1 - (1 - (path_index % 1)) * path_distances[int(np.floor(path_index))]
            if path_distances[int(np.ceil(path_index))] < 0.05:
                break
            path_index = np.ceil(path_index) + distance_remain / path_distances[int(np.ceil(path_index))]
        else:
            path_index = path_index + index_append
        if path_index >= (path.shape[0] - 2):
            break
        point_position = path[int(np.floor(path_index))] + ((path[int(np.floor(path_index)) + 1])
                                                            - path[int(np.floor(path_index))]) * (path_index % 1)
        eq_path.append(point_position)
        if np.min(point_position) < 0 or np.max(point_position) > 41:
            break
    if len(eq_path) > 30:
        eq_path = eq_path[:30]
        eq_path = np.array(eq_path)
    else:
        eq_path_append = np.repeat(eq_path[-1:], 30 - len(eq_path), axis=0)
        eq_path = np.concatenate([eq_path, eq_path_append], axis=0)

    #print(4)
    err_maxs = []
    for i in range(len(eq_path) - 1):
        ini_point = eq_path[i]
        target_point = eq_path[i + 1]
        if np.sum(np.abs(target_point - ini_point)) < 0.05:
            break
        ini_grid = np.floor(ini_point).astype(np.int16)
        target_grid = np.floor(target_point).astype(np.int16)
        area_minx = int(min(ini_grid[0] - 2, target_grid[0] - 2))
        area_minx = np.clip(area_minx, 0, 40)
        area_maxx = int(max(ini_grid[0] + 3, target_grid[0] + 3))
        area_maxx = np.clip(area_maxx, 1, 41)
        area_miny = int(min(ini_grid[1] - 2, target_grid[1] - 2))
        area_miny = np.clip(area_miny, 0, 40)
        area_maxy = int(max(ini_grid[1] + 3, target_grid[1] + 3))
        area_maxy = np.clip(area_maxy, 1, 41)
        area_minz = int(min(ini_grid[2], target_grid[2]))
        area_minz = np.clip(area_minz, 0, 20)
        area_maxz = int(max(ini_grid[2], target_grid[2]) + 1)
        area_maxz = np.clip(area_maxz, 1, 21)
        # area_bias = np.array([area_minx, area_miny])
        map_area = grid_map[area_minx: area_maxx, area_miny: area_maxy, area_minz: area_maxz]
        if np.max(map_area) == 0:
            err_maxs.append([3, 3])
            continue
        map_area = np.clip(np.sum(map_area, axis=2), 0, 1)
        obs_positionx = np.array([[i] for i in range(area_minx, area_maxx)])
        obs_positionx = obs_positionx[:, np.newaxis]
        obs_positionx = np.repeat(obs_positionx, (area_maxy - area_miny), axis=1)
        obs_positiony = np.repeat(np.array([[i] for i in range(area_miny, area_maxy)])[np.newaxis],
                                  (area_maxx - area_minx), axis=0)
        obs_positions = np.concatenate([obs_positionx, obs_positiony], axis=2) + 0.5
        point1_diserr = obs_positions - ini_point[np.newaxis, np.newaxis, :2]
        point1_diserr = (np.sum(point1_diserr ** 2, axis=2)) ** 0.5
        point2_diserr = obs_positions - target_point[np.newaxis, np.newaxis, :2]
        point2_diserr = (np.sum(point2_diserr ** 2, axis=2)) ** 0.5
        point_comparison = point1_diserr < point2_diserr
        point_diserr = point_comparison * point1_diserr + (1 - point_comparison) * point2_diserr
        path_dir = math.atan2(target_point[1] - ini_point[1], target_point[0] - ini_point[0])
        dis_dir = (path_dir + (np.pi / 2)) % (2 * np.pi)
        disx = np.cos(dis_dir)
        disy = np.sin(dis_dir)
        parax = np.cos(path_dir)
        paray = np.sin(path_dir)
        traj_value = disx * ini_point[0] + disy * ini_point[1]
        traj_distance = obs_positions[:, :, 0] * disx + obs_positions[:, :, 1] * disy - traj_value
        positives = traj_distance > 0
        para_start = parax * ini_point[0] + paray * ini_point[1]
        para_end = parax * target_point[0] + paray * target_point[1]
        para_length = para_start - para_end
        para_locations = (obs_positions[:, :, 0] * parax + obs_positions[:, :, 1] * paray - para_start) / para_length
        locate_in = (para_locations > 0) * (para_locations < 1)
        traj_distance = locate_in * np.abs(traj_distance) + (1 - locate_in) * point_diserr + (1 - map_area) * 3
        positive_distance = positives * traj_distance + 3 * (1 - positives)
        nagative_distance = (1 - positives) * traj_distance + 3 * positives
        positive_min = max(np.min(positive_distance), 0.4)
        nagative_min = max(np.min(nagative_distance), 0.4)
        err_maxs.append([positive_min, nagative_min])
    #print(5)
    err_maxs = np.array(err_maxs)
    err_max_append = 30 - len(err_maxs)
    err_maxs = np.concatenate([err_maxs, np.repeat(err_maxs[-1:], err_max_append, axis=0)], axis=0)
    err_maxs = err_maxs / 4
    eq_path = (eq_path - eq_path[:1]) / 4
    eq_path[:, 2] /= 2
    return eq_path, err_maxs, astar_sign



def initial_point_generation(pos, vel, acc, knots=[1/6, 1/6]):
    #check pass
    #Assume that delta_t = 0.5
    x_array = np.array([pos[0], vel[0], acc[0]])
    y_array = np.array([pos[1], vel[1], acc[1]])
    z_array = np.array([pos[2], vel[2], acc[2]])
    weight_matrix = [[1/6, 4/6, 1/6],
                     [-1/(2*knots[0]), (1/(2*knots[0]))-(1/(2*knots[1])), 1/(2 * knots[1])],
                     [1/(knots[0]*knots[1]), -(1/(knots[0]*knots[1]))-(1/(knots[1]**2)), 1/(knots[1]**2)]]
    x_out = np.linalg.solve(weight_matrix, x_array)
    y_out = np.linalg.solve(weight_matrix, y_array)
    z_out = np.linalg.solve(weight_matrix, z_array)
    points = np.array([x_out, y_out, z_out])
    points = np.swapaxes(points, 0, 1)
    return points


weight_matrix = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
                          [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
total_weight_matrix = np.zeros([50, 6])
total_weight_matrix[0: 10, 0: 2] = weight_matrix
total_weight_matrix[10: 20, 1: 3] = weight_matrix
total_weight_matrix[20: 30, 2: 4] = weight_matrix
total_weight_matrix[30: 40, 3: 5] = weight_matrix
total_weight_matrix[40: 50, 4: 6] = weight_matrix
k_j = 0.01
k_a = 0.0
k_forward = 12
k_traj = 3
max_jerk = 80
max_z_jerk = 50
def get_reward(path, max_err, points, previous_forward, jerk_point, acc_point, delta_t, plan_time):
    xy_jerk = (jerk_point[0] ** 2 + jerk_point[1] ** 2) ** 0.5
    z_jerk = jerk_point[2] * 2
    if (xy_jerk > max_jerk or z_jerk > max_z_jerk):
        jerk_done = (max(xy_jerk, max_jerk) - max_jerk) * 1.5 + (max(z_jerk, max_z_jerk) - max_z_jerk) * 1.5 + 20
    else:
        jerk_done = 0
    if (max(xy_jerk, z_jerk) > (max_jerk / 2)):
        k_forward1 = k_forward * ((max_jerk / 2) ** 2 - (max(xy_jerk, z_jerk) -
                                                         (max_jerk / 2)) ** 2) / ((max_jerk / 2) ** 2)
    else:
        k_forward1 = copy.deepcopy(k_forward)
    if k_forward1 < 2:
        k_forward1 = 2
    '''
    acc_point[2] *= 2
    acc_reward = -k_a * (np.sum(np.abs(acc_point[:2]) ** 2.5) + np.abs(acc_point[2])) * delta_t
    '''
    position_weights = np.array([[1 / 6, 4 / 6, 1 / 6]])
    current_position = np.matmul(position_weights, points[-3:])

    previous_forward_point = int(np.floor(previous_forward))

    path_slice = path[previous_forward_point: min(30, previous_forward_point + 10)]#reference trajectory slice
    err_slice = max_err[previous_forward_point: min(30, previous_forward_point + 10)]
    if path_slice.shape[0] < 10:
        path_slice = np.concatenate([path_slice, path_slice[:-1].repeat(10 - path_slice.shape[0], axis=0)], axis=0)
        err_slice = np.concatenate([err_slice, err_slice[:-1].repeat(10 - err_slice.shape[0], axis=0)], axis=0)
    path_points = np.matmul(total_weight_matrix, path_slice[:6])
    delta_traj = np.abs(path_points - current_position)
    delta_traj = np.sum(delta_traj ** 2, axis=1)
    current_forward = np.argmin(delta_traj)
    current_forward = previous_forward_point + current_forward / 10
    current_forward = max(current_forward, previous_forward)
    if current_forward >= 30:
         current_forward = 30
    path_state = path[int(np.floor(current_forward)):]
    err_state = max_err[int(np.floor(current_forward)):]
    if path_state.shape[0] < 30:
        path_state = np.concatenate([path_state, path_state[-1:]
                                    .repeat(30 - path_state.shape[0], axis=0)], axis=0).reshape(90)
        err_state = np.concatenate([err_state, err_state[-1:]
                                    .repeat(29 - err_state.shape[0], axis=0)], axis=0).reshape(58)
    else:
        path_state = path_state[:30]
        err_state = err_state[:29]
    path_state = path_state.reshape(90)
    err_state = err_state.reshape(58)
    forward_reward = k_forward1 * (current_forward - previous_forward)# +3 * min(current_forward - previous_forward, 0.3))
    forward_reward = forward_reward / delta_t
    if current_forward - previous_forward > 0:
        forward_reward += k_forward
    if np.max(np.abs(path_state[6: 9] - path_state[3: 6])) <= 0.01:
        done = True
        reward = 60 + (1.5 - float(plan_time.detach().cpu())) * 150 - jerk_done + forward_reward
        if jerk_done > 0:
            info = "reach the target, jerk too high!"
        else:
            info = "reach the target"
        return reward, path_state, err_state, current_forward, done, info

    traj_reward = []
    for point_index in range(1, 11):
        weights = np.array([[stage3_bdescpline_base((point_index / 10) + 3),
                             stage3_bdescpline_base((point_index / 10) + 2),
                             stage3_bdescpline_base((point_index / 10) + 1),
                             stage3_bdescpline_base((point_index / 10))]])
        traj_point = np.matmul(weights, points)
        traj_delta = path_points - traj_point
        traj_delta_xy = np.sum(traj_delta[:, :2] ** 2, axis=1) ** 0.5
        traj_delta_z = np.abs(traj_delta[:, 2])

        traj_point = traj_point[0]
        xy_minindex = np.argmin(traj_delta_xy)
        err_slice_index = int(xy_minindex // 10)
        if err_slice_index == 5:
            err_slice_index = 4
        current_err = err_slice[err_slice_index]
        ini_point = path_slice[err_slice_index]
        target_point = path_slice[err_slice_index + 1]
        path_dir = math.atan2(target_point[1] - ini_point[1], target_point[0] - ini_point[0])
        dis_dir = (path_dir + (np.pi / 2)) % (2 * np.pi)
        disx = np.cos(dis_dir)
        disy = np.sin(dis_dir)
        parax = np.cos(path_dir)
        paray = np.sin(path_dir)

        traj_value = disx * ini_point[0] + disy * ini_point[1]
        traj_distance = traj_point[0] * disx + traj_point[1] * disy - traj_value
        if traj_distance > 0:
            bias_max = current_err[0]
        else:
            bias_max = current_err[1]

        para_start = parax * ini_point[0] + paray * ini_point[1]
        para_end = parax * target_point[0] + paray * target_point[1]
        para_length = para_end - para_start
        if para_length < 0.01:
            bias = ((traj_point[0] - ini_point[0]) ** 2 + (traj_point[1] - ini_point[1]) ** 2) ** 0.5
        else:
            para_location = (traj_point[0] * parax + traj_point[1] * paray - para_start) / para_length
            if para_location < 0:
                bias = ((traj_point[0] - ini_point[0]) ** 2 + (traj_point[1] - ini_point[1]) ** 2) ** 0.5
            elif para_location > 1:
                bias = ((traj_point[0] - target_point[0]) ** 2 + (traj_point[1] - target_point[1]) ** 2) ** 0.5
            else:
                bias = np.abs(traj_distance)

        delta_z = np.abs(traj_delta_z[xy_minindex])
        if bias > bias_max or delta_z > 0.25:
            reward = - 50 - jerk_done

            done = True
            if jerk_done > 0:
                if delta_z > 0.35:
                    info = "traj crash z, jerk too high!"
                else:
                    info = "traj crash, jerk too high!"
            else:
                if delta_z > 0.35:
                    info = "traj crash z"
                else:
                    info = "traj crash"
            return reward, path_state, err_state, previous_forward, done, info
        traj_reward.append((0.1 - np.clip(bias_max - bias - 0.1, -0.1, 0) - delta_z) * delta_t)

    traj_reward = np.array(traj_reward)
    reward = np.sum(traj_reward) * k_traj - jerk_done + 3
    try:
        reward = np.array(reward)
    except:
        a = 1
    if delta_t > 0.5:
        done = True
        reward -= 50
        if jerk_done > 0:
            info = "delta t too high, jerk too high!"
        else:
            info = "delta t too high"
        return reward, path_state, err_state, current_forward, done, info

    reward += forward_reward
    try:
        reward = np.array(reward)
    except:
        a = 1
    done = False
    if plan_time >= 2:
        done = True
        if jerk_done > 0:
            info = "trajectory finish, jerk too high!"
        else:
            if current_forward > 1:
                reward += current_forward * k_forward
            info = "trajectory finish"
    elif jerk_done > 0:
        done = True
        info = "jerk too high!"
    else:
        info = 'continue'
    return reward, path_state, err_state, current_forward, done, info


def gridmap_process(grid_map, pos, target):
    grid_map = copy.deepcopy(grid_map)
    for its in range(3):
        occs = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 2 and j == 2:
                    occs.append(copy.deepcopy(grid_map[np.newaxis, 4:, 4:]))
                elif i == 2:
                    occs.append(copy.deepcopy(grid_map[np.newaxis, 4: , 2 + j: j - 2]))
                elif j == 2:
                    occs.append(copy.deepcopy(grid_map[np.newaxis, 2 + i: i - 2, 4:]))
                else:
                    occs.append(copy.deepcopy(grid_map[np.newaxis, 2 + i: i - 2, 2 + j: j - 2]))
        del occs[24]
        del occs[20]
        del occs[4]
        del occs[0]
        occs = np.clip(np.sum(np.concatenate(occs, axis=0), axis=0), 0, 1).astype(np.int16)
        occ_map_center = copy.deepcopy(grid_map[30: 33, 30: 33, 14: 17])
        grid_map[2: -2, 2: -2] = occs
        if np.max(occ_map_center) == 1:
            grid_map[30: 33, 30: 33, 14: 17] = occ_map_center
            break
    occs = []
    occs.append(copy.deepcopy(grid_map[np.newaxis, :, :, 0: -2]))
    occs.append(copy.deepcopy(grid_map[np.newaxis, :, :, 1: -1]))
    occs.append(copy.deepcopy(grid_map[np.newaxis, :, :, 2:]))
    occs = np.concatenate(occs, axis=0)
    occs = np.clip(np.sum(occs, axis=0), 0, 1).astype(np.int16)
    occ_map_center = copy.deepcopy(grid_map[30: 33, 30: 33, 14: 17])
    grid_map[:, :, 1: -1] = occs
    if np.max(occ_map_center) == 1:
        grid_map[30: 33, 30: 33, 14: 17] = occ_map_center
    grid_map = np.concatenate([np.zeros([1, 61, 30]), grid_map, np.zeros([1, 61, 30])], axis=0)
    grid_map = np.concatenate([np.zeros([63, 1, 30]), grid_map, np.zeros([63, 1, 30])], axis=1)
    grid_map = np.concatenate([np.zeros([63, 63, 2]), grid_map, np.zeros([63, 63, 1])], axis=2)  
    grid_pos = np.floor(pos * 3).astype(np.int32)
    if grid_pos[2] <= 0:
        raise ValueError('drone too low!')
    try:
        if grid_pos[2] < 18:
            grid_map[:, :, :min(14, 18 - grid_pos[2])] = 1  
    except:
        pass
    if grid_pos[2] >= 36:  
        raise ValueError('drone too high!')
    try:
        if grid_pos[2] > 22:
            grid_map[:, :, 36 - grid_pos[2]:] = 1
    except:
        pass
    if target[2] < 1:
        target[2] = 1
    grid_target = np.floor(target * 3).astype(np.int16)
    relative_target = grid_target - grid_pos
    try:
        if grid_map[relative_target[0] + 31, relative_target[1] + 31, relative_target[2] + 16] != 0:
            print('obstacles appeared at the target! we assume that there are nothing')
            grid_map[relative_target[0] + 31, relative_target[1] + 31, relative_target[2] + 16] = 0 
    except:
        pass
    return grid_map


























