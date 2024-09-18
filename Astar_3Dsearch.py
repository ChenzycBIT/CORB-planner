# -*- coding:utf-8 -*-
import numpy as np
import copy
import rospy
import cv2
from sensor_msgs.msg import  Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import time

def a_search(init, target, occupancy_map):
    #输入量： initchushiweizhi
    occupancy_map = occupancy_map.astype(np.int8)
    init = np.floor(np.array(init)).astype(np.int8)
    target = np.floor(np.array(target)).astype(np.int8)
    occupancy_map[init[0], init[1], init[2]] = 0
    occupancy_map[target[0], target[1], target[2]] = 0
    init_H = np.sum(np.abs(target - init))
    if init_H == 0:
        raise ValueError('init and target can not be the same!')
    close_list = np.array([np.insert(np.insert(init, 0, init_H, axis=0), 0, 0, axis=0)]).astype(np.int8)

    neighbor_list = np.array([[1, 0, 0],
                              [-1, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0],
                              [0, 0, 1],
                              [0, 0, -1]]).astype(np.int8)

    for i in neighbor_list:
        try:
            grid = init + i
            if occupancy_map[grid[0], grid[1], grid[2]] == False:
                G = 1
                H = np.sum(np.abs(target - grid))
                try:
                    open_list = np.insert(open_list, 0, np.concatenate([np.array([G, H]), grid], axis=0), axis=0)
                except:
                    open_list = np.concatenate([np.array([G, H]), grid], axis=0)[np.newaxis]
        except:
            pass
    reach_sign = False
    for search_time in range(10000):
        if open_list.shape[0] == 0:
            raise ValueError('Unreachable!')
        optimal_index = np.argmin(np.sum(open_list[:, :2], axis=1))
        current_point = open_list[optimal_index]
        open_list = np.delete(open_list, optimal_index, axis=0)
        close_list = np.insert(close_list, 0, current_point, axis=0)
        G = current_point[0] + 1
        for i in neighbor_list:
            try:
                grid = current_point[2:] + i
                if occupancy_map[grid[0], grid[1], grid[2]] == False:

                    open_pos = open_list[:, 2:]
                    open_dist = np.sum(np.abs(open_pos - np.array([grid])), axis=1)
                    mindist_index = np.argmin(open_dist)
                    if open_dist[mindist_index] == 0:
                        if open_list[mindist_index, 0] > G:
                            open_list[mindist_index, 0] = G
                    else:
                        close_pos = close_list[:, 2:]
                        close_dist = np.sum(np.abs(close_pos - np.array([grid])), axis=1)
                        if np.min(close_dist) != 0:
                            H = np.sum(np.abs(target - grid))
                            if H == 0:
                                reach_sign = True
                                break
                            else:
                                open_point = np.concatenate([np.array([G, H]), grid], axis=0).astype(np.int8)
                                open_list = np.insert(open_list, 0, open_point, axis=0)
            except:
                pass
        if reach_sign == True:
            break
    traj_list = [target]
    for backward_time in range(100):
        current_point = traj_list[-1]
        G_list = []
        close_index = []
        for i in neighbor_list:
            grid = current_point + i
            close_pos = close_list[:, 2:]
            close_dist = np.sum(np.abs(close_pos - np.array([grid])), axis=1)
            mindist_index = np.argmin(close_dist)
            if close_dist[mindist_index] == 0:
                G_list.append(close_list[mindist_index, 0])
                close_index.append(mindist_index)
            else:
                G_list.append(1000)
                close_index.append(None)
        G_list = np.array(G_list)
        G_minindex = np.argmin(G_list)
        if G_list[G_minindex] == 1000:
            raise ValueError('Error during path backward!')
        else:
            traj_list.append(close_list[close_index[G_minindex], 2:])
        if G_list[G_minindex] == 0:
            break
    return np.flip(np.array(traj_list).astype(np.int16), axis=0)


def a_search_plus(init, target, occupancy_map):
    #输入量： initchushiweizhi
    occupancy_map = np.concatenate([np.ones([1, occupancy_map.shape[1], occupancy_map.shape[2]]), occupancy_map,
                                    np.ones([1, occupancy_map.shape[1], occupancy_map.shape[2]])], axis=0)
    occupancy_map = np.concatenate([np.ones([occupancy_map.shape[0], 1, occupancy_map.shape[2]]), occupancy_map,
                                    np.ones([occupancy_map.shape[0], 1, occupancy_map.shape[2]])], axis=1)
    occupancy_map = np.concatenate([np.ones([occupancy_map.shape[0], occupancy_map.shape[1], 1]), occupancy_map,
                                    np.ones([occupancy_map.shape[0], occupancy_map.shape[1], 1])], axis=2)
    occupancy_map = occupancy_map.astype(np.int8)
    position_x = np.array([i for i in range(occupancy_map.shape[0])])[:, np.newaxis, np.newaxis].repeat(
        occupancy_map.shape[1], axis=1).repeat(occupancy_map.shape[2], axis=2)
    position_y = np.array([i for i in range(occupancy_map.shape[1])])[np.newaxis, :, np.newaxis].repeat(
        occupancy_map.shape[0], axis=0).repeat(occupancy_map.shape[2], axis=2)
    position_z = np.array([i for i in range(occupancy_map.shape[2])])[np.newaxis, np.newaxis].repeat(
        occupancy_map.shape[0], axis=0).repeat(occupancy_map.shape[1], axis=1)
    position_map = np.concatenate([position_x[np.newaxis], position_y[np.newaxis], position_z[np.newaxis]], axis=0)
    init = np.floor(np.array(init + 1)).astype(np.int8)
    target = np.floor(np.array(target + 1)).astype(np.int8)
    occupancy_map[init[0], init[1], init[2]] = 0
    occupancy_map[target[0], target[1], target[2]] = 0
    H_map = (np.sum(np.abs(target[:, np.newaxis, np.newaxis, np.newaxis] - position_map), axis=0) * 10).astype(np.int16)
    G_map = np.zeros_like(H_map).astype(np.int16)
    close_map = np.zeros_like(H_map).astype(np.int16)
    close_map[init[0], init[1], init[2]] = 1
    G_plus = np.array([[[24, 14, 24], [24, 10, 24], [24, 14, 24]],
                       [[22, 10, 22], [20,  0, 20], [22, 10, 22]],
                       [[24, 14, 24], [22, 10, 22], [24, 14, 24]]])
    init_H = H_map[init[0], init[1], init[2]]
    if init_H == 0:
        raise ValueError('init and target can not be the same!')
    local_occ = (1 - occupancy_map[init[0] - 1: init[0] + 2,
                                  init[1] - 1: init[1] + 2,
                                  init[2] - 1: init[2] + 2])
    local_G = G_plus * local_occ
    G_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] = local_G
    y_shape = occupancy_map.shape[1]
    z_shape = occupancy_map.shape[2]
    for search_time in range(10000):
        if np.sum(G_map) == 0:
            raise ValueError('Unreachable! (3D A*)')
        GH = (G_map == 0) * 10000 + G_map + H_map
        GH_index = np.argmin(GH)
        min_index = [GH_index // (y_shape * z_shape), (GH_index % (y_shape * z_shape)) // z_shape, GH_index % z_shape]
        close_G = copy.deepcopy(G_map[min_index[0], min_index[1], min_index[2]])
        close_map[min_index[0], min_index[1], min_index[2]] = close_G
        G_map[min_index[0], min_index[1], min_index[2]] = 0
        local_G = G_plus + close_G
        local_occ = occupancy_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2,
                         min_index[2] - 1: min_index[2] + 2]
        local_H = close_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2,
                         min_index[2] - 1: min_index[2] + 2]
        local_occ = 1 - np.clip(local_occ + local_H, 0, 1)
        original_G = G_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2,
                         min_index[2] - 1: min_index[2] + 2]
        original_G = original_G + (original_G == 0) * 10000
        local_G_refresh = np.min(np.concatenate([(local_G * local_occ)[np.newaxis], original_G[np.newaxis]], axis=0),
                                 axis=0)
        G_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2,
                         min_index[2] - 1: min_index[2] + 2] = local_G_refresh
        if H_map[min_index[0], min_index[1], min_index[2]] == 0:
            break
    traj_list = [target]
    close_map = (close_map == 0) * 10000 + close_map.astype(np.int16)
    for backward_time in range(100):
        current_point = traj_list[-1]
        current_neighbor = close_map[current_point[0] - 1: current_point[0] + 2,
                         current_point[1] - 1: current_point[1] + 2,
                         current_point[2] - 1: current_point[2] + 2]
        neighbor_min = np.argmin(current_neighbor)
        point_index = [int(current_point[0] + neighbor_min // 9 - 1),
                       int(current_point[1] + (neighbor_min % 9) // 3 - 1),
                       int(current_point[2] + neighbor_min % 3) - 1]
        traj_list.append(point_index)
        if np.min(current_neighbor) == 1:
            break
    traj_list = np.array(traj_list) - 1
    return np.flip(np.array(traj_list).astype(np.int16), axis=0)


def a_search_2d(init, target, occupancy_map):
    #输入量： initchushiweizhi
    occupancy_map = np.concatenate([np.ones([1, occupancy_map.shape[1]]), occupancy_map,
                                    np.ones([1, occupancy_map.shape[1]])], axis=0)
    occupancy_map = np.concatenate([np.ones([occupancy_map.shape[0], 1]), occupancy_map,
                                    np.ones([occupancy_map.shape[0], 1])], axis=1)
    occupancy_map = occupancy_map.astype(np.int8)
    position_x = np.array([i for i in range(occupancy_map.shape[0])])[:, np.newaxis].repeat(
        occupancy_map.shape[1], axis=1)
    position_y = np.array([i for i in range(occupancy_map.shape[1])])[np.newaxis, :].repeat(
        occupancy_map.shape[0], axis=0)
    position_map = np.concatenate([position_x[np.newaxis], position_y[np.newaxis]], axis=0)
    init = np.floor(np.array(init + 1)).astype(np.int8)
    target = np.floor(np.array(target + 1)).astype(np.int8)
    occupancy_map[init[0], init[1]] = 0
    occupancy_map[target[0], target[1]] = 0
    H_map = (np.sum(np.abs(target[:, np.newaxis, np.newaxis] - position_map), axis=0) * 10).astype(np.int16)
    G_map = np.zeros_like(H_map).astype(np.int16)
    close_map = np.zeros_like(H_map).astype(np.int16)
    close_map[init[0], init[1]] = 1
    G_plus = np.array([[14, 10, 14], [10, 0, 10], [14, 10, 14]])
    init_H = H_map[init[0], init[1]]
    if init_H == 0:
        raise ValueError('init and target can not be the same!')
    local_occ = (1 - occupancy_map[init[0] - 1: init[0] + 2,
                                  init[1] - 1: init[1] + 2])
    local_G = G_plus * local_occ
    G_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2] = local_G
    y_shape = occupancy_map.shape[1]
    for search_time in range(10000):
        if np.sum(G_map) == 0:
            raise ValueError('Unreachable! (2D A*)')
        GH = (G_map == 0) * 10000 + G_map + H_map
        GH_index = np.argmin(GH)
        min_index = [GH_index // y_shape, GH_index % y_shape]
        close_G = copy.deepcopy(G_map[min_index[0], min_index[1]])
        close_map[min_index[0], min_index[1]] = close_G
        G_map[min_index[0], min_index[1]] = 0
        local_G = G_plus + close_G
        local_occ = occupancy_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2]
        local_H = close_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2]
        local_occ = 1 - np.clip(local_occ + local_H, 0, 1)
        original_G = G_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2]
        original_G = original_G + (original_G == 0) * 10000
        local_G_refresh = np.min(np.concatenate([(local_G * local_occ)[np.newaxis], original_G[np.newaxis]], axis=0),
                                 axis=0)
        G_map[min_index[0] - 1: min_index[0] + 2,
                         min_index[1] - 1: min_index[1] + 2] = local_G_refresh
        if H_map[min_index[0], min_index[1]] == 0:
            break
    traj_list = [target]
    close_map = (close_map == 0) * 10000 + close_map.astype(np.int16)
    for backward_time in range(100):
        current_point = traj_list[-1]
        current_neighbor = close_map[current_point[0] - 1: current_point[0] + 2,
                         current_point[1] - 1: current_point[1] + 2]
        neighbor_min = np.argmin(current_neighbor)
        point_index = [int(current_point[0] + neighbor_min // 3 - 1),
                       int(current_point[1] + neighbor_min % 3 - 1)]
        traj_list.append(point_index)
        if np.min(current_neighbor) == 1:
            break
    traj_list = np.array(traj_list) - 1
    return np.flip(np.array(traj_list).astype(np.int16), axis=0)


def a_search_2d_normal(init, target, occupancy_map):
    occupancy_map = occupancy_map.astype(np.int16)
    init = np.floor(np.array(init)).astype(np.int16)
    target = np.floor(np.array(target)).astype(np.int16)
    occupancy_map[init[0], init[1]] = 0
    occupancy_map[target[0], target[1]] = 0
    init_H = 10 * np.sum(np.abs(target - init))
    if init_H == 0:
        raise ValueError('init and target can not be the same!')
    close_list = np.array([np.insert(np.insert(init, 0, init_H, axis=0), 0, 0, axis=0)]).astype(np.int16)
    neighbor_list = np.array([[1, 0],
                              [-1, 0],
                              [0, 1],
                              [0, -1],
                              [1, 1],
                              [1, -1],
                              [-1, 1],
                              [-1, -1]]).astype(np.int16)

    for i in range(len(neighbor_list)):
        try:
            grid = init + neighbor_list[i]
            if occupancy_map[grid[0], grid[1]] == False:
                if i <= 3:
                    G = 10
                else:
                    G = 14
                H = np.sum(np.abs(target - grid)) * 10
                try:
                    open_list = np.insert(open_list, 0, np.concatenate([np.array([G, H]), grid], axis=0), axis=0)
                except:
                    open_list = np.concatenate([np.array([G, H]), grid], axis=0)[np.newaxis]
        except:
            pass
    reach_sign = False
    for search_time in range(36000):
        if open_list.shape[0] == 0:
            raise ValueError('Unreachable! (2D normal A*)')
        optimal_index = np.argmin(np.sum(open_list[:, :2], axis=1))
        current_point = open_list[optimal_index]
        open_list = np.delete(open_list, optimal_index, axis=0)
        close_list = np.insert(close_list, 0, current_point, axis=0)
        for i in range(len(neighbor_list)):
            if i <= 3:
                G = current_point[0] + 10
            else:
                G = current_point[0] + 14
            try:
                grid = current_point[2:] + neighbor_list[i]
                if occupancy_map[grid[0], grid[1]] == False:
                    open_pos = open_list[:, 2:]
                    open_dist = np.sum(np.abs(open_pos - np.array([grid])), axis=1)
                    mindist_index = np.argmin(open_dist)
                    if open_dist[mindist_index] == 0:
                        if open_list[mindist_index, 0] > G:
                            open_list[mindist_index, 0] = G
                    else:
                        close_pos = close_list[:, 2:]
                        close_dist = np.sum(np.abs(close_pos - np.array([grid])), axis=1)
                        if np.min(close_dist) != 0:
                            H = np.sum(np.abs(target - grid)) * 10
                            if H != 0:
                                open_point = np.concatenate([np.array([G, H]), grid], axis=0).astype(np.int16)
                                open_list = np.insert(open_list, 0, open_point, axis=0)
                if np.min(close_list[:, 1]) <= 10:
                    reach_sign = True
                    break
            except:
                pass
        if reach_sign == True:
            break
    traj_list = [target]
    for backward_time in range(2000):
        current_point = traj_list[-1]
        G_list = []
        close_index = []
        for i in neighbor_list:
            grid = current_point + i
            if grid[0] > occupancy_map.shape[0] - 1 or grid[1] > occupancy_map.shape[1] - 1 or np.min(grid) < 0:
                G_list.append(10000)
                close_index.append(None)
                continue
            close_pos = close_list[:, 2:]
            close_dist = np.sum(np.abs(close_pos - np.array([grid])), axis=1)
            mindist_index = np.argmin(close_dist)
            if close_dist[mindist_index] == 0:
                G_list.append(close_list[mindist_index, 0])
                close_index.append(mindist_index)
            else:
                G_list.append(10000)
                close_index.append(None)
        G_list = np.array(G_list)
        G_minindex = np.argmin(G_list)
        if G_list[G_minindex] == 10000:
            raise ValueError('Error during path backward!')
        else:
            traj_list.append(close_list[close_index[G_minindex], 2:])
        if G_list[G_minindex] == 0:
            break
    return np.flip(np.array(traj_list).astype(np.int16), axis=0)
G_chart = np.array([[14, 10, 14],
                        [10, 0, 10],
                        [14, 10, 14]])
position_x = np.array([i for i in range(100)])[:, np.newaxis].repeat(100, axis=1)
position_y = np.array([i for i in range(100)])[np.newaxis, :].repeat(100, axis=0)
position_map = np.concatenate([position_x[np.newaxis], position_y[np.newaxis]], axis=0)

def fastsearch_2d(init, target, occupancy_map):
    start_time = time.time()
    init = init.astype(np.int8)
    target = target.astype(np.int8)
    occupancy_map[init[0], init[1]] = 0
    occupancy_map[target[0], target[1]] = 0
    part_positionmap = position_map[:, :occupancy_map.shape[0], :occupancy_map.shape[1]]
    H_map = (np.sum(np.abs(target[:, np.newaxis, np.newaxis] - part_positionmap), axis=0) * 10).astype(np.int16)
    sign_map = np.ones_like(occupancy_map).astype(np.uint8)
    sign_map[init[0], init[1]] = 2
    sign_map[target[0], target[1]] = 3
    G_map = np.ones_like(occupancy_map).astype(np.int16) * 5000
    G_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2] = G_chart
    sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2] -= \
        (sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2] == 1)
    sign_map[-1] = 4
    sign_map[0] = 4
    sign_map[:, 0] = 4
    sign_map[:, -1] = 4
    while(True):
        G_index = np.argmin(G_map + 5000 * sign_map + H_map)
        gx_index = int(G_index // occupancy_map.shape[1])
        gy_index = int(G_index % occupancy_map.shape[1])
        a = G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2]
        if a[1, 1] > 3000:
            raise ValueError('can not find path')
        sign_map[gx_index, gy_index] = 2
        if np.min(H_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2]) == 0:
            break
        occupancy_slice = occupancy_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2]
        s_slice = sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2]
        sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2] = s_slice - ((s_slice == 1) * (1 - occupancy_slice))

        b = G_map[gx_index, gy_index] + G_chart
        replace_sign = (b < a) * (1 - occupancy_slice)
        G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2] = replace_sign * b + (1 - replace_sign) * a
        if time.time() - start_time > 0.1:
            raise ValueError('A* time too long')
    path = [target]
    while(True):
        a = G_map[path[-1][0] - 1: path[-1][0] + 2, path[-1][1] - 1: path[-1][1] + 2]
        if np.min(a) == 0:
            break
        a = np.abs(sign_map[path[-1][0] - 1: path[-1][0] + 2, path[-1][-1] - 1: path[-1][-1] + 2] - 2) * 5000 + a
        min_index = np.argmin(a)
        minx = int((min_index // 3) - 1)
        miny = int((min_index % 3) - 1)
        path.append(path[-1] + np.array([minx, miny]))
    path.append(init)
    path = np.array(path)
    path = np.flip(path, axis=0)
    print(time.time() - start_time)
    return path


G_chart_3d = np.array([[[17, 14, 17],
                        [14, 10, 14],
                        [17, 14, 17]],
                       [[14, 10, 14],
                        [100, 0, 100],
                        [14, 10, 14]],
                        [[17, 14, 17],
                        [14, 10, 14],
                        [17, 14, 17]]])
position_x_3d = np.array([i for i in range(100)])[:, np.newaxis, np.newaxis].repeat(100, axis=1).repeat(30, axis=2)
position_y_3d = np.array([i for i in range(100)])[np.newaxis, :, np.newaxis].repeat(100, axis=0).repeat(30, axis=2)
position_z_3d = np.array([i for i in range(30)])[np.newaxis, np.newaxis, :].repeat(100, axis=0).repeat(100, axis=1)
position_map_3d = np.concatenate([position_x_3d[np.newaxis],
                                  position_y_3d[np.newaxis],
                                  position_z_3d[np.newaxis]], axis=0)

def fastsearch_3d(init, target, occupancy_map):
    start_time = time.time()
    init = init.astype(np.int8)
    target = target.astype(np.int8)
    occupancy_map[init[0], init[1], init[2]] = 0
    occupancy_map[target[0], target[1], target[2]] = 0
    part_positionmap = position_map_3d[:, :occupancy_map.shape[0], :occupancy_map.shape[1], :occupancy_map.shape[2]]
    H_map = (np.sum(np.abs(target[:, np.newaxis, np.newaxis, np.newaxis] - part_positionmap), axis=0) * 10).astype(np.int16)
    sign_map = np.ones_like(occupancy_map).astype(np.uint8)
    sign_map[init[0], init[1], init[2]] = 2
    sign_map[target[0], target[1], target[2]] = 3
    G_map = np.ones_like(occupancy_map).astype(np.int16) * 5000
    G_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] = G_chart_3d
    sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] -= \
        (sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] == 1)
    sign_map[-1] = 4
    sign_map[0] = 4
    sign_map[:, 0] = 4
    sign_map[:, -1] = 4
    sign_map[:, :, 0] = 4
    sign_map[:, :, -1] = 4
    while (True):
        G_index = np.argmin(G_map + 5000 * sign_map + H_map)
        gx_index = int(G_index // (occupancy_map.shape[1] * occupancy_map.shape[2]))
        gy_index = int((G_index % (occupancy_map.shape[1] * occupancy_map.shape[2])) // occupancy_map.shape[2])
        gz_index = int(G_index % occupancy_map.shape[2])
        a = G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]
        if a[1, 1, 1] > 3000:
            raise ValueError('can not find path')
        sign_map[gx_index, gy_index, gz_index] = 2
        if np.min(H_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]) == 0:
            break
        occupancy_slice = occupancy_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2,
                          gz_index - 1: gz_index + 2]
        s_slice = sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]
        sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2] \
            = s_slice - ((s_slice == 1) * (1 - occupancy_slice))
        b = G_map[gx_index, gy_index, gz_index] + G_chart_3d
        replace_sign = (b < a) * (1 - occupancy_slice)
        G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2] \
            = replace_sign * b + (1 - replace_sign) * a
        if time.time() - start_time > 0.1:
            raise ValueError('A* time too long')
    path = [target]
    G_map[:, :, 0] += 5000
    G_map[:, :, -1] += 5000
    back_time = 0
    while(True):
        a = G_map[path[-1][0] - 1: path[-1][0] + 2, path[-1][1] - 1: path[-1][1] + 2, path[-1][2] - 1: path[-1][2] + 2]
        back_time += 1
        if np.min(a) == 0:
            break
        a = np.abs(sign_map[path[-1][0] - 1: path[-1][0] + 2,
                            path[-1][1] - 1: path[-1][1] + 2,
                            path[-1][2] - 1: path[-1][2] + 2] - 2) * 5000 + a
        min_index = np.argmin(a)
        minx = int((min_index // 9) - 1)
        miny = int(((min_index % 9) // 3) - 1)
        minz = int(min_index % 3 - 1)
        path.append(path[-1] + np.array([minx, miny, minz]))
    path.append(init)
    path = np.array(path)
    path = np.flip(path, axis=0)
    print(time.time() - start_time)
    return path


def fastsearch_3d_zexpand(init, target, occupancy_map):
    start_time = time.time()
    init = init.astype(np.int8)
    target = target.astype(np.int8)
    occupancy_map = occupancy_map[:, :, :, np.newaxis].repeat(2, axis=3)
    occupancy_map = occupancy_map.reshape([occupancy_map.shape[0], occupancy_map.shape[1], occupancy_map.shape[2] * 2])
    init[2] = init[2] * 2
    target[2] = target[2] * 2
    occupancy_map[init[0], init[1], init[2]] = 0
    occupancy_map[target[0], target[1], target[2]] = 0
    part_positionmap = position_map_3d[:, :occupancy_map.shape[0], :occupancy_map.shape[1], :occupancy_map.shape[2]]
    H_map = (np.sum(np.abs(target[:, np.newaxis, np.newaxis, np.newaxis] - part_positionmap), axis=0) * 10).astype(np.int16)
    sign_map = np.ones_like(occupancy_map).astype(np.uint8)
    sign_map[init[0], init[1], init[2]] = 2
    sign_map[target[0], target[1], target[2]] = 3
    G_map = np.ones_like(occupancy_map).astype(np.int16) * 5000
    G_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] = G_chart_3d
    sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] -= \
        (sign_map[init[0] - 1: init[0] + 2, init[1] - 1: init[1] + 2, init[2] - 1: init[2] + 2] == 1)
    sign_map[-1] = 4
    sign_map[0] = 4
    sign_map[:, 0] = 4
    sign_map[:, -1] = 4
    sign_map[:, :, 0] = 4
    sign_map[:, :, -1] = 4

    while (True):
        G_index = np.argmin(G_map + 5000 * sign_map + H_map)
        gx_index = int(G_index // (occupancy_map.shape[1] * occupancy_map.shape[2]))
        gy_index = int((G_index % (occupancy_map.shape[1] * occupancy_map.shape[2])) // occupancy_map.shape[2])
        gz_index = int(G_index % occupancy_map.shape[2])
        a = G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]
        if a[1, 1, 1] > 3000:
            raise ValueError('can not find path')
        sign_map[gx_index, gy_index, gz_index] = 2
        if np.min(H_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]) == 0:
            break
        occupancy_slice = occupancy_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2,
                          gz_index - 1: gz_index + 2]
        s_slice = sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2]
        sign_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2] \
            = s_slice - ((s_slice == 1) * (1 - occupancy_slice))
        b = G_map[gx_index, gy_index, gz_index] + G_chart_3d
        replace_sign = (b < a) * (1 - occupancy_slice)
        G_map[gx_index - 1: gx_index + 2, gy_index - 1: gy_index + 2, gz_index - 1: gz_index + 2] \
            = replace_sign * b + (1 - replace_sign) * a
        if time.time() - start_time > 0.1:
            raise ValueError('A* time too long')
    path = [target]
    G_map[:, :, 0] += 5000
    G_map[:, :, -1] += 5000
    back_time = 0
    while(True):
        a = G_map[path[-1][0] - 1: path[-1][0] + 2, path[-1][1] - 1: path[-1][1] + 2, path[-1][2] - 1: path[-1][2] + 2]
        back_time += 1
        if np.min(a) == 0:
            break
        a = np.abs(sign_map[path[-1][0] - 1: path[-1][0] + 2,
                            path[-1][1] - 1: path[-1][1] + 2,
                            path[-1][2] - 1: path[-1][2] + 2] - 2) * 5000 + a
        min_index = np.argmin(a)
        minx = int((min_index // 9) - 1)
        miny = int(((min_index % 9) // 3) - 1)
        minz = int(min_index % 3 - 1)
        path.append(path[-1] + np.array([minx, miny, minz]))
    path.append(init)
    path = np.array(path)
    path = np.flip(path, axis=0)
    path[:, 2] = path[:, 2] / 2
    print(time.time() - start_time)
    return path




