# -*- coding:utf-8 -*-

import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch.distributions import Normal
from torch import optim
import random
import time
import itertools
import torch.nn as nn
from RL_process_accurate_3d import *
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from traj_utils.msg import Bspline

bridge = CvBridge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, max_size_exp=int(16)):
        self.max_size = int(2 ** max_size_exp)
        self.ptr = 0
        self.size = 0

        self.grid_map = np.zeros((self.max_size, 4, 45))
        self.err_max = np.zeros((self.max_size, 4, 28))
        self.point_state = np.zeros((self.max_size, 4, 12))
        self.action = np.zeros((self.max_size, 9))
        self.reward = np.zeros((self.max_size, 3))
        self.prob = np.zeros((self.max_size, 3))
        self.dead = np.zeros(self.max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, sample):
        self.grid_map[self.ptr] = sample[0]
        self.err_max[self.ptr] = sample[1]
        self.point_state[self.ptr] = sample[2]
        self.action[self.ptr] = sample[3]
        self.prob[self.ptr] = sample[4]
        self.reward[self.ptr] = sample[5]
        self.dead[self.ptr] = sample[6]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return(torch.FloatTensor(self.grid_map[ind]).to(self.device),
               torch.FloatTensor(self.err_max[ind]).to(self.device),
               torch.FloatTensor(self.point_state[ind]).to(self.device),
               torch.FloatTensor(self.action[ind]).to(self.device),
               torch.FloatTensor(self.prob[ind]).to(self.device),
               torch.FloatTensor(self.reward[ind]).to(self.device),
               torch.FloatTensor(self.dead[ind]).to(self.device))



class DeepQnetwork(nn.Module):
    def __init__(self, state_dim, actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, actions)
        self.relu = nn.LeakyReLU()

    def forward(self, path, errs, points):
        state = torch.cat([path, errs, points], dim=1)
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class SDCQ_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc01 = nn.Linear(state_dim + action_dim, 256)
        self.fc02 = nn.Linear(256, 256)
        self.fc03 = nn.Linear(256, 1)

        self.fc11 = nn.Linear(state_dim + action_dim, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, path, errs, points, action):
        inputs = torch.cat([path, errs, points, action], dim=1)

        out1 = self.fc01(inputs)
        out1 = self.relu(out1)
        out1 = self.fc02(out1)
        out1 = self.relu(out1)
        out1 = self.fc03(out1)

        out2 = self.fc11(inputs)
        out2 = self.relu(out2)
        out2 = self.fc12(out2)
        out2 = self.relu(out2)
        out2 = self.fc13(out2)
        return out1, out2

    def q1(self, path, points, action):
        inputs = torch.cat([path, points, action], dim=1)
        out1 = self.fc01(inputs)
        out1 = self.relu(out1)
        out1 = self.fc02(out1)
        out1 = self.relu(out1)
        out1 = self.fc03(out1)

        return out1



class drone_RL_agent_SDCQ():
    def __init__(self, drone_id):

        #3 4
        # 3.5 5
        # 4.5 7.5
        # 5 8
        # 8 12
        # 10 15
        self.max_speed = 4
        self.max_acceleration = 6
        self.max_jerk = 100.0
        self.replaybuffer = ReplayBuffer(int(15))
        self.basic_interval = 1/6

        self.action_dim = 3
        self.action_slices = 60

        self.action_slice = 2 / self.action_slices
        self.action_base = (1 / self.action_slices) - 1

        self.discrete_net = DeepQnetwork(85, self.action_dim * self.action_slices).to(device)
        self.d_optimizer = torch.optim.Adam(self.discrete_net.parameters(), lr=3e-4)


        self.continuous_net = SDCQ_critic(85, 3).to(device)
        self.c_optimizer = torch.optim.Adam(self.continuous_net.parameters(), lr=3e-4)
        self.target_continuous = copy.deepcopy(self.continuous_net)

        self.log_alpha = torch.tensor(1.0).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = 0
        self.target_logalpha = copy.deepcopy(self.log_alpha).detach()


        self.gamma = 0.97
        self.bc = int(256)
        self.tau = 5e-3
        self.loss = torch.nn.MSELoss()
        self.publisher = rospy.Publisher('/drone_' + str(drone_id) + '_planning/bspline', Bspline, queue_size=10)
        self.astar_publisher = rospy.Publisher('/drone_' + str(drone_id) + '_ego_planner_node/init_list', Marker,
                                               queue_size=10)
        self.rviztraj_publisher = rospy.Publisher('/drone_' + str(drone_id) + '_ego_planner_node/optimal_list', Marker,
                                                  queue_size=10)
        self.training_steps = 0
        self.eps = 1
        self.importance_min = -2
        self.importance_max = 2
        self.prior_alpha = 0

        self.complete_count = 0
        self.fail_count = 1

        self.previous_trajectory = None
        self.previous_deltas = None
        self.plan_count = 0
        self.max_plancount = 5
        self.publish_fail = 0

        global_map = rospy.wait_for_message('/map_generator/global_cloud', PointCloud2)
        global_map = np.frombuffer(global_map.data, dtype=np.float32)
        cloud = np.reshape(global_map, [int(global_map.shape[0] / 4), 4])  
        cloud = cloud[:, :3] * 4
        cloud = np.floor(cloud).astype(np.int32) 

        cloud = ((cloud[:, 0] + 300) * 14400 + (cloud[:, 1] + 300) * 24 + np.clip(cloud[:, 2], 0, 23)).astype(np.int32)
        grid_map = np.zeros([600 * 600 * 24])
        grid_map[cloud] = 1
        grid_map = grid_map.reshape([600, 600, 24])
        self.global_map = np.zeros([600, 600, 48])
        for i in range(24):
            self.global_map[:, :, 2 * i] = grid_map[:, :, i]
            self.global_map[:, :, 2 * i + 1] = grid_map[:, :, i]
        self.drone_id = drone_id

    def select_action(self, path, err, points, greedy=False):
        path = path[:45].unsqueeze(0)
        err = err[:28].unsqueeze(0)
        points = points.unsqueeze(0)
        alpha = torch.exp(torch.clamp(self.log_alpha, -10, 10)).clone()
        q = self.discrete_net(path, err, points)
        #print('path:' + str(path.detach().cpu().numpy()))
        #print('err:' + str(err.detach().cpu().numpy()))
        #print('points:' + str(points.detach().cpu().numpy()))
        #print('q1:' + str(q[0, 0:30].detach().cpu().numpy()))
        #print('q2:' + str(q[0, 30:60].detach().cpu().numpy()))
        #print('q3:' + str(q[0, 60:90].detach().cpu().numpy()))
        qvalue = q.reshape([3, self.action_slices]) / alpha
        qvalue_max, _ = torch.max(qvalue, dim=1)
        qvalue -= qvalue_max.unsqueeze(1)
        qvalue = torch.exp(qvalue)
        distribution = qvalue / (torch.sum(qvalue, dim=1).unsqueeze(1))
        _, max_index = torch.max(distribution, dim=1)
        max_index = max_index.detach().cpu().numpy()
        #max_action = max_index * self.action_slice + self.action_base7
        if not greedy:
            distribution1 = distribution.clone().detach().cpu().numpy()
            action0 = np.random.choice([i for i in range(self.action_slices)], 1, p=distribution1[0])
            action1 = np.random.choice([i for i in range(self.action_slices)], 1, p=distribution1[1])
            action2 = np.random.choice([i for i in range(self.action_slices)], 1, p=distribution1[2])
            action_index = np.concatenate([action0, action1, action2], axis=0)

            action_select = (np.ones(3) * self.eps) < np.random.uniform(0, 1, [3])
            action_index = action_select * max_index + (1 - action_select) * action_index
            #action = action_index * self.action_slice + self.action_base
            action_hot = np.eye(self.action_slices)[action_index]
            prob = np.sum(distribution1 * action_hot, axis=1)
            prob = np.exp(np.sum(np.log(prob + 1e-7)))
        else:
            #action = max_action
            action_index = max_index
            action_hot = np.eye(self.action_slices)[max_index]
            distribution = distribution.detach().cpu().numpy()
            prob = np.sum(distribution * action_hot, axis=1)
            prob = np.exp(np.sum(np.log(prob + 1e-7)))

        return action_index, prob


    def traj_generation(self, path, points, max_err, ini_deltas=[1/6, 1/6], greedy=False):
        deltas = torch.tensor(ini_deltas).to(device)  # first two deltas

        current_forward = 0
        experiences = {'path': [],
                       'max_err': [],
                       'point_state': [],
                       'index': [],
                       'prob': [],
                       'reward': []}
        path_state = copy.deepcopy(path).reshape(90)
        err_state = copy.deepcopy(max_err[:29]).reshape(58)
        experiences['path'].append(path_state[:45])
        experiences['max_err'].append(err_state[:28])
        plan_time = torch.FloatTensor([0]).to(device)
        experiences['point_state'].append(np.concatenate([points.reshape(9), ini_deltas, [0]], axis=0))
        points = torch.FloatTensor(points).to(device)
        for traj_steps in range(1, 100):
            action_index, prob = self.select_action(torch.FloatTensor(path_state).to(device),
                                                                 torch.FloatTensor(err_state).to(device),
                                                                 torch.cat([points[-3:].reshape(9), deltas[-2:],
                                                                            plan_time], dim=0), greedy=greedy)
            experiences['index'].append(action_index)
            experiences['prob'].append(prob)
            new_point, delta_t, jerk_point, acc_point = self.action_bound(points[-3:].detach().cpu().numpy(),
                                                                          action_index,
                                                                          deltas[-2:].detach().cpu().numpy())
            plan_time[0] += delta_t
            points = torch.cat([points, torch.FloatTensor(new_point).to(device).unsqueeze(0)], dim=0)
            deltas = torch.cat([deltas, torch.FloatTensor([delta_t]).to(device)], dim=0)
            current_points = points[-4:].detach().cpu().numpy()
            reward, path_state, err_state, current_forward, done, info = get_reward(path, max_err, current_points,
                                                                                    current_forward, jerk_point,
                                                                                    acc_point, delta_t,
                                                                                    plan_time[0])
            reward = reward * 0.1
            experiences['path'].append(path_state[:45])
            experiences['max_err'].append(err_state[:28])
            experiences['reward'].append(reward)
            experiences['point_state'].append(torch.cat([points[-3:].reshape(9), deltas[-2:], plan_time]
                                                        , dim=0).detach().cpu().numpy())
            if done:
                print("traj_steps: " + str(traj_steps) + ", done_info: " + str(info) + ", current_forward: " + str(current_forward))
                break
        return points, experiences, deltas, current_forward, info, traj_steps

    def to_buffer(self, experiences):
        paths = np.array(experiences['path'])
        paths = np.concatenate([paths, paths[-1:], paths[-1:]], axis=0)
        max_errs = np.array(experiences['max_err'])
        max_errs = np.concatenate([max_errs, max_errs[-1:], max_errs[-1:]], axis=0)
        indices = np.array(experiences['index'])
        indices = np.concatenate([indices, indices[-1:], indices[-1:]], axis=0)
        probs = np.array(experiences['prob'])
        probs = np.concatenate([probs, probs[-1:], probs[-1:]], axis=0)
        rewards = np.array(experiences['reward'])
        rewards = np.concatenate([rewards, np.zeros(2)], axis=0)
        point_states = np.array(experiences['point_state'])
        point_states = np.concatenate([point_states, point_states[-1:], point_states[-1:]], axis=0)
        dones = np.concatenate([np.zeros(paths.shape[0] - 4), np.array([1, 2, 3])], axis=0)
        if paths.shape[0] == 4:
            if random.uniform(0, 1) < 0.5:
                self.replaybuffer.add([paths,
                                       max_errs,
                                       point_states,
                                       indices.reshape(9),
                                       probs,
                                       rewards,
                                       3])
        else:
            for i in range(indices.shape[0] - 2):
                '''
                if dones[i + 3] == 3:
                    if random.uniform(0, 1) < 0.6:
                        self.replaybuffer.add([paths[i: i + 4],
                                               max_errs[i: i + 4],
                                               point_states[i: i + 4],
                                               indices[i: i + 3].reshape(9),
                                               probs[i: i + 3],
                                               rewards[i: i + 3],
                                               dones[i + 3]])
                else:
                '''
                self.replaybuffer.add([paths[i: i + 4],
                                       max_errs[i: i + 4],
                                       point_states[i: i + 4],
                                       indices[i: i + 3].reshape(9),
                                       probs[i: i + 3],
                                       rewards[i: i + 3],
                                       dones[i + 2]])

    def get_multi_trajectory(self, target):
        start_time = time.time()
        while(True):
            try:
                pose = rospy.wait_for_message('drone_' + str(self.drone_id) + '/pos', Odometry, timeout=0.15)
                break
            except:
                print('drone' + str(self.drone_id) + 'get pose failed!')
        pos = np.array([pose.pose.pose.position.x,
               pose.pose.pose.position.y,
               pose.pose.pose.position.z])
        vel = np.array([pose.twist.twist.linear.x + random.uniform(-0.2, 0.2),
               pose.twist.twist.linear.y + random.uniform(-0.2, 0.2),
               pose.twist.twist.linear.z + random.uniform(-0.1, 0.1)])
        speed_xy = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
        if np.abs(speed_xy) > self.max_speed:
            vel[:2] = vel[:2] * self.max_speed / np.abs(speed_xy)
        acc = np.array([pose.twist.twist.angular.x + random.uniform(-1, 1),
               pose.twist.twist.angular.y + random.uniform(-1, 1),
               pose.twist.twist.angular.z + random.uniform(-0.5, 0.5)])

        target_direction = (target - pos) / (np.sum((target - pos) ** 2) ** 0.5)
        target_arc = math.atan2(target_direction[1], target_direction[0])

        grid_pos = (np.floor(pos * 4) + np.array([300, 300, 0])).astype(np.int16)
        grid_pos[2] = np.floor(pos[2] * 8).astype(np.int16)
        bottom_append = np.ones([41, 41, max(0, 10 - grid_pos[2])])
        top_append = np.ones([41, 41, max(0, 10 - (47 - grid_pos[2]))])
        try:
            grid_map = self.global_map[grid_pos[0] - 20: grid_pos[0] + 21,
                                       grid_pos[1] - 20: grid_pos[1] + 21,
                                       max(grid_pos[2] - 10, 0): min(grid_pos[2] + 11, 48)]
        except:
            end_sign = Bspline()
            self.publisher.publish(end_sign)
            ''' return done !!!!!  '''
        grid_map = np.concatenate([bottom_append, grid_map, top_append], axis=2)
        grid_map[1: -1] = np.clip(np.sum(np.concatenate([(grid_map[:-2])[np.newaxis],
                                                         (grid_map[1: -1])[np.newaxis],
                                                         (grid_map[2:])[np.newaxis]], axis=0), axis=0),
                                                          0, 1).astype(np.int16)
        grid_map[:, 1: -1] = np.clip(np.sum(np.concatenate([(grid_map[:, :-2])[np.newaxis],
                                                            (grid_map[:, 1: -1])[np.newaxis],
                                                            (grid_map[:, 2:])[np.newaxis]], axis=0), axis=0),
                                                             0, 1).astype(np.int16)
        grid_map[:, :, 1: -1] = np.clip(np.sum(np.concatenate([(grid_map[:, :, :-2])[np.newaxis],
                                                               (grid_map[:, :, 1: -1])[np.newaxis],
                                                               (grid_map[:, :, 2:])[np.newaxis]],
                                                                axis=0), axis=0), 0, 1).astype(np.int16)#膨胀
        try:
            path, max_errs, astar_sign = find_path(pos, target, vel, grid_map)
        except Exception as e:
            print('find path exception!')
            print(e)
            return [], 0, 0

        if random.uniform(0, 1) < 0.5:
            ini_knots = [self.basic_interval, self.basic_interval]
        else:
            ini_knots = [random.uniform(self.basic_interval, 0.25), random.uniform(self.basic_interval, 0.25)]
        points = initial_point_generation(np.zeros(3), vel, acc, ini_knots)  # points tensor[3, 3]
        try:
            forward_path = np.zeros_like(path)
            forward_path[:, 0] = path[:, 0] * np.cos(target_arc) + path[:, 1] * np.sin(target_arc)
            forward_path[:, 1] = path[:, 0] * np.cos(target_arc + np.pi / 2) + path[:, 1] * np.sin(target_arc + np.pi / 2)
            forward_path[:, 2] = path[:, 2]
            forward_points = np.zeros_like(points)
            forward_points[:, 0] = points[:, 0] * np.cos(target_arc) + points[:, 1] * np.sin(target_arc)
            forward_points[:, 1] = points[:, 0] * np.cos(target_arc + np.pi / 2) + points[:, 1] * np.sin(
                target_arc + np.pi / 2)
            forward_points[:, 2] = points[:, 2]
            traj_points1, experiences, deltas, current_forward, info, traj_steps \
                        = self.traj_generation(forward_path, forward_points, max_errs, ini_knots, greedy=True)
        except Exception as e:
            print('path generation exception!')
            print(e)
            return [], 0, 0


        traj_points1 = traj_points1.detach().cpu().numpy()
        traj_points = copy.deepcopy(traj_points1)
        traj_points[:, 0] = traj_points1[:, 0] * np.cos(-target_arc) + traj_points1[:, 1] * np.sin(-target_arc)
        traj_points[:, 1] = traj_points1[:, 0] * np.cos(-target_arc + np.pi / 2) + traj_points1[:, 1] * np.sin(-target_arc + np.pi / 2)
        traj_points = traj_points + pos[np.newaxis]
        deltas = deltas.detach().cpu().numpy()
        traj_points = np.concatenate([traj_points, traj_points[-1:], traj_points[-1:]], axis=0)
        deltas = np.concatenate([deltas, deltas[-1:], deltas[-1:]], axis=0)
        trajectory = Bspline()
        trajectory.drone_id = 0
        trajectory.knots = deltas.tolist()
        trajectory.start_time.secs = int(rospy.get_time() // 1)
        trajectory.order = 3
        trajectory.pos_pts = []
        for i in traj_points:
            point_now = Point()
            point_now.x = i[0]
            point_now.y = i[1]
            point_now.z = i[2]
            trajectory.pos_pts.append(point_now)
        delta_time = time.time() - start_time
        trajectory.yaw_dt = delta_time
        if traj_points.shape[0] >= 6:
            for i in range(10):
                self.publisher.publish(trajectory)
                time.sleep(0.006)
        visual_path = copy.deepcopy(path) + pos[np.newaxis]
        rviz_path = Marker()
        rviz_path.header.frame_id = 'world'
        rviz_path.color.a = 1
        rviz_path.scale.x = 0.1
        rviz_path.type = 4
        rviz_path.pose.orientation.w = 1
        rviz_bspline = copy.deepcopy(rviz_path)
        rviz_path.color.r = 1
        rviz_bspline.color.b = 1
        for i in visual_path:
            point_rviz = Point()
            point_rviz.x = i[0]
            point_rviz.y = i[1]
            point_rviz.z = i[2]
            rviz_path.points.append(point_rviz)
        self.astar_publisher.publish(rviz_path)
        for i in range(len(traj_points) - 3):
            for j in range(10):
                current_points = traj_points[i: i + 4]
                t = j / 10
                weights = np.array([[stage3_bdescpline_base(t + 3),
                                     stage3_bdescpline_base(t + 2),
                                     stage3_bdescpline_base(t + 1),
                                     stage3_bdescpline_base(t)]])
                point_position = np.matmul(weights, current_points)[0]
                point_rviz = Point()
                point_rviz.x = point_position[0]
                point_rviz.y = point_position[1]
                point_rviz.z = point_position[2]
                rviz_bspline.points.append(point_rviz)
        self.rviztraj_publisher.publish(rviz_bspline)
        if current_forward == 0:
            if random.uniform(0, 1) < 0.05:
                if astar_sign == 1:
                    self.to_buffer(experiences)
                elif random.uniform(0, 1) < 0.3:
                    self.to_buffer(experiences)
        else:
            if astar_sign == 1:
                self.to_buffer(experiences)
            elif random.uniform(0, 1) < 0.3:
                self.to_buffer(experiences)
        extras = 0
        non_onesetps = 0
        while(time.time() - start_time) < 0.35:
            _, extra_experiences, _, traj_forward, _, traj_steps = self.traj_generation(
                                            forward_path, forward_points, max_errs, ini_knots, greedy=False)
            if traj_forward == 0 or traj_steps == 1:
                if non_onesetps == 0:
                    if random.uniform(0, 1) < 0.02:
                            self.to_buffer(extra_experiences)
                            extras += 1
                else:
                    if random.uniform(0, 1) < 0.05:
                            self.to_buffer(extra_experiences)
                            extras += 1
            else:
                non_onesetps += 1
                if astar_sign == 1:
                    self.to_buffer(extra_experiences)
                    extras += 1
                elif random.uniform(0, 1) < 0.15:
                    self.to_buffer(extra_experiences)
                    extras += 1
        #print(extras)
        if non_onesetps == 0:
            asdfg = 0
        return experiences['reward'], current_forward, time.time() - start_time


    def train(self):
        self.training_steps += 1
        with torch.no_grad():
            alpha = torch.exp(torch.clamp(self.target_logalpha, -10, 10)).clone()
            path_state, max_errs, points, a_index, prob, r,  d = self.replaybuffer.sample(self.bc)
            a1 = a_index[:, : self.action_dim]
            a = a1 * self.action_slice + self.action_base
            a2 = a_index[:, self.action_dim: 2 * self.action_dim]
            a3 = a_index[:, 2 * self.action_dim: 3 * self.action_dim]
            prob2 = torch.log(prob[:, 1] + 1e-7) / self.action_dim
            prob3 = torch.log(prob[:, 2] + 1e-7) / self.action_dim
            target_q = self.get_target_q(path_state[:, 3], max_errs[:, 3], points[:, 3], alpha)
            newprob2, entropy2 = self.get_prob(path_state[:, 1], max_errs[:, 1], points[:, 1], a2, alpha)
            newprob3, entropy3 = self.get_prob(path_state[:, 2], max_errs[:, 2], points[:, 2], a3, alpha)

            done0 = (d < 1).float()
            done1 = (d < 2).float()
            done2 = (d < 3).float()
            is2 = (newprob2 - prob2) * done2
            is3 = (newprob3 - prob3) * done1
            # importance = torch.log((is2 * is3) + 1e-7)
            # importance = importance / self.action_dim
            # importance = torch.exp(importance)
            importance = is2 + is3
            importance_mean = torch.mean(importance)
            importance_std = (torch.mean(importance ** 2) - importance_mean ** 2) ** 0.5
            importance = (importance - importance_mean) * (2 / importance_std)
            importance = torch.clamp(importance, self.importance_min, self.importance_max)


            importance = torch.exp(importance)
            importance_sum = torch.sum(importance)
            importance *= (self.bc / importance_sum)

            target_q = r[:, 0] \
                       + self.gamma * done2 * (r[:, 1] - (entropy2 * alpha)) \
                       + (self.gamma ** 2) * done1 * (r[:, 2] - (entropy3 * alpha)) \
                       + (self.gamma ** 3) * done0 * target_q

        current_q1, current_q2 = self.continuous_net(path_state[:, 0], max_errs[:, 0], points[:, 0], a)

        c_loss = (current_q1.squeeze(1) - target_q) ** 2 + (current_q2.squeeze(1) - target_q) ** 2
        c_loss = c_loss * importance
        c_loss_mean = torch.mean(c_loss)
        self.c_optimizer.zero_grad()
        c_loss_mean.backward()
        self.c_optimizer.step()

        qsa_generate_original = self.discrete_net(path_state[:, 0], max_errs[:, 0], points[:, 0])
        qsa_generate = torch.reshape(qsa_generate_original.detach().clone(),
                                     [self.bc, self.action_dim, self.action_slices])

        _, current_optimal_action_index = torch.max(qsa_generate, dim=2)
        current_optimal_action = current_optimal_action_index * self.action_slice + self.action_base
        current_optimal_action = current_optimal_action.detach().clone().unsqueeze(1)

        action_chart = []
        for i in range(self.action_dim):
            for j in range(self.action_slices):
                specified_action = current_optimal_action.detach().clone()
                specified_action[:, :, i] = self.action_base + j * self.action_slice
                action_chart.append(specified_action)

        action_chart = torch.cat(action_chart, dim=1)
        action_chart = torch.reshape(action_chart,
                                     [self.bc * self.action_slices * self.action_dim, self.action_dim])
        path_chart = path_state[:, :1].repeat(1, self.action_slices * self.action_dim, 1)
        err_chart = max_errs[:, :1].repeat(1, self.action_slices * self.action_dim, 1)
        point_chart = points[:, :1].repeat(1, self.action_slices * self.action_dim, 1)
        path_chart = torch.reshape(path_chart, [self.bc * self.action_slices * self.action_dim, 45])
        err_chart = torch.reshape(err_chart, [self.bc * self.action_slices * self.action_dim, 28])
        point_chart = torch.reshape(point_chart, [self.bc * self.action_slices * self.action_dim, 12])
        target_qsa1, target_qsa2 = self.continuous_net(path_chart, err_chart, point_chart, action_chart)
        target_qsa = torch.min(target_qsa1, target_qsa2).detach()
        target_qsa = torch.reshape(target_qsa, [self.bc, self.action_dim, self.action_slices])
        target_qsa_mean = torch.mean(target_qsa, dim=2).unsqueeze(2)
        target_qsa -= target_qsa_mean
        target_qsa = torch.reshape(target_qsa, [self.bc, self.action_dim * self.action_slices])
        qsa_loss = self.loss(qsa_generate_original, target_qsa)
        self.d_optimizer.zero_grad()
        qsa_loss.backward()
        self.d_optimizer.step()

        qsa_generate_original = qsa_generate_original.clone().detach().reshape(
            [self.bc, self.action_dim, self.action_slices])
        current_alpha = torch.exp(torch.clamp(self.log_alpha, -10, 10))
        current_alpha_copy = current_alpha.clone().detach()
        qsa = qsa_generate_original / current_alpha_copy
        qsa_max, _ = torch.max(qsa, dim=2)
        qsa_max = qsa_max.unsqueeze(2)
        qsa = qsa - qsa_max
        distribution = torch.exp(qsa)
        distribution_sum = torch.sum(distribution, dim=2).unsqueeze(2)
        distribution = distribution / distribution_sum
        log_distribution = qsa - torch.log(distribution_sum) - np.log(self.action_slice)
        entropy = torch.sum(log_distribution * distribution, dim=2).detach()
        entropy_mean = torch.mean(entropy)
        entropy_loss = current_alpha * (self.target_entropy - entropy_mean)
        self.alpha_optimizer.zero_grad()
        entropy_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.target_continuous.parameters(), self.continuous_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        self.target_logalpha = self.tau * self.log_alpha + (1 - self.tau) * self.target_logalpha

        # self.target_entropy = np.clip(self.target_entropy + self.entropy_decay, 0, self.final_entropy)

    def get_target_q(self, path4, max_err4, point4, alpha):
        target_q = self.discrete_net(path4, max_err4, point4)
        target_q = torch.reshape(target_q, [self.bc, self.action_dim, self.action_slices]) / alpha
        qvalue_max, _ = torch.max(target_q, dim=2)
        target_q -= qvalue_max.unsqueeze(2)
        target_prob = torch.exp(target_q)
        prob_sum = torch.sum(target_prob, dim=2).unsqueeze(2)
        target_logprob = target_q - torch.log(prob_sum) - np.log(self.action_slice)
        target_distribution = target_prob / prob_sum
        target_entropy = torch.sum(target_logprob * target_distribution, dim=2).detach()
        for i in range(self.action_slices - 1):
            target_distribution[:, :, i + 1] += target_distribution[:, :, i]
        target_distribution_low = torch.cat(
            [torch.zeros([self.bc, self.action_dim, 1]).to(device), target_distribution[:, :, :-1]], dim=2)
        random_generation = torch.rand([self.bc, self.action_dim, 1]).to(device)
        _, target_action_index = torch.max((random_generation > target_distribution_low).float() * (
                random_generation < target_distribution).float(), dim=2)
        target_action = target_action_index * self.action_slice + self.action_base
        target_q1, target_q2 = self.target_continuous(path4, max_err4, point4, target_action)
        target_q = torch.min(target_q1.squeeze(1), target_q2.squeeze(1)) - torch.sum(target_entropy, dim=1) * alpha
        return target_q

    def get_prob(self, path2, max_err2, point2, a2, alpha):
        mid_q1 = self.discrete_net(path2, max_err2, point2)
        mid_q1 = torch.reshape(mid_q1, [self.bc, self.action_dim, self.action_slices]) / alpha
        midq_max, _ = torch.max(mid_q1, dim=2)
        mid_q11 = mid_q1 - midq_max.unsqueeze(2)
        midq1_prob = torch.exp(mid_q11)
        prob_sum = torch.sum(midq1_prob, dim=2).unsqueeze(2)
        midq1_logprob = mid_q11 - torch.log(prob_sum) - np.log(self.action_slice)
        midq1_distribution = midq1_prob / prob_sum
        entropy1 = torch.sum(midq1_logprob * midq1_distribution, dim=2)
        entropy1 = torch.sum(entropy1, dim=1)
        a2 = torch.reshape(a2, [self.bc * self.action_dim, 1]).long()
        a_hot = torch.zeros([self.bc * self.action_dim, self.action_slices]).to(device).scatter(1, a2, 1)
        a_hot = torch.reshape(a_hot, [self.bc, self.action_dim, self.action_slices])
        action_prob = a_hot * midq1_distribution
        action_prob = torch.sum(action_prob, dim=2)
        action_prob = action_prob + 1e-7
        action_prob = torch.log(action_prob)
        log_action_prob = torch.mean(action_prob, dim=1)

        return log_action_prob, entropy1

    def save(self):#, episode):
        torch.save(self.discrete_net.state_dict(), "actor{}-{}.pth".format(self.max_speed, self.max_acceleration))
        torch.save(self.continuous_net.state_dict(), "critic{}.pth".format(self.max_speed))

    def load(self, episode):
        self.discrete_net.load_state_dict(torch.load("actor{}.pth".format(episode)))
        self.continuous_net.load_state_dict(torch.load("critic{}.pth".format(episode)))
        self.target_continuous.load_state_dict(torch.load("critic{}.pth".format(episode)))

    def map_backward(self, point_state, delta):
        point2 = point_state[2]
        point1 = point_state[1]
        speed_point = (point2 - point1) / delta
        speed_point[:2] = speed_point[:2] / self.max_speed
        speed_length = (np.sum(speed_point[:2] ** 2) ** 0.5 + 1e-7)
        speed_max_value = np.max(np.abs(speed_point[:2]) + 1e-7)
        speed_precentage = speed_length / speed_max_value
        speed_point[:2] *= speed_precentage

        return speed_point

    def action_bound(self, point_state, action_index, deltas):
        point0 = point_state[0]
        point1 = point_state[1]
        point2 = point_state[2]
        vel1 = (point1 - point0) / deltas[0]
        vel2 = (point2 - point1) / deltas[1]
        acc2 = (vel2 - vel1) / deltas[1]

        old_action = copy.deepcopy(vel2)
        old_action[:2] = old_action[:2] / self.max_speed
        old_action[2] = old_action[2] * 2 / self.max_speed
        speed_length = (np.sum(old_action[:2] ** 2) ** 0.5 + 1e-7)
        speed_max_value = np.max(np.abs(old_action[:2]) + 1e-7)
        speed_precentage = speed_length / speed_max_value
        old_action[:2] *= speed_precentage
        speed_point = (action_index * self.action_slice + self.action_base)
        if old_action[0] > 0.1:
            if speed_point[0] > 2/3:
                speed_point[0] = 2/3
        elif old_action[0] < -0.1:
            if speed_point[0] < -2/3:
                speed_point[0] = -2/3
        if old_action[1] > 0.1:
            if speed_point[1] > 2/3:
                speed_point[1] = 2/3
        elif old_action[1] < -0.1:
            if speed_point[1] < -2/3:
                speed_point[1] = -2/3
        speed_point = speed_point * self.max_acceleration * self.basic_interval / self.max_speed + old_action
        speed_point = np.clip(speed_point, -1, 1)
        #print(action_index)
        speed_direction = speed_point[:2] / (np.sum(speed_point[:2] ** 2) ** 0.5 + 1e-7)
        speed_action = copy.deepcopy(speed_point)
        speed_action[:2] = speed_direction * self.max_speed * np.max(np.abs(speed_point[:2]))
        speed_action[2] = speed_action[2] * self.max_speed / 2
        delta_vel = speed_action - vel2
        accbound_t = (np.sum(delta_vel[:2] ** 2) ** 0.5) / self.max_acceleration
        '''
        jerkbound_t = np.polynomial.Polynomial([np.sum(delta_vel[:2] ** 2),
                                                2 * np.sum(delta_vel[:2] * acc2[:2]),
                                                np.sum(acc2[:2] ** 2),
                                                0,
                                                self.max_jerk ** 2])
        jerkbound_t = jerkbound_t.roots()
        jerkbound_index = (jerkbound_t.real > 0) * (np.abs(jerkbound_t.imag) < 0.01)
        if np.max(jerkbound_index) == 0:
            jerkbound_t = 1/3
        else:
            jerkbound_t = jerkbound_t.real[np.argmax(jerkbound_index)]
        '''
        accbound_zt = np.abs(delta_vel[2]) / (self.max_acceleration / 2)
        delta_t = max(np.max(np.array([accbound_t, accbound_zt])), self.basic_interval)

        point3 = point2 + speed_action * delta_t
        acc3 = delta_vel / delta_t
        jerk_point = (acc3 - acc2) / delta_t
        #print(jerk_point)
        return point3, delta_t, jerk_point, acc3
