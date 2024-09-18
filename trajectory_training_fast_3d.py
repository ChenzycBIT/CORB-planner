import numpy as np
from RLmethod_accurate_3d import drone_RL_agent_SDCQ
import matplotlib.pyplot as plt
import torch
import copy
import time
import random
import os
import rospy
import airsim
from sensor_msgs.msg import PointCloud2
from traj_utils.msg import Bspline
from visualization_msgs.msg import Marker
from Astar_3Dsearch import *
from nav_msgs.msg import Odometry

def main():
    drone_id = input('please in put drone id')
    rospy.init_node('drone' + str(drone_id) + 'RL_agent')

    momentum = 0.92
    momentum_list = np.array([i for i in range(100)])
    momentum_list = momentum ** momentum_list
    momentum_list = np.concatenate([np.flip(momentum_list[1:], axis=0), momentum_list], axis=0)
    sumweight_matrix = np.zeros([1000, 1000])
    totalweight_matrix = [np.zeros(200), np.zeros(400), np.zeros(600), np.zeros(800), np.zeros(1000)]
    for i in range(1000):
        current_list = momentum_list[max(0, i - 99) - i + 99: min(1000, i + 100) - i + 99]
        sumweight_matrix[i, max(0, i - 99): min(1000, i + 100)] = current_list
        totalweight_matrix[4][i] = np.sum(current_list)
        if i < 800:
            totalweight_matrix[3][i] = np.sum(sumweight_matrix[i, :800])
        if i < 600:
            totalweight_matrix[2][i] = np.sum(sumweight_matrix[i, :600])
        if i < 400:
            totalweight_matrix[1][i] = np.sum(sumweight_matrix[i, :400])
        if i < 200:
            totalweight_matrix[0][i] = np.sum(sumweight_matrix[i, :200])
    datas = []
    agent = drone_RL_agent_SDCQ(drone_id)
    # ---agent.load(4)
    max_episode = 1000000
    global_steps = int(1e6)
    max_traj_step = 100
    all_ep_r = []
    all_ep_r1 = []
    accsteps = []
    wsteps = 5000
    previous_traincount = 10 * agent.bc
    for i_episode in range(max_episode):
        for publish_its in range(10):
            agent.publisher.publish(Bspline())
            time.sleep(0.006)
        time.sleep(0.5)
        while (True):
            drone_target = np.array([random.uniform(-10, 10), 32, random.uniform(1, 4)])
            drone_target_grid = (np.floor(drone_target * 4) + np.array([300, 300, 0])).astype(np.int16)
            if np.max(agent.global_map[drone_target_grid[0] - 2: drone_target_grid[0] + 3,
                      drone_target_grid[1] - 2: drone_target_grid[1] + 3]) == 0:
                break
        ep_r = 0
        forwards = []
        current_time = time.time()
        for t in range(max_traj_step):
            while(time.time() - current_time < 1):
                continue
            current_time = time.time()
            try:
                current_drone_target = drone_target
                current_drone_target[2] = drone_target[2] + random.uniform(-0.15, 0.15)
                rewards, forward_reward, previous_target_index = agent.get_multi_trajectory(current_drone_target)
                if len(rewards) == 0:
                    print('trajectory plan failed!')
                    break
            except Exception as e:
                print(e)
                break
            try:
                ep_r += np.sum(np.array(rewards))
            except:
                ep_r += 0
                print('rewards exception')
            forwards.append(forward_reward)
            if agent.replaybuffer.size > 10 * agent.bc:
                training_count = 0
                if agent.replaybuffer.size < 32500:
                    max_train_count = min(20, agent.replaybuffer.size - previous_traincount)
                else:
                    max_train_count = 10
                #print('training times ' + str(max_train_count))
                while (time.time() - current_time < 0.6) and training_count <= max_train_count:
                    agent.train()
                    training_count += 1
                previous_traincount = copy.deepcopy(agent.replaybuffer.size)
            try:
                drone_info = rospy.wait_for_message('drone_' + str(drone_id) + '/pos', Odometry, timeout=0.1)
                if drone_info.header.frame_id == 'hover':
                    break
            except:
                break
        if i_episode == 0:
            all_ep_r.append(ep_r)
            accsteps.append(t)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
            accsteps.append(accsteps[-1] + t)
        try:
            if len(forwards) == 0:
                forwards.append(0)
            print('episode:', i_episode, 'score:', ep_r, 'forward_score:', np.mean(np.array(forwards)), 'step:', t,
                  'max:', max(all_ep_r), 'time:', time.time())
        except Exception as e:
            print(e)
        if i_episode % 10 == 0 and i_episode > 50:
            agent.save()
        time.sleep(0.2)
        if accsteps[-1] > wsteps:
            forwards = [[], []]
            for tests in range(2):
                for publish_its in range(30):
                    agent.publisher.publish(Bspline())
                    time.sleep(0.005)
                while (True):
                    drone_target = np.array([random.uniform(-25, 25), 32, random.uniform(1, 4)])
                    drone_target_grid = (np.floor(drone_target * 4) + np.array([300, 300, 0])).astype(np.int16)
                    if np.max(agent.global_map[drone_target_grid[0] - 2: drone_target_grid[0] + 3,
                              drone_target_grid[1] - 2: drone_target_grid[1] + 3]) == 0:
                        break
                ep_r = 0
                forwards = []
                time.sleep(3)
                current_time = time.time()
                for t in range(max_traj_step):
                    while (time.time() - current_time < 1):
                        continue
                    current_time = time.time()
                    try:
                        rewards, forward_reward, previous_target_index = agent.get_multi_trajectory(drone_target)
                        if len(rewards) == 0:
                            print('trajectory plan failed!')
                            break
                    except Exception as e:
                        print(e)
                        break
                    ep_r += np.sum(np.array(rewards))
                    forwards.append(forward_reward)
                    try:
                        drone_info = rospy.wait_for_message('drone_' + str(drone_id) + '/pos', Odometry, timeout=0.1)
                        if drone_info.header.frame_id == 'hover':
                            break
                    except:
                        break
                all_ep_r1.append(float(ep_r))
                time.sleep(0.2)
            try:
                print('steps:', wsteps, 'score:', all_ep_r1[-2:],
                      'forward:', [np.mean(np.array(forwards[0])), np.mean(np.array(forwards[1]))],
                      'step:', t, 'max:', max(all_ep_r1))
            except:
                pass
            '''
            if wsteps % 2e5 == 0:
                fig, ax = plt.subplots(figsize=(9.6, 5.4))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                plt.xticks(fontsize=10)
                plt.yticks(rotation=90, fontsize=10)
                ax.set_xlabel('million steps', fontsize=10)
                ax.set_ylabel('reward', fontsize=10)
                ax.tick_params(bottom=False, top=False, left=False, right=False)
                ax.patch.set_facecolor((0.95, 0.95, 0.96))
                ax.set_xlim(0, 1)
                ax.grid(linestyle='-', linewidth=1.5, color='white')
                if len(datas) > 0:
                    data = np.array(datas)
                    result = np.matmul(data, copy.deepcopy(sumweight_matrix)) / copy.deepcopy(totalweight_matrix[4])
                    mean = np.mean(result, axis=0)
                    maxium = np.max(result, axis=0)
                    minium = np.min(result, axis=0)
                    ax.plot([0.001 * (i + 1) for i in range(1000)], mean, linewidth=1.5, color='C0')
                    ax.fill_between([0.001 * (i + 1) for i in range(1000)], maxium, minium, color='C0', alpha=0.1)
                show = np.array(all_ep_r1)
                show = show.reshape([1, -1])


                current_result = np.matmul(show, copy.deepcopy(sumweight_matrix)[:show.shape[1], :show.shape[1]]) / \
                                 copy.deepcopy(
                                     totalweight_matrix)[int((wsteps - 2e5) / 2e5)]
                current_result = current_result[0, :show.shape[1]]
                ax.plot([0.001 * testimes for testimes in range(show.shape[1])], current_result, linewidth=1.5,
                        color='C1')
                plt.title('Drone-RL-trajplan', fontsize=13)
                plt.legend(['mean', 'current'], fontsize=10)
                plt.savefig('results/' + 'DroneRL-trajplan' + str(int(wsteps % 2e5)) + '.png')
                plt.cla()
                plt.close()
            if wsteps >= global_steps:
                datas.append(all_ep_r1)
                with open('RLplan.txt', 'a') as f:
                    f.write('\n')
                    f.write('testing reward:' + str(all_ep_r1))
                    f.write('\n')
                break
            '''
            wsteps += 5000

def get_global_path (bridge, target):
    try:
        plan_map = rospy.wait_for_message('/airsim/plan_map', Image, timeout=1)
        plan_map = bridge.imgmsg_to_cv2(plan_map)
        plan_map = np.reshape(plan_map, [600, 600, 6])
        plan_map = np.sum(plan_map, axis=2)[np.newaxis]
        plan_map1 = copy.deepcopy(plan_map)
        for i in range(2):
            plan_map_ = np.concatenate([plan_map1[:, 0: -2, 0: -2],
                                        plan_map1[:, 0: -2, 1: -1],
                                        plan_map1[:, 0: -2, 2:],
                                        plan_map1[:, 1: -1, 0: -2],
                                        plan_map1[:, 1: -1, 1: -1],
                                        plan_map1[:, 1: -1, 2:],
                                        plan_map1[:, 2:, 0: -2],
                                        plan_map1[:, 2:, 1: -1],
                                        plan_map1[:, 2:, 2:]], axis=0)
            plan_map_ = np.clip(np.sum(plan_map_, axis=0), 0, 1).astype(np.int16)
            plan_map1[0, 1: -1, 1: -1] = plan_map_
        plan_map1 = plan_map1[0]
        plan_map1[299: 302, 299: 302] = 0
        init = np.array([300, 300]).astype(np.int16)
        target = (target + 300).astype(np.int16)
        target = target[:2]
        if np.max(target) >= 600 or np.min(target) <= 0:
            raise ValueError('INVALID TARGET POSITION')
        try:
            traj_list = a_search_2d_normal(init, target, plan_map1)
            traj_list = traj_list - np.array([300, 300])
            traj_list = traj_list[:int(traj_list.shape[0] // 8 * 8)]
            traj_list = np.reshape(traj_list, [int(traj_list.shape[0] // 8), 8, 2])
            traj_list = traj_list[:, -1]
            traj_list = np.append(traj_list, np.array([target - np.array([300, 300])]), axis=0)
        except Exception as e:
            print(e)
            traj_list = np.array([target - np.array([300, 300])])
        return traj_list
    except Exception as e:
        print(e)
        traj_list = np.array([target])
        return traj_list


if __name__ == '__main__':
    main()