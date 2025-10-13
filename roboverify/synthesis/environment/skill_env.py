import gymnasium as gym
import numpy as np
import os
import math

import matplotlib.pyplot as plt

from modules.reward import RewIdentity, RewDebug, RewCritic
from utils.logging import init_logging, log_and_print

import pdb

def common_skill_reward(reward, cost, rew_fun, lagrangian):
    if not isinstance(rew_fun, RewIdentity):
        reward -= 0
        if lagrangian:
            if reward >= 0 and cost <= 0:
                reward += 1
        elif reward >= 0:
            reward += 1
    else:
        if reward == -1:
            reward = -1
        elif reward == 0:
            reward = 1
        elif reward == 1:
            reward = 1
        else:
            print(reward)
            pdb.set_trace()
            raise NotImplementedError
        
    return reward

# disable reward add
def common_skill_reward_noadd(reward, cost, rew_fun, lagrangian):
    if not isinstance(rew_fun, RewIdentity):
        reward -= 0
    else:
        if reward == -1:
            reward = -1
        elif reward == 0:
            reward = 1
        elif reward == 1:
            reward = 1
        else:
            print(reward)
            pdb.set_trace()
            raise NotImplementedError
        
    return reward

def block_skill_reward(reward, cost, rew_fun, lagrangian):
    if not isinstance(rew_fun, RewIdentity):
        # only for debug
        reward -= 0
        if lagrangian:
            if reward >= 0 and cost <= 0:
                reward += 1
        elif reward >= 0:
            reward += 1
    else:
        if reward == -1:
            reward = -1
        elif reward == 0:
            reward = 0
        elif reward == 1:
            reward = 1
        else:
            print(reward)
            pdb.set_trace()
            raise NotImplementedError
        
    return reward

class SkillEnv(gym.Wrapper):
    def __init__(self, env, skill_graph, search_node, rew_fun, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, \
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None, \
                 env_debug=False, new_observation_space=None, node_policy_map=None, obs_transit=None, debug_rew_fun=None):
        super().__init__(env)
        self.attemp_num = 1000
        self.skill_graph = skill_graph
        self.search_node = search_node
        self.ori_rew_fun = rew_fun
        self.rew_fun = rew_fun
        self.env_success_rew = env.env_success_rew
        self.additional_reward_fun = additional_reward_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train
        self.task_limit_mode = task_limit is not None
        self.task_limit = np.inf if task_limit is None else task_limit
        self.task_step = 0
        self.debug_rew_fun = debug_rew_fun

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        self.env_debug = env_debug
        if self.env_debug:
            # init store
            self.ori_search_node = self.search_node
            # other init
            self.cur_id = self.search_node.node_id
            self.crop_id = self.search_node.s_node.crop_id
            self.node_policy_map = node_policy_map
            self.observation_space = new_observation_space
            self.obs_transit = obs_transit

    def set_train(self):
        self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # loop
        cur_attempt = 0
        self.hold_step = 0

        if self.env_debug:
            # init
            self.search_node = self.ori_search_node
            self.rew_fun = self.ori_rew_fun
            self.cur_id = self.search_node.node_id
            self.crop_id = self.search_node.s_node.crop_id

        # only for debug
        drop_num = 0
        while True:
            self.traj_id = 0
            self.task_step = 0

            if self.env_debug:
                # obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, self.skill_graph.valid_blocks, \
                #                                                     drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
                obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, \
                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            else:
                obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, \
                                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            self.traj_id += traj_len
            self.task_step += traj_len
            if info['drop']:
                drop_num += 1

            # success
            if success:
                break
            else:
                cur_attempt += 1
                # print('{}/{}'.format(drop_num, cur_attempt))
                if cur_attempt > self.attemp_num:
                    # print(drop_num)
                    raise Exception('reset fail on skill environment')

        self.reset_time += 1

        if self.env_debug:
            self.crop_id = self.search_node.s_node.crop_id
            # obs = self.env.env.get_custom_obs(self.crop_id)
            if self.crop_id is not None:
                obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                # obs = self.obs_transit.add_idx_state(np.expand_dims(obs, 0), self.crop_id)[0]
            info['policy_id'] = self.node_policy_map[self.cur_id]
            info['stage_done'] = False
            info['state_store'] = 'keep'

        # only for debug
        else:
            info['policy_id'] = 0
            info['stage_done'] = False

        if self.debug_rew_fun is not None:
            info['stage_success'] = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.train:
            self.rew_fun.set_eval()
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            self.rew_fun.set_train()
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            new_cost = None
            info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            # penalty
            new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)

            if self.env_debug:
                assert self.crop_id is not None
                info['stage_done'] = False
                info['state_store'] = 'keep'
                # directly return if truncate
                if truncated:
                    if self.crop_id is not None:
                        obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                    info['policy_id'] = self.node_policy_map[self.cur_id]
                    if reward >= 0:
                        info['state_store'] = 'store'
                    else:
                        info['state_store'] = 'drop'
                    return obs, new_reward, False, truncated, info
                
                # check reward
                if new_reward >= 0 and new_cost <= 0:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                if self.hold_step >= self.hold_len:
                    self.hold_step = 0
                    # TODO: need further consider
                    # define new
                    info['stage_obs'] = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                    info['stage_reward'] = new_reward
                    start_id = self.search_node.s_node.node_id
                    # stay current if final task
                    if self.search_node.s_node.s_node is not None:
                        # pdb.set_trace()
                        end_node, _, task, _ = self.skill_graph.get_search_node(start_id)
                        # rollout
                        success = True
                        if start_id != end_node.node_id:
                            print('should not get here for now')
                            pdb.set_trace()
                            obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len-self.traj_id, end_node.node_id, \
                                                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, start_id=start_id, exist_obs=obs)            
                            self.traj_id += traj_len
                            self.task_step += traj_len
                        # next
                        self.cur_id = end_node.node_id
                        self.crop_id = end_node.s_node.crop_id
                        self.search_node = end_node
                        self.rew_fun = task[0][0]

                        # new truncate
                        truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit or not success

                    # others
                    self.rew_fun.set_train()
                    # new_reward, new_cost = self.rew_fun.get_reward(obs, new_reward)
                    # new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)
                    info['cost'] = new_cost
                    info['stage_done'] = True
                    # if reward >= 0:
                    #     info['state_store'] = 'store'

                if self.debug_rew_fun is not None:
                    if self.debug_rew_fun.get_reward(obs, reward):
                        info['state_store'] = 'store'
                elif reward >= 0:
                    info['state_store'] = 'store'

                # get observation
                # obs = self.env.env.get_custom_obs(self.crop_id)
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                
                info['policy_id'] = self.node_policy_map[self.cur_id]

            elif self.debug_rew_fun is not None:
                info['stage_done'] = False
                info['policy_id'] = 0
                # info['stage_success'] = self.debug_rew_fun.get_reward(obs, reward)
                if truncated:
                    info['state_store'] = 'drop'
                elif self.debug_rew_fun.get_reward(obs, reward):
                    info['state_store'] = 'store'
                else:
                    info['state_store'] = 'keep'

            return obs, new_reward, False, truncated, info

        # eval mode
        else:
            # environment success
            if reward >= 0:
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = True
                new_reward = 1
            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = self.hold_step >= self.hold_len

            if self.env_debug:
                info['stage_done'] = False
                # next if success
                if terminated and new_reward >= self.done_thre:
                    reward_node = self.search_node.s_node.s_node
                    if reward_node is not None:
                        self.search_node = self.search_node.s_node
                        self.crop_id = reward_node.crop_id
                        self.rew_fun = self.search_node.s_node.reward_fun[0]
                        self.rew_fun.set_eval()
                        terminated = False
                    # update
                    info['stage_done'] = True

                # obs = self.env.env.get_custom_obs(0)
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                    # obs = self.obs_transit.add_idx_state(np.expand_dims(obs, 0), self.crop_id)[0]

            return obs, new_reward, terminated, truncated, info

class SkillEnvIter(gym.Wrapper):
    def __init__(self, env, skill_graph, search_node, rew_fun, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, \
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None, \
                 env_debug=False, new_observation_space=None, node_policy_map=None, obs_transit=None, debug_rew_fun=None):
        super().__init__(env)
        self.attemp_num = 1000
        self.skill_graph = skill_graph
        self.search_node = search_node
        self.ori_rew_fun = rew_fun
        self.rew_fun = rew_fun
        self.env_success_rew = env.env_success_rew
        self.additional_reward_fun = additional_reward_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train
        self.task_limit_mode = task_limit is not None
        self.task_limit = np.inf if task_limit is None else task_limit
        self.task_step = 0
        self.debug_rew_fun = debug_rew_fun

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        # init store
        self.ori_search_node = self.search_node
        # other init
        self.cur_id = self.search_node.node_id
        self.crop_id = self.search_node.s_node.crop_id
        self.node_policy_map = node_policy_map
        self.observation_space = new_observation_space
        self.obs_transit = obs_transit
        self.last_raw_obs = None

    def set_train(self):
        self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # loop
        cur_attempt = 0
        self.hold_step = 0

        # init
        self.search_node = self.ori_search_node
        self.rew_fun = self.ori_rew_fun
        self.cur_id = self.search_node.node_id
        self.crop_id = self.search_node.s_node.crop_id

        # only for debug
        drop_num = 0
        while True:
            self.traj_id = 0
            self.task_step = 0

            try:
                obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, self.skill_graph.valid_blocks, \
                                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            except:
                obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, \
                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            self.traj_id += traj_len
            self.task_step += traj_len
            if info['drop']:
                drop_num += 1

            # success
            if success:
                break
            else:
                cur_attempt += 1
                # print('{}/{}'.format(drop_num, cur_attempt))
                if cur_attempt > self.attemp_num:
                    # print(drop_num)
                    raise Exception('reset fail on skill environment')

        self.reset_time += 1

        # reset initial state for in case rollout
        self.last_raw_obs = obs
        self.crop_id = self.search_node.s_node.crop_id
        if self.crop_id is not None:
            obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
        info['policy_id'] = self.node_policy_map[self.cur_id]
        info['stage_done'] = False
        info['state_store'] = 'keep'

        if self.debug_rew_fun is not None:
            info['stage_success'] = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_raw_obs = obs
        if not self.train:
            self.rew_fun.set_eval()
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            self.rew_fun.set_train()
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            new_cost = None
            info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            # penalty
            new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)

            # check finish
            # assert self.crop_id is not None
            info['stage_done'] = False

            # directly return if truncate
            if truncated:
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                info['policy_id'] = self.node_policy_map[self.cur_id]
                if reward >= 0:
                    terminated = True
                else:
                    terminated = False
                return obs, new_reward, terminated, truncated, info
            
            # already success environment reward
            if reward >= 0:
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                info['stage_done'] = True
                terminated = True
                if new_reward < 1:
                    new_reward = 1

                return obs, new_reward, terminated, truncated, info
                
            # check reward
            if new_reward >= 0 and new_cost <= 0:
                self.hold_step += 1
            else:
                self.hold_step = 0
            if self.hold_step >= self.hold_len:
                self.hold_step = 0
                start_id = self.search_node.s_node.node_id
                # stay current if final task
                if self.search_node.s_node.s_node is not None:
                    end_node, _, task, _ = self.skill_graph.get_search_node(start_id)
                    # rollout
                    success = True
                    if start_id != end_node.node_id:
                        print('should not get here for now')
                        pdb.set_trace()
                        obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len-self.traj_id, end_node.node_id, \
                                                                                drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, start_id=start_id, exist_obs=obs)            
                        self.traj_id += traj_len
                        self.task_step += traj_len
                    # next
                    self.cur_id = end_node.node_id
                    self.crop_id = end_node.s_node.crop_id
                    self.search_node = end_node
                    self.rew_fun = task[0][0]

                    # new truncate
                    truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                    terminated = False
                else:
                    terminated = True

                # others
                self.rew_fun.set_train()
                info['stage_done'] = True
            else:
                terminated = False

            # get observation
            if self.crop_id is not None:
                obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
            
            info['policy_id'] = self.node_policy_map[self.cur_id]

            return obs, new_reward, terminated, truncated, info

        # eval mode
        else:
            info['env_success'] = False
            # environment success
            if reward >= 0:
                info['env_success'] = True
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = True
                info['stage_done'] = True
                new_reward = 1
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                return obs, new_reward, terminated, truncated, info

            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = self.hold_step >= self.hold_len

            info['stage_done'] = False
            # next if success
            if terminated and new_reward >= self.done_thre:
                reward_node = self.search_node.s_node.s_node
                if reward_node is not None:
                    self.search_node = self.search_node.s_node
                    self.cur_id = self.search_node.node_id
                    self.crop_id = reward_node.crop_id
                    self.rew_fun = self.search_node.s_node.reward_fun[0]
                    self.rew_fun.set_eval()
                    terminated = False
                # update
                info['stage_done'] = True
                self.hold_step = 0

            if self.crop_id is not None:
                obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]

            return obs, new_reward, terminated, truncated, info

class SkillEnvIterNew(gym.Wrapper):
    def __init__(self, env, iter_program, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, \
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None, \
                 env_debug=False, new_observation_space=None, node_policy_map=None, obs_transit=None):
        super().__init__(env)
        self.attemp_num = 1000
        self.iter_program = iter_program
        self.env_success_rew = env.env_success_rew
        self.additional_reward_fun = additional_reward_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train
        self.task_limit_mode = task_limit is not None
        self.task_limit = np.inf if task_limit is None else task_limit
        self.task_step = 0

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        # init store
        self.node_policy_map = node_policy_map
        self.observation_space = new_observation_space
        self.obs_transit = obs_transit
        self.rew_fun = None

    def set_train(self):
        if self.rew_fun is not None:
            self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        if self.rew_fun is not None:
            self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # loop
        cur_attempt = 0
        self.hold_step = 0

        # init
        obs, info, search_node, rew_node, _, _ = self.iter_program.do_reset(self.env, **kwargs)
        
        self.traj_id = 0
        self.task_step = 0
        self.search_node = search_node
        self.rew_fun = rew_node.reward_fun[0]
        self.cur_id = search_node.node_id
        self.crop_id = rew_node.crop_id

        if self.crop_id is not None:
            obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
            # obs = self.obs_transit.do_crop_state_all(np.expand_dims(obs, 0), self.crop_id)[0]
        info['policy_id'] = self.node_policy_map[self.cur_id]
        info['stage_done'] = False
        info['state_store'] = 'keep'

        return obs, info

    # only reset iterative program and maintain environment state
    def loop_reset(self, obs, **kwargs):
        # reset
        search_node, rew_node, _, _ = self.iter_program.do_loop_reset(obs)

        self.search_node = search_node
        self.rew_fun = rew_node.reward_fun[0]
        self.cur_id = search_node.node_id
        self.crop_id = rew_node.crop_id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_raw_obs = obs
        if not self.train:
            self.rew_fun.set_eval()
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            self.rew_fun.set_train()
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            new_cost = None
            info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            # penalty
            new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)

            # check finish
            # assert self.crop_id is not None
            info['stage_done'] = False

            # directly return if truncate
            if truncated:
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                info['policy_id'] = self.node_policy_map[self.cur_id]
                if reward >= 0:
                    terminated = True
                else:
                    terminated = False
                return obs, new_reward, terminated, truncated, info
            
            # already success environment reward
            if reward >= 0:
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
                info['stage_done'] = True
                terminated = True
                if new_reward < 1:
                    new_reward = 1

                info['policy_id'] = self.node_policy_map[self.cur_id]

                return obs, new_reward, terminated, truncated, info
                
            # check reward
            if new_reward >= 0 and new_cost <= 0:
                self.hold_step += 1
            else:
                self.hold_step = 0
            if self.hold_step >= self.hold_len:
                self.hold_step = 0
                start_id = self.search_node.s_node.node_id
                # try find the next node
                end_node, _, task, _ = self.iter_program.get_search_node(start_id, obs)

                if end_node is not None:
                    # rollout
                    # assert start_id == end_node.node_id     # only consider this situation for now
                    # next
                    self.search_node = end_node
                    self.cur_id = self.search_node.node_id
                    self.crop_id = self.search_node.s_node.crop_id
                    self.rew_fun = task[0][0]

                    # new truncate
                    if self.cur_id == 0:
                        self.task_step = 0
                    truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                    terminated = False
                else:
                    terminated = True

                # others
                self.rew_fun.set_train()
                info['stage_done'] = True
            else:
                terminated = False

            # get observation
            if self.crop_id is not None:
                obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]
            
            info['policy_id'] = self.node_policy_map[self.cur_id]

            return obs, new_reward, terminated, truncated, info

        # eval mode
        else:
            info['env_success'] = False
            # pdb.set_trace()
            # environment success
            if reward >= 0:
                info['env_success'] = True
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = True
                info['stage_done'] = True
                new_reward = 1
                if self.crop_id is not None:
                    obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]

                info['policy_id'] = self.node_policy_map[self.cur_id]

                return obs, new_reward, terminated, truncated, info

            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = self.hold_step >= self.hold_len

            info['stage_done'] = False
            # next if success
            if terminated and new_reward >= self.done_thre:
                # get next node
                start_id = self.search_node.s_node.node_id
                search_node, reward_node, task, _ = self.iter_program.get_search_node(start_id, obs)

                if reward_node is not None:
                    self.search_node = search_node
                    self.cur_id = search_node.node_id
                    self.crop_id = reward_node.crop_id
                    self.rew_fun = reward_node.reward_fun[0]
                    self.rew_fun.set_eval()
                    terminated = False
                    if self.cur_id == 0:
                        self.task_step = 0
                if terminated:
                    pdb.set_trace()

                # update
                info['stage_done'] = True
                self.hold_step = 0

            if self.crop_id is not None:
                obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_id)[0]

            info['policy_id'] = self.node_policy_map[self.cur_id]

            return obs, new_reward, terminated, truncated, info


class SkillEnvPure(gym.Wrapper):
    def __init__(self, env, skill_graph, search_node, rew_fun, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, \
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None, \
                 env_debug=False, new_observation_space=None, node_policy_map=None, obs_transit=None, debug_rew_fun=None):
        super().__init__(env)
        self.attemp_num = 1000
        self.skill_graph = skill_graph
        self.search_node = search_node
        self.ori_rew_fun = rew_fun
        self.rew_fun = rew_fun
        self.env_success_rew = env.env_success_rew
        self.additional_reward_fun = additional_reward_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train
        self.task_limit_mode = task_limit is not None
        self.task_limit = np.inf if task_limit is None else task_limit
        self.task_step = 0
        self.debug_rew_fun = debug_rew_fun

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        self.env_debug = env_debug
        assert not self.env_debug

    def set_train(self):
        self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # loop
        cur_attempt = 0
        self.hold_step = 0

        # only for debug
        drop_num = 0
        while True:
            self.traj_id = 0
            self.task_step = 0

            obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, \
                                                                drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            self.traj_id += traj_len
            self.task_step += traj_len
            if info['drop']:
                drop_num += 1

            # success
            if success:
                break
            else:
                cur_attempt += 1
                # print('{}/{}'.format(drop_num, cur_attempt))
                if cur_attempt > self.attemp_num:
                    # print(drop_num)
                    raise Exception('reset fail on skill environment')

        self.reset_time += 1

        info['policy_id'] = 0
        info['stage_done'] = False

        if self.debug_rew_fun is not None:
            info['stage_success'] = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.train:
            self.rew_fun.set_eval()
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            self.rew_fun.set_train()
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            new_cost = None
            info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            # penalty
            new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)

            # already success environment reward
            # if reward >= 0:
            #     terminated = True
            #     if new_reward < 1:
            #         new_reward = 1
            #     return obs, new_reward, terminated, truncated, info

            # otherwise
            terminated = False
            # if new_reward >= 0 and new_cost <= 0:
            #     self.hold_step += 1
            # else:
            #     self.hold_step = 0
            
            # if self.hold_step > self.hold_len:
            #     terminated = True

            return obs, new_reward, terminated, truncated, info

        # eval mode
        else:
            info['stage_done'] = False
            # environment success
            if reward >= 0:
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = True
                new_reward = 1
                info['stage_done'] = True
            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = self.hold_step >= self.hold_len
                if terminated:
                    info['stage_done'] = True

            return obs, new_reward, terminated, truncated, info


class CustomMonitor(gym.Wrapper):
    def __init__(self, env, log_freq, rew_fun):
        super().__init__(env)
        self.log_freq = log_freq
        self.reset_time = 0
        self.rew_fun = rew_fun
        self.rew_store = []
        self.act_store = []
        self.img_store = []

    def reset(self, **kwargs):
        self.reset_time += 1
        if self.reset_time % self.log_freq == 0:
            print('do store')
            self.rew_store = []
            self.act_store = []
            self.img_store = []
            # self.rew_store.append([])
            # self.act_store.append([])

        obs, info = self.env.reset(**kwargs)
        if self.reset_time % self.log_freq == 0:
            # self.rew_store.append(self.env.get_reward())
            self.img_store.append(self.env.render())

        if self.reset_time % self.log_freq == 1 and self.reset_time!=1:
            print('print')
            # plt.figure()
            # rew_data = self.rew_store
            # plt.plot(np.arange(len(rew_data)), rew_data, 'k-')
            # plt.savefig('store/tree_push_debug_4/debug_fig/rew_{}.png'.format(self.reset_time*100))
            # plt.close()

            plt.figure()
            action_data = np.array(self.act_store)
            plt.plot(np.arange(len(action_data)), action_data[:, -1], 'k-')
            plt.savefig('store/tree_push_debug_3/debug_fig/act_{}.png'.format(self.reset_time*100))
            plt.close()

            os.makedirs('store/tree_push_debug_3/debug_fig/{}'.format(self.reset_time*100))
            plt.figure()
            for img_id, img in enumerate(self.img_store):
                plt.imshow(img)
                plt.savefig('store/tree_push_debug_3/debug_fig/{}/{}.png'.format(self.reset_time*100, img_id))
                plt.cla()
            plt.close()

        return obs, info
        
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.reset_time % self.log_freq == 0:
            # dense_result = self.rew_fun.equ.execute_dense_details(np.expand_dims(obs, axis=0))
            dense_result = reward
            self.rew_store.append(dense_result)
            self.act_store.append(action)
            self.img_store.append(self.env.render())

        return obs, reward, terminated, truncated, info

class AbsTransit_Pick:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)
        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist = self.get_z_distance(obs)

        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist], \
                            axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, \
        #                        gripper_dist], \
        #                     axis=1)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist], \
        #                     axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        obs[:, 3:5], \
        #                        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist], \
        #                     axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, obs[:, 3:5], obs[:, 0:3], obs[:, 5:8], obs[:, 8:11]], axis=1)

    def get_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]

        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_block_dist = np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True))
        goal_block_dist = np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True))
        gripper_goal_dist = np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True))
        gripper_dist = np.expand_dims(obs[:, 9] + obs[:, 10], axis=1)

        return gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist
    
    def get_z_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]

        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        return gripper_pos[:, -1:]-block_pos[:, -1:], goal_pos[:, -1:]-block_pos[:, -1:], goal_pos[:, -1:]-gripper_pos[:, -1:]

class AbsTransit_Push:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)
        # goal_block_angle, goal_gripper_angle, gripper_block_angle = self.get_angle(obs)
        gripper_angle, block_angle, goal_angle = self.get_angle_diff(obs)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        goal_block_angle, goal_gripper_angle, gripper_block_angle], \
        #                     axis=1)
        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist, \
                               gripper_angle, block_angle, goal_angle], \
                            axis=1)

    def get_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]
        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_block_dist = np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True))
        goal_block_dist = np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True))
        gripper_goal_dist = np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True))
        gripper_dist = np.expand_dims(obs[:, 9] + obs[:, 10], axis=1)

        # only for debug
        # gripper_block_dist = 10 * gripper_block_dist
        # goal_block_dist = 10 * goal_block_dist
        # gripper_goal_dist = 10 * gripper_goal_dist

        return gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist
    
    def get_angle_diff(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]
        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_angle = np.arctan2(gripper_pos[:, 0] - block_pos[:, 0], gripper_pos[:, 1] - block_pos[:, 1]) -  \
                        np.arctan2(gripper_pos[:, 0] - goal_pos[:, 0], gripper_pos[:, 1] - goal_pos[:, 1])
        block_angle = np.arctan2(block_pos[:, 0] - gripper_pos[:, 0], block_pos[:, 1] - gripper_pos[:, 1]) -  \
                      np.arctan2(block_pos[:, 0] - goal_pos[:, 0], block_pos[:, 1] - goal_pos[:, 1])
        goal_angle = np.arctan2(goal_pos[:, 0] - block_pos[:, 0], goal_pos[:, 1] - block_pos[:, 1]) - \
                     np.arctan2(goal_pos[:, 0] - gripper_pos[:, 0], goal_pos[:, 1] - gripper_pos[:, 1])

        gripper_angle = np.abs((gripper_angle + np.pi) % (2*np.pi) - np.pi)
        block_angle = np.abs((block_angle + np.pi) % (2*np.pi) - np.pi)
        goal_angle = np.abs((goal_angle + np.pi) % (2*np.pi) - np.pi)

        # only for debug
        # gripper_angle = 0.1 * gripper_angle
        # block_angle = 0.1 * block_angle
        # goal_angle = 0.1 * goal_angle

        return np.expand_dims(gripper_angle, axis=1), np.expand_dims(block_angle, axis=1), np.expand_dims(goal_angle, axis=1)

class AbsTransit_Ant:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        x_dist, y_dist = self.get_distance(obs)

        return np.concatenate([x_dist, y_dist], axis=1)

    def get_distance(self, obs):
        ant_pos = obs[:, :2]
        goal_pos = obs[:, 29:31]

        return np.abs(ant_pos[:,:1]-goal_pos[:,:1]), np.abs(ant_pos[:,1:2]-goal_pos[:,1:2])

class AbsTransit_Block:
    def __init__(self):
        self.add_dim = 3
        self.block_num = None
    
    def set_num(self, block_num):
        self.block_num = block_num
        gripper_block_ids = np.arange(self.block_num)
        goal_block_dist = np.arange(self.block_num) + gripper_block_ids[-1] + 1
        gripper_goal_dist = np.arange(self.block_num) + goal_block_dist[-1] + 1
        gripper_dist = np.arange(self.block_num) + gripper_goal_dist[-1] + 1
        self.check_list = [gripper_block_ids, goal_block_dist, gripper_goal_dist, gripper_dist]

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)

        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist], \
                            axis=1)

    def get_distance(self, obs):
        # get block number
        block_num = (obs.shape[-1] - 13) // 18

        gripper_pos = obs[:, :3]
        block_pos_list = [obs[:, 10+i*15 : 10+i*15+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*15+3*i : 10+block_num*15+3*i+3] for i in range(block_num)]

        gripper_block_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos in block_pos_list], axis=1)
        goal_block_dist_list = np.concatenate([np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos, goal_pos in zip(block_pos_list, goal_pos_list)], axis=1)
        gripper_goal_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True)) for goal_pos in goal_pos_list], axis=1)

        gripper_dist = np.expand_dims(obs[:, 3] + obs[:, 4], axis=1)

        return gripper_block_dist_list, goal_block_dist_list, gripper_goal_dist_list, gripper_dist

    # crop state dimension based on valid object id
    def do_crop_state(self, obs, obj_ids):
        # get block number
        block_num = (obs.shape[-1] - 13) // 18

        # comment state
        comment_state = obs[:, :10]
        # remain object state
        remain_obj_states = [obs[:, 10+i*15: 10+(i+1)*15] for i in obj_ids]
        # remain goal state
        remain_goal_states = [obs[:, 10+block_num*15+3*i:10+block_num*15+3*(i+1)] for i in obj_ids]
        # final state
        final_state = obs[:, -3:]

        return np.concatenate([comment_state]+remain_obj_states+remain_goal_states+[final_state], axis=1)

    # get obj id
    def get_obj_id(self, ori_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return np.where(each_list == ori_id)[0][0]

        return None

    # abstract state id mapping if switch require
    def get_switch_ids(self, ori_id, goal_obj):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return each_list[goal_obj]
            
        return None
    
    # check match
    def check_match_ids(self, ori_id, comp_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if (ori_id <= each_list[-1]) != (comp_id <= each_list[-1]):
                return False
            if ori_id <= each_list[-1]:
                return True

        return True
    

class AbsTransit_Custom_Block:
    def __init__(self, no_rank=False, decimal_pos=None):
        self.add_dim = 3
        self.block_num = None
        self.no_rank = no_rank
        self.decimal_pos = decimal_pos
    
    def set_num(self, block_num):
        self.block_num = block_num
        gripper_block_ids = np.arange(self.block_num)
        goal_block_dist = np.arange(self.block_num) + gripper_block_ids[-1] + 1
        gripper_goal_dist = np.arange(self.block_num) + goal_block_dist[-1] + 1
        gripper_dist = np.zeros(self.block_num, dtype=int) + gripper_goal_dist[-1] + 1
        self.check_list = [gripper_block_ids, goal_block_dist, gripper_goal_dist, gripper_dist]

    def get_abs_obs(self, state, put_first=None):
        # distance between objects
        gripper_block_dist, goal_block_dist, goal_block_z_dist, gripper_goal_dist, gripper_dist = self.get_distance(state, put_first)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist], \
        #                     axis=1)
        res = np.concatenate([gripper_block_dist, goal_block_dist, goal_block_z_dist, gripper_dist], \
                    axis=1)
        
        if self.decimal_pos is not None:
            res = np.around(res, self.decimal_pos)

        return res

    def unnorm_ob(self, ob):
        obs_max = np.array([[1.36, 0.86, 0.59, 0.05, 0.05, 1.36, 0.86, 0.59, 1.36, 0.86, 0.59, 1.36, 0.86, 0.59]])
        obs_min = np.array([[1.3 , 0.64, 0.42, 0.  , 0.  , 1.3 , 0.64, 0.42, 1.3 , 0.64, 0.42, 1.3 , 0.64, 0.42]])

        new_obs = ob.copy()
        new_obs[:, :new_obs.shape[1]//2] = obs_min + new_obs[:, :new_obs.shape[1]//2] * (obs_max -  obs_min)
        new_obs[:, new_obs.shape[1]//2:] = obs_min + new_obs[:, new_obs.shape[1]//2:] * (obs_max -  obs_min)
        return new_obs

    def get_distance(self, state, put_first):
        # TODO: try to unnorm
        # state = self.unnorm_ob(state)

        # split dimension
        obs, goal = state[:, :state.shape[1]//2], state[:, state.shape[1]//2:]
        block_num = (obs.shape[-1] - 5) // 3

        grip_pos = obs[:, :3]
        grip_state = obs[:, 3:5]
        all_obj_pos = np.split(obs[:, 5:5+3*block_num], block_num, axis=1)

        grip_pos_goal = goal[:, :3]
        grip_state_goal = goal[:, 3:5]
        all_obj_pos_goal = np.split(goal[:, 5:5+3*block_num], block_num, axis=1)

        # get distance
        gripper_block_dist_list = np.concatenate([np.sqrt(np.sum((grip_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos in all_obj_pos], axis=1)
        goal_block_dist_list = np.concatenate([np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos, goal_pos in zip(all_obj_pos, all_obj_pos_goal)], axis=1)
        goal_block_z_dist_list = np.concatenate([block_pos[:,[-1]]-goal_pos[:,[-1]] for block_pos, goal_pos in zip(all_obj_pos, all_obj_pos_goal)], axis=1)
        gripper_goal_dist_list = np.sqrt(np.sum((grip_pos-grip_pos_goal)**2, axis=-1, keepdims=True), )

        # change order
        if put_first is not None:
            for row_id in range(gripper_block_dist_list.shape[0]):
                gripper_block_dist_list[row_id] = np.concatenate([gripper_block_dist_list[row_id][put_first[row_id]:put_first[row_id]+1], \
                                                                  gripper_block_dist_list[row_id][:put_first[row_id]], \
                                                                  gripper_block_dist_list[row_id][put_first[row_id]+1:]])
                goal_block_z_dist_list[row_id] = np.concatenate([goal_block_z_dist_list[row_id][put_first[row_id]:put_first[row_id]+1], \
                                                                 goal_block_z_dist_list[row_id][:put_first[row_id]], \
                                                                 goal_block_z_dist_list[row_id][put_first[row_id]+1:]])
                goal_block_dist_list[row_id] = np.concatenate([goal_block_dist_list[row_id][put_first[row_id]:put_first[row_id]+1], \
                                                               goal_block_dist_list[row_id][:put_first[row_id]], \
                                                               goal_block_dist_list[row_id][put_first[row_id]+1:]])
            
        elif not self.no_rank:
            obj_goal_last_dim = np.concatenate([each_obj_pos_goal[:, -1:] for each_obj_pos_goal in all_obj_pos_goal], axis=1)
            obj_idxs = np.argsort(obj_goal_last_dim, axis=1)

            for row_id in range(gripper_block_dist_list.shape[0]):
                gripper_block_dist_list[row_id] = gripper_block_dist_list[row_id][obj_idxs[row_id]]
                goal_block_z_dist_list[row_id] = goal_block_z_dist_list[row_id][obj_idxs[row_id]]
                goal_block_dist_list[row_id] = goal_block_dist_list[row_id][obj_idxs[row_id]]

        gripper_dist = np.expand_dims(grip_state[:, 0] + grip_state[:, 1], axis=1)

        return gripper_block_dist_list, goal_block_dist_list, goal_block_z_dist_list, gripper_goal_dist_list, gripper_dist

    # crop state dimension based on valid object id
    def do_crop_state(self, state, crop_id, return_dict=False):
        # pdb.set_trace()
        # init
        sep_dim  = state.shape[1]//2
        block_num = (state.shape[-1] - 10) // 6
        
        # get real id based on order
        all_obj_pos_goal = np.split(state[:, 5+sep_dim:5+sep_dim+3*block_num], block_num, axis=1)
        obj_goal_last_dim = np.concatenate([each_obj_pos_goal[:, -1:] for each_obj_pos_goal in all_obj_pos_goal], axis=1)
        obj_idxs = np.argsort(obj_goal_last_dim, axis=1)
        crop_idxs = obj_idxs[:, crop_id]

        # do crop
        grip_state = state[:, :5]
        goal_grip_state = state[:, sep_dim: sep_dim+5]

        keep_block_state = []
        goal_keep_block_state = []
        for state_id in range(state.shape[0]):
            keep_block_state.append(state[state_id, 5+3*crop_idxs[state_id]: 5+3*crop_idxs[state_id]+3])
            goal_keep_block_state.append(state[state_id, sep_dim+5+3*crop_idxs[state_id]: sep_dim+5+3*crop_idxs[state_id]+3])

        keep_block_state = np.vstack(keep_block_state)
        goal_keep_block_state = np.vstack(goal_keep_block_state)

        if return_dict:
            return {'grip': grip_state, 'block': keep_block_state, 'grip_goal': goal_grip_state, 'block_goal': goal_keep_block_state}
        return np.concatenate([grip_state, keep_block_state, goal_grip_state, goal_keep_block_state], axis=1)

    # add current id as indicator
    def add_idx_state(self, state, block_id):
        # init
        sep_dim  = state.shape[1]//2
        block_num = (state.shape[-1] - 10) // 6
        # get real id based on order
        all_obj_pos_goal = np.split(state[:, 5+sep_dim:5+sep_dim+3*block_num], block_num, axis=1)
        obj_goal_last_dim = np.concatenate([each_obj_pos_goal[:, -1:] for each_obj_pos_goal in all_obj_pos_goal], axis=1)
        obj_idxs = np.argsort(obj_goal_last_dim, axis=1)
        # add state
        idx_state = np.zeros((state.shape[0], block_num))
        idx_state[:, obj_idxs[:, block_id]] = 1

        return np.concatenate([state, idx_state], 1)

    # get obj id based on rank
    def get_obj_rank_id(self, state):
        # split dimension
        obs, goal = state[:, :state.shape[1]//2], state[:, state.shape[1]//2:]
        block_num = (obs.shape[-1] - 5) // 3

        # get order based on goal
        all_obj_pos_goal = np.split(goal[:, 5:5+3*block_num], block_num, axis=1)
        obj_goal_last_dim = np.concatenate([each_obj_pos_goal[:, -1:] for each_obj_pos_goal in all_obj_pos_goal], axis=1)
        obj_idxs = np.argsort(obj_goal_last_dim, axis=1)

        return obj_idxs

    # get obj id
    def get_obj_id(self, ori_id):
        pdb.set_trace()
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return np.where(each_list == ori_id)[0][0]

        return None

    # abstract state id mapping if switch require
    def get_switch_ids(self, ori_id, goal_obj):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return each_list[goal_obj]
            
        return None
    
    # check match
    def check_match_ids(self, ori_id, comp_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if (ori_id <= each_list[-1]) != (comp_id <= each_list[-1]):
                return False
            if ori_id <= each_list[-1]:
                return True

        return True
    

class AbsTransit_Pick_Multi:
    def __init__(self):
        self.add_dim = 3
        self.block_num = None
        self.full_dim = False
    
    def set_num(self, block_num):
        self.block_num = block_num
        gripper_block_ids = np.arange(self.block_num)
        goal_block_dist = np.arange(self.block_num) + gripper_block_ids[-1] + 1
        gripper_dist = np.zeros(self.block_num, dtype=int) + goal_block_dist[-1] + 1
        self.check_list = [gripper_block_ids, goal_block_dist, gripper_dist]

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist], \
        #                     axis=1)
        return np.concatenate([gripper_block_dist, goal_block_dist, \
                               gripper_dist], \
                            axis=1)

    def get_distance(self, obs):
        # get block number
        block_num = (obs.shape[-1] - 13) // 15

        gripper_pos = obs[:, :3]
        block_pos_list = [obs[:, 10+i*12 : 10+i*12+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*12+3*i : 10+block_num*12+3*i+3] for i in range(block_num)]

        gripper_block_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos in block_pos_list], axis=1)
        goal_block_dist_list = np.concatenate([np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos, goal_pos in zip(block_pos_list, goal_pos_list)], axis=1)
        gripper_goal_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True)) for goal_pos in goal_pos_list], axis=1)

        gripper_dist = np.expand_dims(obs[:, 3] + obs[:, 4], axis=1)

        return gripper_block_dist_list, goal_block_dist_list, gripper_goal_dist_list, gripper_dist

    # crop state dimension based on valid object id
    def do_crop_state(self, obs, obj_id, return_dict=False):
        # get block number
        block_num = (obs.shape[-1] - 13) // 15

        gripper_pos = obs[:, :5]
        block_pos_list = [obs[:, 10+i*12 : 10+i*12+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*12+3*i : 10+block_num*12+3*i+3] for i in range(block_num)]

        if return_dict:
            return {'grip': gripper_pos, 'block': block_pos_list[obj_id], 'grip_goal': np.zeros_like(gripper_pos), 'block_goal': goal_pos_list[obj_id]}
        else:
            # return self.do_crop_state_all(obs, obj_id, return_dict)
            return np.concatenate([gripper_pos, block_pos_list[obj_id], np.zeros_like(gripper_pos), goal_pos_list[obj_id]], axis=1)

    # crop state dimension based on valid object id
    def do_crop_state_all(self, obs, obj_id, return_dict=False):
        # get block number
        block_num = (obs.shape[-1] - 13) // 15

        gripper_pos = obs[:, :10]
        block_list = [obs[:, 10+i*12 : 10+(i+1)*12] for i in range(block_num)]
        goal_list = [obs[:, 10+block_num*12+3*i : 10+block_num*12+3*i+3] for i in range(block_num)]

        if return_dict:
            return {'grip': gripper_pos, 'block': block_list[obj_id], 'grip_goal': np.zeros_like(gripper_pos), 'block_goal': goal_list[obj_id]}
        # return np.concatenate([gripper_pos, block_list[obj_id], np.zeros_like(gripper_pos), goal_list[obj_id]], axis=1)
        return np.concatenate([gripper_pos, block_list[obj_id], goal_list[obj_id]], axis=1)

    # get obj id
    def get_obj_id(self, ori_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return np.where(each_list == ori_id)[0][0]

        return None

    # abstract state id mapping if switch require
    def get_switch_ids(self, ori_id, goal_obj):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return each_list[goal_obj]
            
        return None
    
    # check match
    def check_match_ids(self, ori_id, comp_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if (ori_id <= each_list[-1]) != (comp_id <= each_list[-1]):
                return False
            if ori_id <= each_list[-1]:
                return True

        return True
    
class AbsTransit_Push_Multi:
    def __init__(self):
        self.add_dim = 3
        self.block_num = None
    
    def set_num(self, block_num):
        self.block_num = block_num
        gripper_block_ids = np.arange(self.block_num)
        goal_block_dist = np.arange(self.block_num) + gripper_block_ids[-1] + 1
        gripper_goal_dist = np.arange(self.block_num) + goal_block_dist[-1] + 1
        gripper_dist = np.arange(self.block_num) + gripper_goal_dist[-1] + 1
        gripper_angle = np.arange(self.block_num) + gripper_dist[-1] + 1
        block_angle = np.arange(self.block_num) + gripper_angle[-1] + 1
        goal_angle = np.arange(self.block_num) + block_angle[-1] + 1

        self.check_list = [gripper_block_ids, goal_block_dist, gripper_goal_dist, gripper_dist, gripper_angle, block_angle, goal_angle]


    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)
        gripper_angle, block_angle, goal_angle = self.get_angle_diff(obs)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        goal_block_angle, goal_gripper_angle, gripper_block_angle], \
        #                     axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        gripper_angle, block_angle, goal_angle], \
        #                     axis=1)
        return np.concatenate([gripper_block_dist, goal_block_dist, \
                               gripper_dist, \
                               block_angle], \
                            axis=1)

    def get_distance(self, obs):
        # get block number
        block_num = (obs.shape[-1] - 10) // 19

        gripper_pos = obs[:, :3]
        block_pos_list = [obs[:, 10+i*16 : 10+i*16+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*16+3*i : 10+block_num*16+3*i+3] for i in range(block_num)]

        gripper_block_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos in block_pos_list], axis=1)
        goal_block_dist_list = np.concatenate([np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos, goal_pos in zip(block_pos_list, goal_pos_list)], axis=1)
        gripper_goal_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True)) for goal_pos in goal_pos_list], axis=1)

        gripper_dist = np.expand_dims(obs[:, 3] + obs[:, 4], axis=1)

        return gripper_block_dist_list, goal_block_dist_list, gripper_goal_dist_list, gripper_dist

    def get_angle_diff(self, obs):
        # get block number
        block_num = (obs.shape[-1] - 10) // 19

        gripper_pos = obs[:, :3]
        block_pos_list = [obs[:, 10+i*16 : 10+i*16+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*16+3*i : 10+block_num*16+3*i+3] for i in range(block_num)]

        gripper_angle_list = []
        block_angle_list = []
        goal_angle_list = []
        for block_pos, goal_pos in zip(block_pos_list, goal_pos_list):
            gripper_angle = np.arctan2(gripper_pos[:, 0] - block_pos[:, 0], gripper_pos[:, 1] - block_pos[:, 1]) -  \
                            np.arctan2(gripper_pos[:, 0] - goal_pos[:, 0], gripper_pos[:, 1] - goal_pos[:, 1])
            block_angle = np.arctan2(block_pos[:, 0] - gripper_pos[:, 0], block_pos[:, 1] - gripper_pos[:, 1]) -  \
                        np.arctan2(block_pos[:, 0] - goal_pos[:, 0], block_pos[:, 1] - goal_pos[:, 1])
            goal_angle = np.arctan2(goal_pos[:, 0] - block_pos[:, 0], goal_pos[:, 1] - block_pos[:, 1]) - \
                        np.arctan2(goal_pos[:, 0] - gripper_pos[:, 0], goal_pos[:, 1] - gripper_pos[:, 1])

            gripper_angle = np.abs((gripper_angle + np.pi) % (2*np.pi) - np.pi)
            block_angle = np.abs((block_angle + np.pi) % (2*np.pi) - np.pi)
            goal_angle = np.abs((goal_angle + np.pi) % (2*np.pi) - np.pi)

            gripper_angle_list.append(np.expand_dims(gripper_angle, 1))
            block_angle_list.append(np.expand_dims(block_angle, 1))
            goal_angle_list.append(np.expand_dims(goal_angle, 1))

        gripper_angle_list = np.concatenate(gripper_angle_list, axis=1)
        block_angle_list = np.concatenate(block_angle_list, axis=1)
        goal_angle_list = np.concatenate(goal_angle_list, axis=1)

        return gripper_angle_list, block_angle_list, goal_angle_list

    # crop state dimension based on valid object id
    def do_crop_state(self, obs, obj_id, return_dict=False):
        # get block number
        block_num = (obs.shape[-1] - 10) // 19

        gripper_pos = obs[:, :5]
        block_pos_list = [obs[:, 10+i*16 : 10+i*16+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*16+3*i : 10+block_num*16+3*i+3] for i in range(block_num)]

        if return_dict:
            return {'grip': gripper_pos, 'block': block_pos_list[obj_id], 'grip_goal': np.zeros_like(gripper_pos), 'block_goal': goal_pos_list[obj_id]}
        else:
            return self.do_crop_state_all(obs, obj_id, return_dict)
        # return np.concatenate([gripper_pos, block_pos_list[obj_id], np.zeros_like(gripper_pos), goal_pos_list[obj_id]], axis=1)

    # crop state dimension based on valid object id
    def do_crop_state_all(self, obs, obj_id, return_dict=False):
        # get block number
        block_num = (obs.shape[-1] - 10) // 19

        gripper_pos = obs[:, :10]
        block_list = [obs[:, 10+i*16 : 10+(i+1)*16] for i in range(block_num)]
        goal_list = [obs[:, 10+block_num*16+3*i : 10+block_num*16+3*i+3] for i in range(block_num)]

        if return_dict:
            return {'grip': gripper_pos, 'block': block_list[obj_id], 'grip_goal': np.zeros_like(gripper_pos), 'block_goal': goal_list[obj_id]}
        # return np.concatenate([gripper_pos, block_list[obj_id], np.zeros_like(gripper_pos), goal_list[obj_id]], axis=1)
        return np.concatenate([gripper_pos, block_list[obj_id], goal_list[obj_id]], axis=1)


    # get obj id
    def get_obj_id(self, ori_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return np.where(each_list == ori_id)[0][0]

        return None

    # abstract state id mapping if switch require
    def get_switch_ids(self, ori_id, goal_obj):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return each_list[goal_obj]
            
        return None
    
    # check match
    def check_match_ids(self, ori_id, comp_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if (ori_id <= each_list[-1]) != (comp_id <= each_list[-1]):
                return False
            if ori_id <= each_list[-1]:
                return True

        return True