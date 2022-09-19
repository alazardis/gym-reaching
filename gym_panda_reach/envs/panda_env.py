import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

MAX_EPISODE_LEN = 20 * 100


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(np.array(goal_a) - np.array(goal_b), axis=-1)


class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.reward_type = "dense"
        p.connect(p.GUI, options='--background_color_red=0.0 --background_color_green=0.93--background_color_blue=0.54')
        p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=45, cameraPitch=-30,
                                     cameraTargetPosition=[0.65, -0.0, 0.65])
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = spaces.Box(np.array([-1] * 24), np.array([1] * 24))

    # def compute_reward(self, achieved_goal, goal):
    #     # Compute distance between goal and the achieved goal.
    #     d = goal_distance(achieved_goal, goal)
    #     if self.reward_type == 'sparse':
    #         return -(d > self.distance_threshold).astype(np.float32)
    #     else:
    #         return -d
    def compute_reward(self, joint_position, desired_position):
        reward_finger1 = -abs(joint_position[0] - desired_position[0])
        reward_finger2 = -abs(joint_position[1] - desired_position[1])
        total_reward = reward_finger1 + reward_finger2
        return total_reward

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0., -math.pi, -math.pi])
        #         dv = 0.005
        # dv = 0.05
        # dx = action[0] * dv
        # dy = action[1] * dv
        # dz = action[2] * dv
        # fingers = action[3]
        finger1 = action[0]
        finger2 = action[1]

        hand_link = 9
        finger1_link = 10
        finger2_link = 11

        hand_joint = 8
        finger1_joint = 9
        finger2_joint = 10

        # currentPose = p.getLinkState(self.pandaUid, hand_link)
        # currentPosition = currentPose[0]
        state_finger1 = p.getJointState(self.pandaUid, finger1_joint)
        state_finger2 = p.getJointState(self.pandaUid, finger2_joint)
        new_state_finger1 = state_finger1[0]
        new_state_finger2 = state_finger2[0]
        # newPosition = [currentPosition[0] + dx,
        #                currentPosition[1] + dy,
        #                currentPosition[2] + dz]
        newPosition = [new_state_finger1 + finger1,
                       new_state_finger2 + finger2]
        # jointPoses = p.calculateInverseKinematics(self.pandaUid, hand_joint, newPosition, orientation)[0]
        #
        # p.setJointMotorControlArray(self.pandaUid, [finger1_joint + finger2_joint], p.POSITION_CONTROL,
        #                             list(jointPoses) + finger1_joint + finger2_joint)

        p.stepSimulation()

        state_object, state_object_orienation = p.getBasePositionAndOrientation(self.objectUid)
        twist_object, twist_object_orienation = p.getBaseVelocity(self.objectUid)
        # state_robot = p.getLinkState(self.pandaUid, hand_link)[0]
        # state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        # state_finger1 = p.getJointState(self.pandaUid, finger1_joint)[0]
        # state_finger2 = p.getJointState(self.pandaUid, finger2_joint)[1]

        # Compute reward and completition based: the reward is either dense or sparse
        self.distance_threshold = 0.05
        d = goal_distance(newPosition, [0.02, 0.02])
        if d < self.distance_threshold:
            reward = self.compute_reward(newPosition, [0.02, 0.02])
            done = True
        else:
            reward = self.compute_reward(newPosition, [0.02, 0.02])
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            # reward = 0
            done = True

        info = {'object_position': state_object}

        # src -> https://github.com/openai/gym/issues/1503
        grip_pos = np.array([0.0, 0.0, 0.0])
        object_pos = np.array(state_object)
        object_rel_pos = object_pos - grip_pos
        gripper_state = np.array([0.5 * (state_finger1[0] + state_finger2[0])])  # this is the gripper q
        object_rot = np.array(state_object_orienation)  # quaternions?
        object_velp = np.array(twist_object)
        object_velr = np.array(twist_object_orienation)
        grip_velp = np.array([0.0, 0.0, 0.0])  # The velocity of gripper moving
        gripper_vel = np.array([finger1 + finger2])  # The velocity of gripper opening/closing

        obs = np.concatenate([
            grip_pos.ravel(), object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp.ravel(), gripper_vel.ravel(),
        ])
        print(np.shape(obs))
        return obs.copy(), reward, done, info

    def reset(self, *args, **kwargs):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)

        #         planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planeUid = p.loadURDF(os.path.join(dir_path, "floor.urdf"), basePosition=[0.5, 0, 0], useFixedBase=True)

        rest_poses = [0, -0.215, 0, -2.57, 0.0, 2.356, math.pi / 4, 0.0, 0.0, 0.0]
        #         rest_poses = [0, 0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.65],
                                   useFixedBase=True)

        for i in range(len(rest_poses)):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        baseUid = p.loadURDF(os.path.join(dir_path, "base.urdf"), basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
        #         tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tableUid = p.loadURDF(os.path.join(dir_path, "flat_table.urdf"), basePosition=[0.65, 0, 0.0], useFixedBase=True)

        # trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        p.addUserDebugLine([-1, 0, 0.05], [1, 0, 0.05], [0.9, 0.9, 0.9], parentObjectUniqueId=self.pandaUid,
                           parentLinkIndex=8)
        p.addUserDebugLine([0, -1, 0.05], [0, 1, 0.05], [0.9, 0.9, 0.9], parentObjectUniqueId=self.pandaUid,
                           parentLinkIndex=8)
        p.addUserDebugLine([0, 0, -1], [0, 0, 1], [0.9, 0.9, 0.9], parentObjectUniqueId=self.pandaUid,
                           parentLinkIndex=8)

        state_object = [random.uniform(0.5, 0.8), random.uniform(-0.2, 0.2), random.uniform(0.65 + 0.0, 0.65 + 0.2)]
        #         self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        self.objectUid = p.loadURDF(os.path.join(dir_path, "goal.urdf"), basePosition=state_object, useFixedBase=True)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        # self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # return np.array(self.observation).astype(np.float32)

        # src -> https://github.com/openai/gym/issues/1503
        grip_pos = np.array(state_robot)
        object_pos = np.array(state_object)
        object_rel_pos = object_pos - grip_pos
        gripper_state = np.array([0.5 * (state_fingers[0] + state_fingers[1])])  # this is the gripper q
        object_rot = np.array([0, 0, 0, 1])  # quaternions?
        object_velp = np.array([0, 0, 0])
        object_velr = np.array([0, 0, 0])
        grip_velp = np.array([0, 0, 0])  # The velocity of gripper moving
        gripper_vel = np.array([0])  # The velocity of gripper opening/closing

        obs = np.concatenate([
            grip_pos.ravel(), object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp.ravel(), gripper_vel.ravel(),
        ])
        return obs.copy()

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.65 + 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-50,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
