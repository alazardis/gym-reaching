import gym
from gym import spaces
from .gws import gws

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

    def __init__(self): #change GUI to DIRECT to disable rendering
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

    def closest_points(self):
        finger1_ID = 9
        finger2_ID = 10
        results1 = p.getClosestPoints(self.pandaUid, self.objectUid, math.inf, linkIndexA = finger1_ID)results1 = {tuple: 23340} ((0, 1, 4, 9, -1, (0.4410865099454862, -0.03376097024598358, 0.8805910742178501), (0.527107514073492, -0.02328899306833008, 0.8344509921307419), (-0.8762074310673812, -0.10666725312080749, 0.46998152607731497), 0.09817424627775233, 0.0, 0.0, (0.0, 0.0, 0.0... View
        distance_r1 = []
        for i in results1:
            distance_r1.append(results1[len(results1), 8])
        results2 = p.getClosestPoints(self.pandaUid, self.objectUid, math.inf, linkIndexA = finger2_ID)
        distance_r2 = []
        for i in results1:
            distance_r2.append(results2[len(results1), 8])
        return total_closest_points

    def compute_reward(self, pandaUid, objectUid):
        gws_matrix = gws(pandaUid, objectUid)
        grasp_quality = max(gws_matrix) if len(gws_matrix) else -self.closest_points()
        return grasp_quality


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

        currentPose = p.getLinkState(self.pandaUid, hand_link)
        currentPosition = currentPose[0]
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

        p.setJointMotorControlArray(self.pandaUid, [finger1_joint, finger2_joint], p.POSITION_CONTROL,
                                    newPosition)

        p.stepSimulation()

        p.performCollisionDetection()

        state_object, state_object_orienation = p.getBasePositionAndOrientation(self.objectUid)
        twist_object, twist_object_orienation = p.getBaseVelocity(self.objectUid)
        # state_robot = p.getLinkState(self.pandaUid, hand_link)[0]
        # state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        # state_finger1 = p.getJointState(self.pandaUid, finger1_joint)[0]
        # state_finger2 = p.getJointState(self.pandaUid, finger2_joint)[1]

        # Compute reward and completition based: the reward is either dense or sparse
        gws_quality = self.compute_reward(self.pandaUid, self.objectUid)
        min_gws_quality = 0.5
        min_steps_done = math.round(0.5 / (1.0/240.0))

        if gws_quality > min_gws_quality:
            self.__sim_done += 1
        else:
            self.__sim_done = 0

        if self.__sim_done > min_steps_done:
            done = True
        else:
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
        gripper_vel = np.array([0])  # The velocity of gripper opening/closing

        obs = np.concatenate([
            grip_pos.ravel(), object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp.ravel(), gripper_vel.ravel(),
        ])
        # print(np.shape(obs))
        return obs.copy(), reward, done, info

    def reset(self, *args, **kwargs):
        self.step_counter = 0
        self.__sim_done = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, 0)

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

        state_object = [0.45, 0, 0.825]
        state_object_orientation = p.getQuaternionFromEuler([random.uniform(0, 2.0 * math.pi),
                                                             random.uniform(0, 2.0 * math.pi),
                                                             random.uniform(0, 2.0 * math.pi)])
        #         self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        self.objectUid = p.loadURDF(os.path.join(dir_path, "goal.urdf"), basePosition=state_object,
                                    baseOrientation=state_object_orientation, useFixedBase=False)
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

        #Tonio's solution
        if "return_info" in kwargs.keys():
            return obs.copy(), {}
        else:
            return obs.copy()

    def render(self, mode='human'): #is this the one to remove to disable rendering?
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
