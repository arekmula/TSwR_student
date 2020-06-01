import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel



class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # Use parameters from manipulators/mm_planar_2dof.py
        model1 = ManiuplatorModel(Tp, 0.1, 0.05)
        model2 = ManiuplatorModel(Tp, 0.01, 0.01)
        model3 = ManiuplatorModel(Tp, 1., 0.3)
        self.models = [model1, model2, model3]
        self.i = 0
        self.Kd = np.diag((1, 1)) * [1, -1]
        self.Kp = np.diag((1, 1)) * [1.3, 1.2]

    def choose_model(self, x, u, x_dot):

        zeros = np.zeros((2, 2), dtype=np.float32)
        e_list = []

        for model in self.models:
            invM = np.linalg.inv(model.M(x))
            A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1),
                                np.concatenate([zeros, -invM @ model.C(x)], 1)], 0)
            B = np.concatenate([zeros, invM], 0)
            x_m = A @ x[:, np.newaxis] + B @ u
            dif = abs(x_m - x_dot)
            e_list.append(np.sum(dif))

        new_model = np.argmin(e_list)

        if new_model != self.i:
            print("Changed model")
        self.i = new_model

    def calculate_control(self, x, _q_d_ddot, _q_d_dot, _q_d):
        q1, q2, q1_dot, q2_dot = x

        q1_d_dot, q2_d_dot = _q_d_ddot

        # create q** vector from given v
        q_d_ddot = np.array([[q1_d_dot[0]],
                            [q2_d_dot[0]]])

        # create q* vector from given x
        q_d_dot = np.array([[q1_dot],
                      [q2_dot]])

        # create q vector from given x
        q_d = np.array([[q1],
                        [q2]])

        # feedback
        q_d_ddot = q_d_ddot + np.dot(self.Kd, (q_d_dot - _q_d_dot)) + np.dot(self.Kp, (q_d - _q_d))

        # create output vector
        tau = np.zeros((2, 1), float)

        # calculate tau from equation no 20
        M_v = np.dot(self.models[self.i].M(x), q_d_ddot)
        C_q = np.dot(self.models[self.i].C(x), q_d_dot)
        tau = M_v + C_q

        return tau
