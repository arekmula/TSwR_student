import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel



class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # Use parameters from manipulators/mm_planar_2dof.py
        model1 = ManiuplatorModel(Tp, 0.1, 0.01)
        model2 = ManiuplatorModel(Tp, 0.11, 0.01)
        model3 = ManiuplatorModel(Tp, 0.1, 0.012)
        self.models = [model1, model2, model3]
        self.i = 0

        self.Kd = np.diag((1, 1)) * [1, -1]
        self.Kp = np.diag((1, 1)) * [1.3, 1.2]

    def choose_model(self, x, u, x_dot):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        e_list = []  # list of errors

        # for each model calculate xmi_dot, reshape x_dot and calculate error
        for model in self.models:
            xmi_dot = np.dot(model.M(x), u) + np.dot(model.C(x), x[2:].reshape(-1, 1))
            x_dot = x_dot[:2].reshape(-1, 1)
            # print(f"xmi_dot size {xmi_dot.shape} and x size {x_dot.shape}")
            e_list.append(x_dot - xmi_dot)
        # choose model with smallest error
        e_list = np.sum(np.abs(e_list), axis=1)
        i_before = self.i
        self.i = np.argmin(e_list)
        if self.i != i_before:
            print("Choosed new model")


    def calculate_control(self, x, q_d_ddot, q_d_dot, q_d):
        # create output vector
        tau = np.zeros((2, 1), float)

        v = q_d_ddot + self.Kd.dot(x[2:].reshape(-1, 1) - q_d_dot.reshape(-1, 1)) + self.Kp.dot(
            x[:2].reshape(-1, 1) - q_d.reshape(-1, 1))

        q_dot = x[2:, np.newaxis]
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        M_v = np.dot(M, v)
        C_q = np.dot(C, q_dot)

        tau = M_v + C_q

        return tau
