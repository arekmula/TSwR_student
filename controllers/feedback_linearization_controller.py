import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp, 0.1, 0.01)
        self.Kd = np.diag((1, 1)) * [1, -1]
        self.Kp = np.diag((1, 1)) * [1.3, 1.2]

    def calculate_control(self, x, _q_d_ddot, _q_d_dot, _q_d):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        print("x: ", x)
        print("v: ", _q_d_ddot)

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
        M_v = np.dot(self.model.M(x), q_d_ddot)
        C_q = np.dot(self.model.C(x), q_d_dot)
        tau = M_v + C_q

        return tau
