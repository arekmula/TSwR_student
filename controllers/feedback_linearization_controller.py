import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, v):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        q1_d_dot, q2_d_dot = v

        # create q** vector from given v
        q_d_dot = np.array([[q1_d_dot[0]],
                            [q2_d_dot[0]]])

        # create q* vector from given x
        q_dot = np.array([[q1_dot],
                      [q2_dot]])

        # create output vector
        tau = np.zeros((2, 1), float)

        # calculate tau from equation no 20
        M_v = np.dot(self.model.M(x), q_d_dot)
        C_q = np.dot(self.model.C(x), q_dot)
        tau = M_v + C_q

        return tau
