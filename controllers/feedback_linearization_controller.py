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
        tau = self.model.M(x) * np.transpose(v) + self.model.C(x) * np.transpose([q1_dot, q2_dot])

        return tau
