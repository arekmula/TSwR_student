import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3, r3):
        self.Tp = Tp
        self.l1 = 0.5

        self.r1 = 0.25
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.25
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        
        # calculate substitutions: alpha, beta and gamma from equation no (14)
        self.alpha = self.I_1 + self.I_2 + self.m1 * self.r1 ** 2 + self.m2 * (self.l1 ** 2 + self.r2 ** 2) +\
                     self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        self.beta = self.m2 * self.l1 * self.r2 + self.m3 * self.l1 * self.l2
        self.gamma = self.I_2 + self.m2 * self.r2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        M = np.zeros((2, 2), float)  # create 2x2 array

        # fill matrix with values from equation no 20
        M[0, 0] = self.alpha + 2 * self.beta * np.cos(q2)
        M[0, 1] = self.gamma + self.beta * np.cos(q2)
        M[1, 0] = self.gamma + self.beta * np.cos(q2)
        M[1, 1] = self.gamma

        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        C = np.zeros((2, 2), float)

        # fill matrix with values from equation no 20
        C[0][0] = -self.beta * np.sin(q2) * q2_dot
        C[0][1] = -self.beta * np.sin(q2) * (q1_dot + q2_dot)
        C[1][0] = self.beta * np.sin(q2) * q1_dot
        C[1][1] = 0

        return C
