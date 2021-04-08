import numpy as np
from math import sqrt
from math import atan2
from tools import Jacobian

class KalmanFilter:
    def __init__(self, x_in, P_in, F_in, H_in, R_in, Q_in):
        self.x = x_in       # x = [px, py, vx, vy]
        self.P = P_in
        self.F = F_in
        self.H = H_in
        self.R = R_in
        self.Q = Q_in

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Calculate new estimates
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    # 입력으로 받는 z = [rho, phi, rho_dot]
    # 이 함수가 호출될 때는 외부에서 이미 저 위의 predict() 함수가 실행된 상태이다.
    # 즉, x -> x', P -> P'으로 업데이트 된 상태.
    def update_ekf(self, z):
        def h(x):
            rho = sqrt(x[0]**2 + x[1]**2)            
            phi = atan2(x[1], x[0])
            rho_dot = (x[0]*x[2] + x[1]*x[3]) / rho

            return np.array([rho, phi, rho_dot])

        # 1. Compute Jacobian Matrix H_j
        H_j = Jacobian(self.x)
        # 2. Calculate S = H_j * P' * H_j^T + R
        S = np.dot(np.dot(H_j, self.P), H_j.T) + self.R
        # 3. Calculate Kalman gain K = P' * Hj^T * S^-1
        K = np.dot(np.dot(self.P, H_j.T), np.linalg.inv(S))
        # 4. Estimate y = z - h(x')
        y = z - h(self.x)
        # 5. Normalize phi so that it is between -PI and +PI
        y[1] = y[1] % np.pi if y[1] > 0 else -(-y[1] % np.pi)
        # 6. Calculate new estimates
        #    x = x' + K * y
        #    P = (I - K * H_j) * P
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(4) - np.dot(K, H_j)), self.P)