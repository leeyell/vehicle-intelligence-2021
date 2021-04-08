# Week 3 - Kalman Filters, EKF and Sensor Fusion

---

[//]: # (Image References)
[kalman-result]: ./kalman_filter/graph.png
[EKF-results]: ./EKF/plot.png

## Kalman Filter Example

In directory [`./kalman_filter`](./kalman_filter), a sample program for a small-scale demonstration of a Kalman filter is provided. Run the following command to test:

```
$ python testKalman.py
```

This program consists of four modules:

* `testKalman.py` is the module you want to run; it initializes a simple Kalman filter and estimates the position and velocity of an object that is assumed to move at a constant speed (but with measurement error).
* `kalman.py` implements a basic Kalman fitler as described in class.
* `plot.py` generates a plot in the format shown below.
* `data.py` provides measurement and ground truth data used in the example.

The result of running this program with test input data is illustrated below:

![Testing of Kalman Filter Example][kalman-result]

Interpretation of the above results is given in the lecture.

In addition, you can run `inputgen.py` to generate your own sample data. It will be interesting to experiment with a number of data sets with different characteristics (mainly in terms of variance, i.e., noise, involved in control and measurement).

---

## Assignment - EFK & Sensor Fusion Example

In directory [`./EKF`](./EKF), template code is provided for a simple implementation of EKF (extended Kalman filter) with sensor fusion. Run the following command to test:

```
$ python run.py
```

The program consists of five modules:

* `run.py` is the modele you want to run. It reads the input data from a text file ([data.txt](./EKF/data.txt)) and feed them to the filter; after execution summarizes the result using a 2D plot.
* `sensor_fusion.py` processees measurements by (1) adjusting the state transition matrix according to the time elapsed since the last measuremenet, and (2) setting up the process noise covariance matrix for prediction; selectively calls updated based on the measurement type (lidar or radar).
* `kalman_filter.py` implements prediction and update algorithm for EKF. All the other parts are already written, while completing `update_ekf()` is left for assignment. See below.
* `tools.py` provides a function `Jacobian()` to calculate the Jacobian matrix needed in our filter's update algorithm.
*  `plot.py` creates a 2D plot comparing the ground truth against our estimation. The following figure illustrates an example:

![Testing of EKF with Sensor Fusion][EKF-results]

### Assignment

Complete the implementation of EKF with sensor fusion by writing the function `update_ekf()` in the module `kalman_filter`. Details are given in class and instructions are included in comments.

***

## Report

#### update_ekf
KalmanFilter 클래스 객체의 멤버 함수이며, 다음과 같은 parameter를 input으로 받는다.
__Input__
* z: [rho, phi, rho_dot]. Radar 센서의 measurement 값이다. main으로 실행되는 함수로부터 입력받는다.

매 time t에서의 measurement가 입력으로 들어올 때마다 위의 파라미터를 통해 클래스 객체가 가지고 있는 x (px, py, vx, vy)와 P를 업데이트한다.

```
# 입력으로 받는 z = [rho, phi, rho_dot]
# 이 함수가 호출될 때는 외부에서 이미 저 위의 predict() 함수가 실행된 상태이다.
# 즉, x -> x', P -> P'으로 업데이트 되어 있음.
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
```