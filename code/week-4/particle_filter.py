import numpy as np
from helpers import distance

class ParticleFilter:
    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles

    # Set the number of particles.
    # Initialize all the particles to the initial position
    #   (based on esimates of x, y, theta and their uncertainties from GPS)
    #   and all weights to 1.0.
    # Add Gaussian noise to each particle.
    def initialize(self, x, y, theta, std_x, std_y, std_theta):
        self.particles = []
        for i in range(self.num_particles):
            self.particles.append({
                'x': np.random.normal(x, std_x),
                'y': np.random.normal(y, std_y),
                't': np.random.normal(theta, std_theta),
                'w': 1.0,
                'assoc': [],
            })
        self.initialized = True

    # Add measurements to each particle and add random Gaussian noise.
    def predict(self, dt, velocity, yawrate, std_x, std_y, std_theta):
        # Be careful not to divide by zero.
        v_yr = velocity / yawrate if yawrate else 0
        yr_dt = yawrate * dt
        for p in self.particles:
            # We have to take care of very small yaw rates;
            #   apply formula for constant yaw.
            if np.fabs(yawrate) < 0.0001:
                xf = p['x'] + velocity * dt * np.cos(p['t'])
                yf = p['y'] + velocity * dt * np.sin(p['t'])
                tf = p['t']
            # Nonzero yaw rate - apply integrated formula.
            else:
                xf = p['x'] + v_yr * (np.sin(p['t'] + yr_dt) - np.sin(p['t']))
                yf = p['y'] + v_yr * (np.cos(p['t']) - np.cos(p['t'] + yr_dt))
                tf = p['t'] + yr_dt
            p['x'] = np.random.normal(xf, std_x)
            p['y'] = np.random.normal(yf, std_y)
            p['t'] = np.random.normal(tf, std_theta)

    # Find the predicted measurement that is closest to each observed
    #   measurement and assign the observed measurement to this
    #   particular landmark.
    def associate(self, predicted, observations):
        associations = []
        # For each observation, find the nearest landmark and associate it.
        #   You might want to devise and implement a more efficient algorithm.
        for o in observations:
            min_dist = -1.0
            for p in predicted:
                dist = distance(o, p)
                if min_dist < 0.0 or dist < min_dist:
                    min_dist = dist
                    min_id = p['id']
                    min_x = p['x']
                    min_y = p['y']
            association = {
                'id': min_id,
                'x': min_x,
                'y': min_y,
            }
            associations.append(association)
        # Return a list of associated landmarks that corresponds to
        #   the list of (coordinates transformed) predictions.
        return associations

    # Update the weights of each particle using a multi-variate
    #   Gaussian distribution.
    def update_weights(self, sensor_range, std_landmark_x, std_landmark_y,
                       observations, map_landmarks):
        # 각각의 particle에 대해서...
        for particle in self.particles:
            # 1. 해당 particle의 위치에서 sensor_range 이내에 있는
            #    landmarks들의 목록을  구한다.
            visible_landmarks = []
            landmark_id = 0
            for landmark_id in map_landmarks:
                dist = distance(map_landmarks[landmark_id], particle)
                if dist <= sensor_range:
                    visible_landmarks.append({'x' : map_landmarks[landmark_id]['x'],
                                              'y' : map_landmarks[landmark_id]['y'],
                                              'id' : landmark_id})

            # 2. 관측된 landmarks(observations)의 좌표를 map 기준 좌표계로 변환한다.
            observed_in_map = []
            for observation in observations:
                x = (particle['x'] + observation['x']*np.cos(particle['t']) 
                                    - observation['y']*np.sin(particle['t']))
                y = (particle['x'] + observation['x']*np.sin(particle['t']) 
                                    - observation['y']*np.cos(particle['t']))
                observed_in_map.append({'x': x, 'y' : y})
            
            # 3. (1)에서 계산한 센서 범위 내의 landmarks와 
            #    (2)에서 map 좌표계로 변환한 관측된 landmarks 중 같은 것끼리 짝지음.
            #    아래 함수는 가장 가까운 landmarks list를 리턴한다.
            assoc = self.associate(visible_landmarks, observed_in_map)

            # 4. assoc에 포함된 각 landmark와 차량 사이의 거리만큼 떨어진 위치에서,
            #    우리가 얻은 observation이 관측된 확률을 1.0부터 시작하여 누적하여 곱한다.
            # 5. 하나의 particle에서 계산한 (4)의 최종 값이 해당 particle의 weight가 된다.
            particle['w'] = 1.0
            particle['assoc'] = []
            for i in range(len(assoc)):
                dist = distance(assoc[i], particle)
                observ_dist = np.sqrt(observations[i]['x']**2 + observations[i]['y']**2)
                s = np.sqrt(std_landmark_x**2 + std_landmark_y**2)
                particle['w'] *= self.norm_pdf(dist, observ_dist, s) + 0.001
                particle['assoc'].append(assoc[i]['id'])

    # Resample particles with replacement with probability proportional to
    #   their weights.
    def resample(self):
        # 각 particle들의 weights만 추출하여 list로 만든다.
        weights = [p['w'] for p in self.particles]
        # 모든 weight의 합이 1이 되도록 normalize 해준다.
        weights = [w/sum(weights) for w in weights]
        # 이 weights 값을 토대로 particle을 샘플링하여 복사본을 만들고,
        # self.particles를 새로 구성한다.
        new_particles = []
        for i in range(0, len(self.particles)):
            sampled_particle = np.random.choice(self.particles, p=weights)
            new_particles.append(sampled_particle.copy())
        
        del(self.particles)
        self.particles = new_particles

    # Choose the particle with the highest weight (probability)
    def get_best_particle(self):
        highest_weight = -1.0
        for p in self.particles:
            if p['w'] > highest_weight:
                highest_weight = p['w']
                best_particle = p

        return best_particle


    def norm_pdf(self, x, m, s):
        one_over_sqrt_2pi = 1 / np.sqrt(2 * np.pi)

        return (one_over_sqrt_2pi / s) * np.exp(-0.5 * ((x - m) / s) ** 2)