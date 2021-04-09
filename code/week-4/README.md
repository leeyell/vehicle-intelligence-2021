# Week 4 - Motion Model & Particle Filters

---

[//]: # (Image References)
[empty-update]: ./empty-update.gif
[example]: ./example.gif

## Assignment

You will complete the implementation of a simple particle filter by writing the following two methods of `class ParticleFilter` defined in `particle_filter.py`:

* `update_weights()`: For each particle in the sample set, calculate the probability of the set of observations based on a multi-variate Gaussian distribution.
* `resample()`: Reconstruct the set of particles that capture the posterior belief distribution by drawing samples according to the weights.

To run the program (which generates a 2D plot), execute the following command:

```
$ python run.py
```

Without any modification to the code, you will see a resulting plot like the one below:

![Particle Filter without Proper Update & Resample][empty-update]

while a reasonable implementation of the above mentioned methods (assignments) will give you something like

![Particle Filter Example][example]

Carefully read comments in the two method bodies and write Python code that does the job.

***

## Report

#### (1) update_weights
ParticleFilter 클래스의 멤버 함수로써 main() 함수에서 호출되며, 다음과 같은 parameters를 input으로 받는다.
__Input__
* sensor_range: 차량의 위치를 중점으로 센서가 landmarks를 인식할 수 범위
* std_landmark_x: landmark measurement의 x 좌표값에 대한 uncertainty
* std_landmark_y: landmark measurement의 y 좌표값에 대한 uncertainty
* observations: 현재 시각 t에서 관측된 landmarks들의 위치. 차량의 위치를 원점으로 한다.
* map_landmarks: map 정보. 모든 landmarks의 위치가 map 좌표계를 기준으로 나타나 있다.

매 time t에서 관측된 landmark들을 보고 particles의 weights를 업데이트 하는 함수.

모든 landmarks 중 각 particle의 위치에서 sensor_range 범위 이내로 들어온 landmarks를 찾고, observations 정보에서의 각 landmark들이 어떤 것과 대응이 되는지 계산한다.

이 결과와 가우시안 분포를 이용하여, 해당 state에서 그 observation 값을 얻었을 확률을 계산해 particle의 weight로 설정한다.

```
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
```

#### (2) resample():
ParticleFilter 클래스의 멤버 함수로써 main() 함수에서 호출된다.

클래스가 가지고 있는 particles로부터 각 particle들의 weights를 토대로 particle을 샘플링 하고, 이 결과가 다음 time t에서의 particles 목록이 된다.

```
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
```