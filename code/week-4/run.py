from particle_filter import ParticleFilter
from plot import plot_2D

# Time elapsed between measurements (sec)
delta_t = 0.1
# Sensor range (m)
sensor_range = 50
# GNSS measurement uncertainty [x (m), y (m), theta (rad)]
pos_std = [0.3, 0.3, 0.01]
# Landmark measurement uncertainty [x (m), y (m)]
landmark_std = [0.3, 0.3]

# map_data.txt 파일로부터 landmarks map을 읽어 온다.
# line 형식 : (landmark x, landmark y, id).
# map: {0 : {'x': 좌표, 'y':좌표}, 1 : {'x': 좌표, 'y':좌표}, ...}
def read_map_from_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    landmarks = {}
    for line in lines:
        tokens = line.split()
        landmarks[int(tokens[2])] = {
            'x': eval(tokens[0]),
            'y': eval(tokens[1]),
        }

    return landmarks

# Read in the measurement (lidar assumed) data line by line and
#   prepare them to be fed to the filter.
def read_measurements_from_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    i = 0
    # file에서 line 3개씩 읽음.
    while i < len(lines):
        # 첫 번째 line
        x, y, theta, velocity, yawrate = map(float, lines[i].split())
        # 두 번째 line: x
        measurement_x = list(map(float, lines[i + 1].split()))
        # 세 번째 line: y
        measurement_y = list(map(float, lines[i + 2].split()))

        measurement = {
            'gnss_x': x,
            'gnss_y': y,
            'gnss_theta': theta,
            'previous_velocity': velocity,
            'previous_yawrate': yawrate,
            'measurement_x': measurement_x,
            'measurement_y': measurement_y,
        }
        i += 3

        yield measurement

# Main driver code
if __name__ == '__main__':
    Map = read_map_from_file('map_data.txt')
    measurements = read_measurements_from_file('measurements.txt')

    # Particle의 수는 30으로 설정하여 객체 생성
    pf = ParticleFilter(30)

    # Fill in the graph data as defined below so that
    #   a 2-D plot can be drawn.
    graph = []
    count = 0
    for m in measurements:
        count += 1
        if not pf.initialized:
            # Initialize the particle set using GNSS measurement.
            pf.initialize(m['gnss_x'], m['gnss_y'], m['gnss_theta'], *pos_std)
        else:
            # Prediction step
            pf.predict(delta_t, m['previous_velocity'],
                       m['previous_yawrate'], *pos_std)

        # Particles의 weight를 업데이트하기 위해 observation의 (x, y)쌍들을 입력으로 줌.
        observations = [{'x': x, 'y': y} for (x, y) in \
                        zip (m['measurement_x'], m['measurement_y'])]
        pf.update_weights(sensor_range, *landmark_std, observations, Map)
        # Resample particles to capture posterior belief distribution.
        pf.resample()

        # 가장 weight가 높은 particle을 선택한다.
        # 이 particle이 vehicle의 position을 나타낸다고 가정함.
        particle = pf.get_best_particle()
        # Print for debugging purposes.
        print("[%d] %f, %f" % (count, particle['x'], particle['y']))

        # - position: 매 time t에서의 ego vehicle의 map coordinate position
        # - particles: set에 포함된 particles의 coordinates list
        # - landmarks: best particle에서 보이는 landmarks의 coordinates list
        graph.append({
            'position': (particle['x'], particle['y']),
            'particles': [(p['x'], p['y']) for p in pf.particles],
            'landmarks': [(Map[l]['x'], Map[l]['y']) \
                          for l in particle['assoc']],
        })
    # Go plot the results.
    plot_2D(graph)
