from helper import norm_pdf

# 이전의 state에 대한 확률분포를 가장 처음 초기화하는 함수.
# 어떤 나무 근처에 있는 차의 위치를 임력으로 받음.
def initialize_priors(map_size, landmarks, stdev):
    # 모든 위치에 대한 확률을 0으로 초기화.
    priors = [0.0] * map_size

    # 차량이 모든 랜드마크의 +/- 1.0 범위(position)에 있다고 확률을 세팅함
    positions = []
    for p in landmarks:                 # landmarks = [3, 9, 14, 23]
        start = int(p - stdev) - 1
        if start < p:
            start += 1
        c = 0
        while start + c <= p + stdev:
            # Gather positions to set initial probability.
            positions.append(start + c)
            c += 1
    # 각 position에 세팅될 확률은 모두 동일 -> 합 = 1
    prob = 1.0 / len(positions)

    # 위에서 계산된 position 위치에 prob만큼의 확률을 준다.
    for p in positions:
        priors[p] += prob

    return priors

# 주어진 pseudo position에 대해 pseudo range를 예측하는 함수
# 차량의 앞에 놓여 있는 각 landmark까지의 거리를 리턴함
# p = range(map_size)의 값이 하나씩 들어옴 = 0, 1, 2, ..., 24
def estimate_pseudo_range(landmarks, p):
    pseudo_ranges = []
    # 각 landmark의 위치를 돌며 pseudo ranges를 추정함
    for landmark in landmarks:
        # dist = p 위치에서 해당 landmark까지의 거리
        dist = landmark - p
        # 차량의 앞에 놓여 있는 landmark라면 리턴 목록에 추가함
        if dist > 0:
            pseudo_ranges.append(dist)

    return pseudo_ranges

# Motion model (assuming 1-D Gaussian dist)
# 내가 움직이고, 그 움직임에 의해서 belief가 약해질 건데 얼만큼 약해질 거냐... 하는 함수
# position = range(map_size)의 값이 하나씩 들어옴 = 0, 1, 2, ..., 24
# priors = [차량이 0에 있을 확률, 1에 있을 확률, 2에 있을 확률, ..., 24에 있을 확률]
def motion_model(position, mov, priors, map_size, stdev):
    # position에서의 확률을 0으로 초기화
    position_prob = 0.0

    # prior positions의 가능한 모든 state space를 돌면서
    # 해당 위치가 과거 position이라 가정하고, 현재 position까지
    # 차량이 움직일 확률을 norm_pdf를 이용하여 계산한다.
    # 이전 시점에 차량이 과거 position에 있었을 확률(priors의 과거 position에 대한 확률값)과
    # 과거 position부터 현재 position까지 이동하는 확률값(norm_pdf)를 곱한다.
    # 모든 범위에 대해서 합하면 최종적으로 time t에 현재 position에 있을 확률이 나오게 된다.
    for p in range(0, map_size):
        position_prob += norm_pdf(position-p, 1, 1) * priors[p]

    return position_prob

# Observation model (assuming independent Gaussian)
# 내 관측에 의해서 belief가 강해지는데 얼만큼 강해질 거냐... 하는 함수
def observation_model(landmarks, observations, pseudo_ranges, stdev):
    # Initialize the measurement's probability to one.
    distance_prob = 1.0

    # (1) 관측된 landmark가 없다면 확률 분포를 계산할 수가 없다.
    # (2) 관측된 landmark의 수가, 차량의 앞에 놓인 landmark까지의 거리 목록(pseudo_ranges)보다
    #     많다면 잘못된 값이다.
    # (3) 이 관측 결과가 옳을 확률은, 목록의 각 landmark가 해당 거리에서 보일 확률을
    #     모두 곱한 것과 같다.
    #     N(해당 랜드마크까지의 관측된 거리, 해당 랜드마크까지의 거리, 1)
    if len(observations) is None:
        return 0.0
    elif len(observations) > len(pseudo_ranges):
        return 0.0
    else:
        for i in range(len(observations)):
            distance_prob *= norm_pdf(observations[i], pseudo_ranges[i], 1)

    return distance_prob

# Normalize a probability distribution so that the sum equals 1.0.
def normalize_distribution(prob_dist):
    normalized = [0.0] * len(prob_dist)
    total = sum(prob_dist)
    for i in range(len(prob_dist)):
        if (total != 0.0):
            normalized[i] = prob_dist[i] / total

    return normalized
