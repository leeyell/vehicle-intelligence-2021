# Week 2 - Markov Localization

---

[//]: # (Image References)
[plot]: ./markov.gif

## Assignment

You will complete the implementation of a simple Markov localizer by writing the following two functions in `markov_localizer.py`:

* `motion_model()`: For each possible prior positions, calculate the probability that the vehicle will move to the position specified by `position` given as input.
* `observation_model()`: Given the `observations`, calculate the probability of this measurement being observed using `pseudo_ranges`.

The algorithm is presented and explained in class.

All the other source files (`main.py` and `helper.py`) should be left as they are.

If you correctly implement the above functions, you expect to see a plot similar to the following:

![Expected Result of Markov Localization][plot]

If you run the program (`main.py`) without any modification to the code, it will generate only the frame of the above plot because all probabilities returned by `motion_model()` are zero by default.

***

## Report

#### (1) motion_model
main() 함수에서 호출되며 다음과 같은 parameters를 input으로 받는다.
__Input__
* position: main() 함수에서 for문을 돌며 range(map_size)의 값이 하나씩 입력으로 주어진다. 즉 0 ~ 24 범위의 정수값 하나.
* mov: 매 timestep마다 이동하는 값으로 main() 함수에서는 상수 1.0으로 설정되어 있다.
* priors: 이전 시점인 time (t-1)에서의 차량이 map의 모든 position에 있을 확률. 즉, [0에 있을 확률, 1에 있을 확률, ..., 24에 있을 확률]
* map_size: map의 사이즈인 25 int값.
* stdev: control(movement)에 대한 std dev 값으로 main() 함수에서는 상수 1.0으로 설정되어 있다.

motion_model() 함수가 호출되는 main() 함수에서 for문으로 해당 observation 한 건에 대해 map_size만큼의 for문을 돌며 map의 모든 위치를 입력으로 주고 있기 때문에,

이 함수에서는 __주어진 위치 한 곳(position)에 대해 map의 모든 위치에서 이 곳으로 이동했을 확률의 합계를 구하면 된다.

특정 위치 a에서 주어진 위치(position)으로 이동했을 확률은 __(차량이 이전 시점에 a에 있을 확률)*(a부터 position까지 이동하는 확률)__로 계산할 수 있다.

따라서 motion_model() 함수는 map_size만큼의 for-loop을 돌며 위의 곱셈 결과를 합하여 누적한 값을 리턴한다.

```
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
```

#### (2) observation_model
main() 함수에서 호출되며 다음과 같은 parameters를 받는다.
__Input__
* landmarks: 모든 landmarks의 위치인 [3, 9, 14, 23].
* observations: 각 time t에서의 관측 결과. 보이는 landmarks까지의 거리를 측정한 값이 list로 들어온다.
* pseudo_ranges: 차량의 진행 방향에 있는 각 landmark까지의 거리.
* stdev: obseration(measurement)에 대한 std 값으로 main() 함수에서는 상수 1.0으로 설정되어 있다.

Landmarks가 어디에 있는지 관측한 값을 토대로 차량의 위치가 어디에 있는지 belief를 높이는 함수이다.

어떠한 관측 결과를 얻었을 때, 미리 알고 있는 landmarks의 위치 정보를 고려하여 현재 차량의 위치가 어디에 있을지 확률을 리턴한다.
1. 관측된 landmark가 없을 경우 확률을 계산할 수 없고,
2. 관측된 landmark의 수가 차량 앞에 놓여 있을 landmark의 개수보다 많다면 이 관측 결과는 올바른 값이 아니라 사용할 수 없다.
3. 이 관측 결과가 정말로 믿을 만한 결과일 확률은, observation 목록의 각 landmark가 그 거리에서 보일 확률을 모두 곱한 것과 같다.

따라서 observation_model() 함수는 observations만큼의 for-loop을 돌며 차량의 앞에 놓인 landmark의 거리차 정보를 가지고 실제 관측된 위치에서 보일 확률을 곱하여 누적한 값을 리턴한다.

```
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
```