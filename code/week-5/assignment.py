import numpy as np
import itertools

# Given map
grid = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1]
])

# List of possible actions defined in terms of changes in
# the coordinates (y, x)
# 현재 차가 보고 있는 방향 theta로 한 칸 이동.
forward = [
    (-1,  0),   # Up
    ( 0, -1),   # Left
    ( 1,  0),   # Down
    ( 0,  1),   # Right
]

# Three actions are defined:
# - right turn & move forward
# - straight forward
# - left turn & move forward
# Note that each action transforms the orientation along the
# forward array defined above.
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

init = (4, 3, 0)    # Representing (y, x, o), where
                    # o denotes the orientation as follows:
                    # 0: up
                    # 1: left
                    # 2: down
                    # 3: right
                    # Note that this order corresponds to forward above.
goal = (2, 0)
cost = (2, 1, 20)   # Cost for each action (right, straight, left)

# EXAMPLE OUTPUT:
# calling optimum_policy_2D with the given parameters should return
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]

def optimum_policy_2D(grid, init, goal, cost):
    # Initialize the value function with (infeasibly) high costs.
    value = np.full((4, ) + grid.shape, 999, dtype=np.int32)
    # Initialize the policy function with negative (unused) values.
    policy = np.full((4,) + grid.shape, -1, dtype=np.int32)
    # Final path policy will be in 2D, instead of 3D.
    policy2D = np.full(grid.shape, ' ')

    # Apply dynamic programming with the flag change.
    change = True
    while change:
        change = False
        # This will provide a useful iterator for the state space.
        p = itertools.product(
            range(grid.shape[0]),
            range(grid.shape[1]),
            range(len(forward))
        )
        # Compute the value function for each state and
        # update policy function accordingly.
        for y, x, t in p:
            # Mark the final state with a special value that we will
            # use in generating the final path policy.
            # 목적지를 표시
            if (y, x) == goal and value[(t, y, x)] > 0:
                value[(t, y, x)] = 0
                policy[(t, y, x)] = -100
                change = True
            # Try to use simple arithmetic to capture state transitions.
            # value function과 policy function 계산.
            elif grid[(y, x)] == 0:
                # 할 수 있는 모든 action
                for action_idx in range(len(action)):
                    # 해당 action을 취한 후 차량이 보고 있을 방향
                    theta = (t + action[action_idx]) % 4
                    # 해당 action을 취한 후 차량의 위치
                    y_2 = y + forward[theta][0]
                    x_2 = x + forward[theta][1]

                    # map 안의 갈 수 있는 범위라면
                    if (x_2 >= 0 and x_2 < grid.shape[1]) and (y_2 >= 0 and y_2 < grid.shape[0]) and grid[y_2, x_2] == 0:
                        cur_cost = value[theta, y_2, x_2] + cost[action_idx]

                        # 방금의 value가 더 적은 cost였다면 업데이트
                        if cur_cost < value[t, y, x]:
                            value[t, y, x] = cur_cost
                            policy[t, y, x] = action[action_idx]
                            change = True
                            
    # 시작 위치
    y, x, theta = init
    policy2D[y, x] = '#'
    # 목표 지점에 도착할 때까지
    while policy[theta, y, x] != -100:
        # 'R'
        if policy[theta, y, x] == action[0]:
            theta_2 = (theta - 1) % 4
        elif policy[theta, y, x] == action[1]:
            theta_2 = theta
        else:
            theta_2 = (theta + 1) % 4
        
        # 이동
        y += forward[theta_2][0]
        x += forward[theta_2][1]
        theta = theta_2

        policy2D[y, x] = '*' if policy[theta, y, x] == -100 else action_name[policy[theta, y, x] + 1]



    # Return the optimum policy generated above.
    return policy2D

print(optimum_policy_2D(grid, init, goal, cost))
