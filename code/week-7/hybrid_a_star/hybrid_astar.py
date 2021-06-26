import numpy as np
import math

class HybridAStar:
    # Determine how many grid cells to have for theta-axis.
    NUM_THETA_CELLS = 90

    # Define min, max, and resolution of steering angles
    omega_min = -20
    omega_max = 20
    omega_step = 5

    # A very simple bicycle model
    speed = 1.5
    length = 0.5

    distance_method = "L2"

    # Initialize the search structure.
    def __init__(self, dim):
        self.dim = dim
        self.closed = np.zeros(self.dim, dtype=np.int)
        self.came_from = np.full(self.dim, None)

    # Expand from a given state by enumerating reachable states.
    def expand(self, current, goal):
        g = current['g']
        x, y, theta = current['x'], current['y'], current['t']

        # The g value of a newly expanded cell increases by 1 from the
        # previously expanded cell.
        g2 = g + 1
        next_states = []

        # delta_t는 앞바퀴의 각도를 뜻함.
        # delta_t 각도를 [omega_min, omega_max] 범위 내에서 omega_step 간격마다 이산적으로 정의해두고 
        # for문을 돌며 하나씩 전부 고려한다.
        for delta_t in range(self.omega_min, self.omega_max, self.omega_step):
            # TODO: implement the trajectory generation based on
            # a simple bicycle model.
            # Let theta2 be the vehicle's heading (in radian)
            # between 0 and 2 * PI.
            # Check validity and then add to the next_states list.
            omega = self.speed / self.length * np.tan(delta_t)
            next_x = x + (self.speed * np.cos(theta))
            next_y = y + (self.speed * np.sin(theta))
            # theta 값을 normalize 해준다.
            next_theta = self.theta_normalize(theta + omega)
            next_g = g2
            next_f = next_g + self.heuristic(next_x, next_y, goal, self.distance_method)
            state = {
                'x': next_x,
                'y': next_y,
                't': next_theta,
                'g': next_g,
                'f': next_f
            }
            next_states.append(state)

        # 위의 delta_t_list에서의 각각에 대해서 state가 나오므로
        # len(delta_t_list)만큼 return 된다.
        return next_states

    # Perform a breadth-first search based on the Hybrid A* algorithm.
    def search(self, grid, start, goal):
        # Initial heading of the vehicle is given in the
        # last component of the tuple start.
        theta = start[-1]
        # Determine the cell to contain the initial state, as well as
        # the state itself.
        stack = self.theta_to_stack_num(theta)
        g = 0
        s = {
            'f': self.heuristic(start[0], start[1], goal, self.distance_method),
            'g': g,
            'x': start[0],
            'y': start[1],
            't': theta,
        }
        self.final = s
        # Close the initial cell and record the starting state for
        # the sake of path reconstruction.
        # 이미 방문했던 곳을 저장하는 array. >> 3차원
        self.closed[stack][self.idx(s['x'])][self.idx(s['y'])] = 1
        # 현재 상태에 도달하기 위해 어떤 경로를 거쳐서 왔는지 저장하는 array.
        self.came_from[stack][self.idx(s['x'])][self.idx(s['y'])] = s
        total_closed = 1
        opened = [s]
        # Examine the open list, according to the order dictated by
        # the heuristic function.
        while len(opened) > 0:
            # TODO: implement prioritized breadth-first search
            # for the hybrid A* algorithm.

            # opened를 정렬한 다음 현재 state를 하나 꺼내온다.
            opened.sort(key=lambda s : s['f'], reverse=True)
            curr = opened.pop()
            x, y = curr['x'], curr['y']

            # goal에 도달했다면 found를 True로 하고 리턴.
            if (self.idx(x), self.idx(y)) == goal:
                self.final = curr
                found = True
                break

            # Compute reachable new states and process each of them.
            # 다음 state가 reachable 한지 검사하고 opened 리스트에 추가한다.
            next_states = self.expand(curr, goal)
            for n in next_states:
                # 가져온 state의 x, y, theta 정보를 구한다.
                n_x, n_y, n_theta = n['x'], n['y'], n['t']

                n_stack_idx = self.theta_to_stack_num(n_theta)
                if n_stack_idx == self.closed.shape[0]:
                    n_stack_idx -= 1
                    
                n_x_idx = self.idx(n_x)
                n_y_idx = self.idx(n_y)
                if n_x_idx >= self.closed.shape[1]:
                    continue
                if n_y_idx >= self.closed.shape[2]:
                    continue

                # 1. n_x, n_y 위치가 장애물이 없는 곳인지
                # 2. map 범위를 벗어나지 않는지
                # 3. 이미 방문했던 곳(closed == 1)이 아닌지를 확인한다.
                if grid[n_x_idx][n_y_idx] != 1 and 0 <= n_x_idx < grid.shape[0] and \
                    0 <= n_y_idx < grid.shape[1] and self.closed[n_stack_idx][n_x_idx][n_y_idx] != 1:
                    # 대각선에 위치한 cell로 이동하면서
                    # 장애물이 있는 cell을 통과했는지를 먼저 검사한다.
                    # 장애물을 통과하는 경로라면 불가능한 경로가 되므로 제외.
                    if self.diagonal_obstacle_2(x, y, n_x, n_y, grid):
                        continue
                    
                    # 해당 지점을 가능한 리스트에 추가
                    opened.append(n)
                    # n 직전에 거쳤던 state는 현재의 state가 된다.
                    self.came_from[n_stack_idx][n_x_idx][n_y_idx] = curr
                    # 현재 state를 방문한 리스트에 표시
                    self.closed[n_stack_idx][n_x_idx][n_y_idx] = 1

        else:
            # We weren't able to find a valid path; this does not necessarily
            # mean there is no feasible trajectory to reach the goal.
            # In other words, the hybrid A* algorithm is not complete.
            found = False

        return found, total_closed

    # Calculate the stack index of a state based on the vehicle's heading.
    # theta 값이 주어지면 heading을 세분화한 stack에서
    # 몇 번째 idx에 가야하는 theta 값인지를 판단하는 함수.
    def theta_to_stack_num(self, theta):
        # TODO: implement a function that calculate the stack number
        # given theta represented in radian. Note that the calculation
        # should partition 360 degrees (2 * PI rad) into different
        # cells whose number is given by NUM_THETA_CELLS.
        
        interval = 360 / self.NUM_THETA_CELLS

        # 라디안 값인 theta를 360도 기준으로 변환
        theta = math.degrees(theta)

        # idx 값은 theta 값을 interval로 나눈 몫이 된다.
        idx = int(theta // interval)
        if idx == self.NUM_THETA_CELLS:
            idx -= 1

        return idx

    # Calculate the index of the grid cell based on the vehicle's position.
    # x, y 포지션이 주어졌을 때 몇 번째 인덱스를 갖는 cell에 있는지를 리턴하는 함수.
    def idx(self, pos):
        # We simply assume that each of the grid cell is the size 1 X 1.
        return int(np.floor(pos))

    # Implement a heuristic function to be used in the hybrid A* algorithm.
    # A* 알고리즘에서는 goal까지의 남은 칸 수를 장애물을 고려하지 않고 따졌는데
    # Hybrid A*에서는 x, y가 연속한... 정수가 아닌 포지션이기 때문에
    # 어떻게 구현할지?
    def heuristic(self, x, y, goal, distance_method):
        # TODO: implement a heuristic function.
        # goal은 (x, y) 형태.
        if distance_method == "L1":
            # 현재 지점과 goal 지점 사이의 L1-distance를 측정
            distance = np.abs(goal[0] - x) + np.abs(goal[1] - y)
        else:
            # 현재 지점과 goal 지점 사이의 L2-distance를 측정
            distance = math.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)

        return distance

    # Reconstruct the path taken by the hybrid A* algorithm.
    def reconstruct_path(self, start, goal):
        # Start from the final state, and follow the link to the
        # previous state using the came_from matrix.
        curr = self.final
        x, y = curr['x'], curr['y']
        path = []
        while x != start[0] and y != start[1]:
            path.append(curr)
            stack = self.theta_to_stack_num(curr['t'])
            x, y = curr['x'], curr['y']
            curr = self.came_from[stack][self.idx(x)][self.idx(y)]
        # Reverse the path so that it begins at the starting state
        # and ends at the final state.
        path.reverse()
        return path

    # 입력으로 받은 theta 값을 0 ~ 2pi 범위로 normalize 하는 함수.
    def theta_normalize(self, theta):
        # theta가 양수라면 2pi로 나눈 나머지
        if (theta > 0):
            norm_theta = theta % (2*np.pi)
        # theta가 음수라면 절대값을 취한 후 2pi로 나눈 나머지를 2pi에서 뺀다.
        else:
            norm_theta = 2*np.pi - (np.abs(theta) % (2*np.pi))
        
        return norm_theta

    # 대각선에 위치한 cell로 이동할 때 
    # (x, y)와 (n_x, n_y)를 지나는 직선의 x절편, y절편을 구하여
    # 중간에 장애물이 있는 cell을 통과했는지 검사하는 함수
    # 이 함수에서는 기존의 좌표축이 아닌 일반적인 xy-평면으로 바꾸어 계산한다.
    def diagonal_obstacle_1(self, x, y, n_x, n_y, grid):
        # 먼저 (x, y)와 (n_x, n_y) 사이의 기울기를 구한다.
        # 좌표축이 반대로 설정되어 있음.
        # (x 증가량 / y 증가량)
        # 일반적인 xy-평면으로 생각하기 때문에 -를 곱함.
        a = -(n_x - x) / (n_y - y)

        # 원점을 서로 대각선에 위치해 있는 cell들이 맞닿는 꼭짓점으로 설정한다.
        # 좌표축도 일반적인 xy-평면처럼 만듦
        o_x = np.floor(x) if x >= n_x else np.floor(n_x)
        o_y = np.floor(y) if y >= n_y else np.floor(n_y)

        p1_x = -(o_y - y) if o_y >= y else y - o_y
        p1_y = o_x - x if o_x >= x else -(x - o_x)

        # y 절편
        y_inter = p1_y - a*p1_x
        # x 절편
        x_inter = -y_inter/a

        # x절편과 y절편을 이용하여 직선이 대각으로 마주보는 사분면 외에 지나는 사분면을 구하고
        # 해당 사분면이 장애물인지를 확인한다.
        o_x = int(o_x)
        o_y = int(o_y)
        if 0 <= o_x < grid.shape[0] and 0 <= o_y < grid.shape[1]:
            if y_inter >= 0 and x_inter >= 0:
                if grid[o_x-1][o_y] == 1:
                    return True
            elif y_inter > 0 and x_inter < 0:
                if grid[o_x-1][o_y-1] == 1:
                    return True
            elif y_inter < 0 and x_inter < 0:
                if grid[o_x][o_y-1] == 1:
                    return True
            elif y_inter < 0 and x_inter > 0:
                if grid[o_x][o_y] == 1:
                    return True
        
        return False

    # (x, y) (n_x, n_y) 두 점을 지나는 직선을 따라 조금씩 이동하면서
    # 중간에 지나는 cell이 장애물인지를 검사하는 함수.
    def diagonal_obstacle_2(self, x, y, n_x, n_y, grid):
        # 좌표축이 반대로 설정되어 있음.
        # (x 증가량 / y 증가량)
        a = (n_x - x) / (n_y - y)

        # 직선의 방정식 : y = a(x - x1) + y1
        # x1, y1에는 인자로 받은 x, y값이 들어간다.

        # (x, y)지점에서 (n_x, n_y)지점 방향으로 y값을 step씩 늘리면서 
        # 중간에 장애물을 통과하는지를 판단한다.
        if (y != n_y):
            inter_y_list = np.linspace(y, n_y, int(self.speed//0.1))
            inter_x_list = a*(inter_y_list - y) + x

            for inter_x, inter_y in zip(inter_x_list, inter_y_list):
                inter_x_idx = self.idx(inter_x)
                inter_y_idx = self.idx(inter_y)

                if 0 <= inter_x_idx < grid.shape[0] and 0 <= inter_y_idx < grid.shape[1] and \
                    grid[inter_x_idx][inter_y_idx] == 1:
                    return True
        # y와 n_y 값이 동일하다면 y = c인 직선이므로
        # (x, y)지점에서 (n_x, n_y)지점 방향으로 x값을 step씩 늘리면서 
        # 중간에 장애물을 통과하는지를 판단한다.
        else:
            inter_x_list = np.linspace(x, n_x, int(self.speed//0.1))
            inter_y_list = [y]*len(inter_x_list)

            for inter_x, inter_y in zip(inter_x_list, inter_y_list):
                inter_x_idx = self.idx(inter_x)
                inter_y_idx = self.idx(inter_y)
                
                if 0 <= inter_x_idx < grid.shape[0] and 0 <= inter_y_idx < grid.shape[1] and \
                    grid[inter_x_idx][inter_y_idx] == 1:
                    return True
        
        
        return False
