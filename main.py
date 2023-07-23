import matplotlib.colors
import numpy as np
from mazelib import Maze
from mazelib.generate.HuntAndKill import HuntAndKill
from mazelib.generate.CellularAutomaton import CellularAutomaton
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time

#Maze size
SIZE = 12
random_seeds = np.random.randint(1, 200, 1)

#Method to convert mazelib's string maze to a grid
def string2matrix(stringmaze):
    data = np.empty((2 * SIZE + 1, 2 * SIZE + 1), dtype=int)
    for i in range(0, 2 * SIZE + 1):
        for j in range(0, 2 * SIZE + 1):
            position = 2 * (SIZE + 1) * i + j
            if stringmaze[position] == ' ':
                data[i][j] = 0
            elif stringmaze[position] == '#':
                data[i][j] = 1
            elif stringmaze[position] == 'S':
                data[i][j] = 5
            elif stringmaze[position] == 'E':
                data[i][j] = 6
            elif stringmaze[position] == '+':
                data[i][j] = 8
            elif stringmaze[position] == 'x':
                data[i][j] = 7
    return data

#Plotting method
def showMaze(stringmaze, colors = ['white', 'black', 'green', 'red', 'cyan', 'purple'],
             bounds = [-0.5, 0.5, 1.5, 5.5, 6.5, 7.5, 9]):
    #Initial setup
    plt.rcParams['figure.constrained_layout.use'] = True
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(frameon=True)
    #Converting stringmaze to numeric numpy 2D array
    data = string2matrix(stringmaze)
    #Plotting grid
    plt.grid(axis = 'both', color = 'k', linewidth = 2)
    plt.xticks(np.arange(0.5, data.shape[1], 1))
    plt.yticks(np.arange(0.5, data.shape[0], 1))
    plt.tick_params(bottom = False, top = False, left = False,
                    right = False, labelbottom = False, labelleft = False)
    plt.imshow(data, cmap = cmap, norm = norm)

#Creates auxiliary matrix of nodes' distances from the start
def BFS_matrixbuilder(data, dist_matrix):
    #Initiate variables
    q = []
    visited = []
    #Find start and end
    start = np.argwhere(data == 5)[0]
    q.append(start)
    while(len(q) != 0):
        current_node = q.pop(0)
        i = current_node[0]
        j = current_node[1]
        #Find valid neighbours of current node, ignoring walls
        neighbours = []
        if i>0 and dist_matrix[i-1][j] != -1:
                 neighbours.append([i-1, j])
        if i<len(dist_matrix)-1 and dist_matrix[i+1][j] != -1:
                neighbours.append([i+1, j])
        if j>0 and dist_matrix[i][j-1] != -1:
                neighbours.append([i, j-1])
        if j<len(dist_matrix)-1 and dist_matrix[i][j+1] != -1:
                neighbours.append([i, j+1])
        for node in neighbours:
            #Unvisited nodes are assigned distance from start
            if dist_matrix[node[0]][node[1]] == 0 and (node[0] != start[0] or node[1] != start[1]):
                dist_matrix[node[0]][node[1]] = dist_matrix[current_node[0]][current_node[1]] + 1
                q.append(node)
                visited.append(node)
            #Return matrix and exit's precursor if the exit is found
            elif dist_matrix[node[0]][node[1]] == -6:
                return dist_matrix, current_node, visited

#Incorporates shortest path and visited nodes into the string maze
def BFS_pathmaker(stringmaze, dist_matrix, last_node, visited):
     length = 1
     current_node = last_node
     i = current_node[0]
     j = current_node[1]
     stringmaze = stringmaze[:(2*(SIZE+1)*i + j)] + '+' + stringmaze[(2*(SIZE+1)*i + j + 1):]
     visited.remove(current_node)
     while (dist_matrix[i][j] != 1):
         if dist_matrix[i - 1][j] == dist_matrix[i][j]-1:
             i = i-1
         elif dist_matrix[i + 1][j] == dist_matrix[i][j]-1:
             i = i+1
         elif dist_matrix[i][j - 1] == dist_matrix[i][j]-1:
             j = j-1
         elif dist_matrix[i][j + 1] == dist_matrix[i][j]-1:
             j = j+1
         stringmaze = stringmaze[:(2*(SIZE+1)*i + j)] + '+' + stringmaze[(2*(SIZE+1)*i + j + 1):]
         length += 1
         visited.remove([i, j])
     for node in visited:
         i = node[0]
         j = node[1]
         stringmaze = stringmaze[:(2 * (SIZE + 1) * i + j)] + 'x' + stringmaze[(2 * (SIZE + 1) * i + j + 1):]
     return stringmaze, len(visited), length

#Builds the path and retains visited nodes
def DFS_matrixbuilder(dist_matrix, node, visited, heritage):
    #Collect neighbours
    i = node[0]
    j = node[1]
    visited.append(node)
    neighbours = []
    if i > 0 and dist_matrix[i - 1][j] != -1:
        neighbours.append([i - 1, j])
    if i < len(dist_matrix) - 1 and dist_matrix[i + 1][j] != -1:
        neighbours.append([i + 1, j])
    if j > 0 and dist_matrix[i][j - 1] != -1:
        neighbours.append([i, j - 1])
    if j < len(dist_matrix) - 1 and dist_matrix[i][j + 1] != -1:
        neighbours.append([i, j + 1])
    #Pursue sub-maze from each neighbour
    for new in neighbours:
        if (dist_matrix[new[0]][new[1]] == 0 and not (new in visited)):
            #Remember the parent and make recursive call
            heritage[tuple(new)] = tuple(node)
            DFS_matrixbuilder(dist_matrix, new, visited, heritage)
        if dist_matrix[new[0]][new[1]] == -6:
            heritage[tuple(new)] = tuple(node)
            break
        if tuple(np.argwhere(dist_matrix == -6)[0]) in heritage:
            break
    return heritage, visited

#Incorporates DFS path and visited nodes into the string maze
def DFS_pathmaker(stringmaze, dist_matrix, heritage, visited):
    # Constructing Path
    path = []
    current = np.argwhere(dist_matrix == -6)[0]
    while not heritage[tuple(current)] == tuple(np.argwhere(dist_matrix == -5)[0]):
        current = heritage[tuple(current)]
        path.append(current)
    visited.remove(np.argwhere(dist_matrix == -5)[0].tolist())
    # Drawing on string maze
    for node in path:
        i = node[0]
        j = node[1]
        stringmaze = stringmaze[:(2 * (SIZE + 1) * i + j)] + '+' + stringmaze[(2 * (SIZE + 1) * i + j + 1):]
        visited.remove([i, j])
    for node in visited:
        i = node[0]
        j = node[1]
        stringmaze = stringmaze[:(2 * (SIZE + 1) * i + j)] + 'x' + stringmaze[(2 * (SIZE + 1) * i + j + 1):]
    return stringmaze, len(visited), len(path)

def AStar(gridmaze, dist_matrix, node):
    start = np.argwhere(gridmaze == 5)[0]
    q = []
    visited = [node]
    heritage = {tuple(node): None}
    q.append([node[0], node[1], 0])
    while (len(q) != 0):
        q.sort(key = lambda x: x[2])
        current_node = q.pop(0)
        i = current_node[0]
        j = current_node[1]
        # Find valid neighbours of current node, ignoring walls
        neighbours = []
        if i > 0 and dist_matrix[i - 1][j] != -1:
            neighbours.append([i - 1, j, dist_matrix[i-1][j]])
        if i < len(dist_matrix) - 1 and dist_matrix[i + 1][j] != -1:
            neighbours.append([i + 1, j, dist_matrix[i+1][j]])
        if j > 0 and dist_matrix[i][j - 1] != -1:
            neighbours.append([i, j - 1, dist_matrix[i][j-1]])
        if j < len(dist_matrix) - 1 and dist_matrix[i][j + 1] != -1:
            neighbours.append([i, j + 1, dist_matrix[i][j+1]])
        for new in neighbours:
            new_coord = [new[0], new[1]]
            if (not new_coord in visited):
                heritage[(new[0], new[1])] = (current_node[0], current_node[1])
                q.append(new)
                visited.append([new[0], new[1]])
            # Return matrix and exit's precursor if the exit is found
            if dist_matrix[new[0]][new[1]] == -6:
                visited.remove([new[0], new[1]])
                return heritage, visited

def search_analysis(num_sims):
    BFS_time_set = []
    BFS_number_visited_set = []
    BFS_path_length_set = []
    DFS_time_set = []
    DFS_number_visited_set = []
    DFS_path_length_set = []
    AStar_time_set = []
    AStar_number_visited_set = []
    AStar_path_length_set = []
    for i in range(num_sims):
        #Maze creation
        m = Maze(random_seeds[i])
        m.generator = CellularAutomaton(SIZE, SIZE, complexity=1, density=0.5)
        m.generate()
        m.generate_entrances()
        stringmaze = str(m)
        gridmaze = string2matrix(stringmaze)
        start = np.argwhere(gridmaze == 5)[0].tolist()
        exit = np.argwhere(gridmaze == 6)[0].tolist()
        showMaze(stringmaze)

        # BFS execution
        dist_matrix = np.copy(gridmaze)
        dist_matrix[dist_matrix == 1] = -1
        dist_matrix[dist_matrix == 5] = 0
        dist_matrix[dist_matrix == 6] = -6
        BFS_time = time.time()
        BFS_dist_matrix, last_node, BFS_visited = BFS_matrixbuilder(gridmaze, dist_matrix)
        solved_stringmaze, number_visited, path_length = BFS_pathmaker(stringmaze,
                                                                               BFS_dist_matrix, last_node, BFS_visited)
        BFS_time_set.append(time.time() - BFS_time)
        BFS_number_visited_set.append(number_visited)
        BFS_path_length_set.append(path_length)
        showMaze(solved_stringmaze)

        # DFS execution
        dist_matrix = np.copy(gridmaze)
        dist_matrix[dist_matrix == 1] = -1
        dist_matrix[dist_matrix == 5] = -5
        dist_matrix[dist_matrix == 6] = -6

        heritage = {tuple(start): None}
        DFS_time = time.time()
        DFS_heritage, DFS_visited = DFS_matrixbuilder(dist_matrix, start, [], heritage)
        solved_stringmaze, number_visited, path_length = DFS_pathmaker(stringmaze,
                                                                       dist_matrix, DFS_heritage, DFS_visited)
        DFS_time_set.append(time.time() - DFS_time)
        DFS_number_visited_set.append(number_visited)
        DFS_path_length_set.append(path_length)
        showMaze(solved_stringmaze)

        # A* execution
        # Compute Manhattan distances and stitch to backwards cost matrix
        for i in range(1, len(gridmaze) - 1):
            for j in range(1, len(gridmaze[0]) - 1):
                if gridmaze[i][j] == 0:
                    BFS_dist_matrix[i][j] += (abs(i - exit[0]) + abs(j - exit[1]))
        AStar_time = time.time()
        AStar_heritage, AStar_visited = AStar(gridmaze, BFS_dist_matrix, start)
        solved_stringmaze, number_visited, path_length = DFS_pathmaker(stringmaze, dist_matrix,
                                                                       AStar_heritage, AStar_visited)
        AStar_time_set.append(time.time() - AStar_time)
        AStar_number_visited_set.append(number_visited)
        AStar_path_length_set.append(path_length)
        showMaze(solved_stringmaze)

    time_stats = np.zeros((3,2))
    time_stats[0][0] = np.mean(BFS_time_set)
    time_stats[0][1] = np.std(BFS_time_set)
    time_stats[1][0] = np.mean(DFS_time_set)
    time_stats[1][1] = np.std(DFS_time_set)
    time_stats[2][0] = np.mean(AStar_time_set)
    time_stats[2][1] = np.std(AStar_time_set)

    visited_stats = np.zeros((3,2))
    visited_stats[0][0] = np.mean(BFS_number_visited_set)
    visited_stats[0][1] = np.std(BFS_number_visited_set)
    visited_stats[1][0] = np.mean(DFS_number_visited_set)
    visited_stats[1][1] = np.std(DFS_number_visited_set)
    visited_stats[2][0] = np.mean(AStar_number_visited_set)
    visited_stats[2][1] = np.std(AStar_number_visited_set)

    path_stats = np.zeros((3,2))
    path_stats[0][0] = np.mean(BFS_path_length_set)
    path_stats[0][1] = np.std(BFS_path_length_set)
    path_stats[1][0] = np.mean(DFS_path_length_set)
    path_stats[1][1] = np.std(DFS_path_length_set)
    path_stats[2][0] = np.mean(AStar_path_length_set)
    path_stats[2][1] = np.std(AStar_path_length_set)

    print('Time stats:')
    print(time_stats)
    print('Visit stats:')
    print(visited_stats)
    print('Path stats:')
    print(path_stats)

#MDP section

actions = ['^', '>', 'v', '<']
rewards = {0: 0, 1: -5, 5: 0, 6: 10}
noise = 0.2
decay = 0.9
epsilon = 10**(-6)

def generate_rewards(gridmaze, rewards):
    reward_matrix = np.copy(gridmaze)
    for k in rewards:
        reward_matrix[reward_matrix == k] = rewards[k]
    return reward_matrix

def coord_to_pos(coord):
    return int((2*SIZE+1)*coord[0] + coord[1])

def pos_to_coord(p):
    i = int(p/(2*SIZE+1))
    j = int(p%(2*SIZE+1))
    return [i, j]

#Transitions and policies generated for all nodes (including walls), mostly to cover starting point too
def generate_transition(gridmaze, noise):
    transition_matrix = np.zeros((len(gridmaze)*len(gridmaze[0]), len(actions), len(gridmaze)*len(gridmaze[0])))
    for i in range(len(gridmaze)):
        for j in range(len(gridmaze[0])):
            node = [i, j]
            node = coord_to_pos(node)
            #Determine possible next moves
            if gridmaze[i][j] == 0 or gridmaze[i][j] == 5 or gridmaze[i][j] == 6:
                next = np.zeros(len(actions), dtype = 'int')
                for a in range(len(actions)):
                    k = i
                    l = j
                    if actions[a] == '^':
                        k = max(i - 1, 0)
                    elif actions[a] == '>':
                        l = min(j+1, len(gridmaze[0]) - 1)
                    elif actions[a] == 'v':
                        k = min(i+1, len(gridmaze) - 1)
                    elif actions[a] == '<':
                        l = max(j-1, 0)
                    #Stay put if a wall is hit
                    if gridmaze[k][l] == 1:
                        k=i
                        l=j
                    #Fill in next node from current by action
                    next[a] = coord_to_pos([k, l])
            else:
                next = [coord_to_pos([i, j])] * len(actions)
            #Compute transitions
            for a in range(len(actions)):
                transition_matrix[node, a, next[a]] += (1 - noise)
                transition_matrix[node, a, next[(a + 1 ) % len(actions)]] += noise / 2.0
                transition_matrix[node, a, next[(a - 1) % len(actions)]] += noise / 2.0
    return transition_matrix

def random_policy(gridmaze):
    return np.random.randint(len(actions), size = (len(gridmaze)*len(gridmaze[0])))

def print_policy(gridmaze, policy):
    printed_maze = np.copy(gridmaze)
    printed_maze[printed_maze == 1] = -1
    for i in range(len(printed_maze)):
        for j in range(len(printed_maze[0])):
            if printed_maze[i][j] == 0 or printed_maze[i][j] == 5:
                printed_maze[i][j] = policy[coord_to_pos([i, j])]
    return printed_maze

#Modifies values
def policy_evaluation(gridmaze, values, policy, transition, reward_matrix):
    iterations = 1
    diff = 0
    #Iterative policy evaluation to solve Bellman equations
    while(True):
        iterations += 1
        current_values = np.copy(values)
        for i in range(len(gridmaze)):
            for j in range(len(gridmaze[0])):
                node = coord_to_pos([i, j])
                p = transition[node, policy[node]]
                values[node] = np.sum(p * (decay * current_values + reward_matrix.flatten()))
        diff = np.max(np.abs(values - current_values))
        if (diff < epsilon):
            break
    return values

#Modifies policy, update counting to terminate iteration
def policy_improvement(gridmaze, values, policy, transition, reward_matrix):
    updates = 0
    for i in range(len(gridmaze)):
        for j in range(len(gridmaze[0])):
            #Updating policies in wall is useless
            if gridmaze[i][j] == 1:
                continue
            node = coord_to_pos([i, j])
            current_policy = policy[node]
            potential_values = np.zeros(len(actions))
            for a in range(len(actions)):
                p = transition[node, a]
                potential_values[a] = np.sum(p * (decay * values + reward_matrix.flatten()))
            policy[node] = np.argmax(potential_values)
            if current_policy != policy[node]:
                updates += 1
    return policy, updates

def policy_iteration(gridmaze, values, policy, transition, reward_matrix):
    iterations = 1
    max_iterations = 10000
    while(iterations <= max_iterations):
        values = policy_evaluation(gridmaze, values, policy, transition, reward_matrix)
        policy, updates = policy_improvement(gridmaze, values, policy, transition, reward_matrix)
        if updates == 0:
            break
        iterations += 1
    print('PI iterations: ' + str(iterations))
    return policy, iterations, values

def extract_policy(gridmaze, values, transition, reward_matrix):
    policy = np.zeros((len(gridmaze)*len(gridmaze[0]))) - 1
    for i in range(len(gridmaze)):
        for j in range(len(gridmaze[0])):
            if gridmaze[i][j] == 1:
                continue
            node = coord_to_pos([i, j])
            potential_values = np.zeros(len(actions))
            for a in range(len(actions)):
                p = transition[node, a]
                potential_values[a] = np.sum(p * (decay * values + reward_matrix.flatten()))
            policy[node] = np.random.choice(np.argwhere(potential_values == np.max(potential_values))[0])
    return np.asarray(policy, dtype = 'int')

def value_iteration(gridmaze, values, transition, reward_matrix):
    iterations = 1
    diff = 0
    while (True):
        iterations += 1
        current_values = np.copy(values)
        for i in range(len(gridmaze)):
            for j in range(len(gridmaze[0])):
                node = coord_to_pos([i, j])
                potential_values = np.zeros(len(actions))
                for a in range(len(actions)):
                    p = transition[node, a]
                    potential_values[a] = np.sum(p * (decay * values + reward_matrix.flatten()))
                values[node] = np.max(potential_values)
        diff = np.max(np.abs(values - current_values))
        if (diff < epsilon):
            break
    print('VI iterations: ' + str(iterations))
    return extract_policy(gridmaze, values, transition, reward_matrix), iterations, values

def showMDP(gridmaze, values, policy, path = False):
    #Transform values into matrix
    #Transform policy into directional matrix
    maze_copy = np.copy(gridmaze)
    start = np.argwhere(maze_copy == 5)[0]
    exit = np.argwhere(maze_copy == 6)[0]
    length = 0
    maze_copy[maze_copy == 5] = 0
    values_matrix = np.zeros(policy.shape)
    direction_matrix = np.chararray(policy.shape, unicode=True)
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            values_matrix[i][j] = values[coord_to_pos([i, j])]
            if policy[i][j] == -1:
                direction_matrix[i][j] = ' '
            elif policy[i][j] == 6:
                direction_matrix[i][j] = 'E'
            else:
                direction_matrix[i][j] = actions[policy[i][j]]
    if path:
        node = start
        while np.any(node != exit):
            i = node[0]
            j = node[1]
            values_matrix[i][j] = values[coord_to_pos(exit)]
            #Move in the direction of the policy
            if policy[i][j] == 0:
                node = [max(0, i-1),j]
            elif policy[i][j] == 1:
                node = [i, min(len(gridmaze[0])-1, j+1)]
            elif policy[i][j] == 2:
                node = [min(len(gridmaze) - 1, i+1), j]
            elif policy[i][j] == 3:
                node = [i, max(0, j-1)]
            #Break loop if path is not continuous
            if node[0] == i and node[1] == j:
                break
            length += 1

    # Initial setup
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.figure(frameon=True)
    data = maze_copy
    # Plotting grid
    plt.grid(axis='both', color='k', linewidth=2)
    plt.xticks(np.arange(0.5, data.shape[1], 1))
    plt.yticks(np.arange(0.5, data.shape[0], 1))
    # Encoding value in color saturation:
    plt.tick_params(bottom=False, top=False, left=False,
                    right=False, labelbottom=False, labelleft=False)
    plt.imshow(values_matrix, cmap = 'magma', interpolation= 'nearest')
    # Plotting directions
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = plt.text(j, i, direction_matrix[i][j], ha = "center", va = "center", color = 'black')
    #Subtract end square from length
    return length - 1

def plot_iterations(PI_iterations_set, VI_iterations_set):
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots()
    #x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]
    x = np.linspace(0.1, 0.9, 9)
    #x = np.linspace(-10, -1, 10)
    #for i in range(len(x)):
    #    x[i] = 10**(x[i])
    ax.set(xlabel = 'noise', ylabel = 'iterations', title = 'Settings: 10^(-6) tolerance')
    #ax.semilogx(x, PI_iterations_set, linewidth = 2, color = 'purple', label = 'PI')
    #ax.semilogx(x, VI_iterations_set, linewidth = 2, color = 'orange', label = 'VI')
    ax.plot(x, PI_iterations_set[:9], linewidth = 2, color = 'purple', label = 'PI (0.2 noise)')
    ax.plot(x, VI_iterations_set[:9], linewidth = 2, color = 'orange', label = 'VI (0.2 noise)')
    ax.plot(x, PI_iterations_set[9:], linewidth=2, color='red', label='PI (0.6 noise)')
    ax.plot(x, VI_iterations_set[9:], linewidth=2, color='green', label='VI (0.6 noise)', linestyle = 'dotted')
    ax.legend(loc = 'upper left', title = 'Policy', bbox_to_anchor = (1.05, 1))

def MDP_analysis(num_sims):
    PI_times_set = []
    PI_iterations_set = []
    VI_times_set = []
    VI_iterations_set = []
    VI_path = []
    PI_path = []
    #for exp in np.linspace(-10, -1, 10):
        #epsilon = 10**(exp)
    #for decay in [0.5, 0.9]:
    #    for noise in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]:
    #for noise in [0.2, 0.6]:
        #for decay in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for simulation in range(num_sims):
        # Maze creation
        m = Maze(random_seeds[simulation])
        m.generator = CellularAutomaton(SIZE, SIZE, complexity=1, density=0.5)
        m.generate()
        m.generate_entrances()
        stringmaze = str(m)
        gridmaze = string2matrix(stringmaze)
        start = np.argwhere(gridmaze == 5)[0].tolist()
        exit = np.argwhere(gridmaze == 6)[0].tolist()

        #MDP policy iteration
        MDP_PI_maze = np.copy(gridmaze)
        reward_matrix = generate_rewards(MDP_PI_maze, rewards)
        transition_matrix = generate_transition(MDP_PI_maze, noise)
        policy = random_policy(MDP_PI_maze)
        values = np.zeros((len(MDP_PI_maze)*len(MDP_PI_maze[0])))
        values[coord_to_pos(exit)] = rewards[6]

        PI_time = time.time()
        final_policy_PI, final_iterations_PI, final_values_PI = policy_iteration(MDP_PI_maze,
                                                                                 values, policy,
                                                                                 transition_matrix, reward_matrix)
        PI_times_set.append(time.time() - PI_time)
        solved_MDP_PI = print_policy(MDP_PI_maze, final_policy_PI)
        PI_iterations_set.append(final_iterations_PI)

        #MDP value iteration
        MDP_VI_maze = np.copy(gridmaze)
        reward_matrix = generate_rewards(MDP_VI_maze, rewards)
        transition_matrix = generate_transition(MDP_VI_maze, noise)
        values = np.zeros((len(gridmaze)*len(gridmaze[0])))
        values[coord_to_pos(exit)] = rewards[6]

        VI_time = time.time()
        final_policy_VI, final_iterations_VI, final_values_VI = value_iteration(gridmaze,
                                                                                values, transition_matrix,
                                                                                reward_matrix)
        VI_times_set.append(time.time() - VI_time)
        solved_MDP_VI = print_policy(MDP_VI_maze, final_policy_VI)
        VI_iterations_set.append(final_iterations_VI)

        PI_path.append(showMDP(gridmaze, final_values_PI, solved_MDP_PI, path = False))
        VI_path.append(showMDP(gridmaze, final_values_VI, solved_MDP_VI, path = True))

    time_stats = np.zeros((2, 2))
    time_stats[0][0] = np.mean(PI_times_set)
    time_stats[0][1] = np.std(PI_times_set)
    time_stats[1][0] = np.mean(VI_times_set)
    time_stats[1][1] = np.std(VI_times_set)

    #iteration_stats = np.zeros((2, 2))
    #iteration_stats[0][0] = np.mean(PI_iterations_set)
    #iteration_stats[0][1] = np.std(PI_iterations_set)
    #iteration_stats[1][0] = np.mean(VI_iterations_set)
    #iteration_stats[1][1] = np.std(VI_iterations_set)

    path_stats = np.zeros((2, 2))
    path_stats[0][0] = np.mean(PI_path)
    path_stats[0][1] = np.std(PI_path)
    path_stats[1][0] = np.mean(VI_path)
    path_stats[1][1] = np.std(VI_path)

    print('Time stats:')
    print(time_stats)
    #print('Iteration stats:')
    #print(iteration_stats)
    print('Path stats:')
    print(path_stats)

#plot_iterations(PI_iterations_set, VI_iterations_set)

search_analysis(1)

MDP_analysis(1)

plt.show()



