import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value

    def __repr__(self):  # built in function so that list with objects displays a more understandable format
        return ("Node %d has value %d" % (self.index, self.value))

    def get_neighbours(self):
        return [i for i in range(len(self.connections)) if self.connections[
            i] == 1]  # comprehension list that displays all indexes of the neighbours of node chosen


class Queue:  # queue class imported from previous assignments
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)

    def pop(self, position):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(position)

    def is_empty(self):
        return len(self.queue) == 0


class Network:
    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(
            self):  # question 1 for task 3, uses get neighbours function and computes average for each node
        total = 0
        for i in range(len(self.nodes)):
            total += len(self.nodes[i].get_neighbours())
        return total / len(self.nodes)  # mean part

    def get_mean_path_length(
            self):  # question 2 for task 3, uses breadth-first search to find the mean path from one node to all others
        total = 0
        for i in range(len(self.nodes)):  # for every node
            total1 = 0
            for elem in self.nodes:
                if elem != self.nodes[i]:  # for every different node than the one looped through
                    start_node = self.nodes[i]
                    goal = elem
                    search_queue = Queue()
                    search_queue.push(self.nodes[i])
                    visited = []

                    while not search_queue.is_empty():
                        node_to_check = search_queue.pop(0)

                        if node_to_check == goal:  # when we end up on the destination
                            break

                        for neighbour_index in node_to_check.get_neighbours():
                            neighbour = self.nodes[neighbour_index]
                            if neighbour_index not in visited:
                                search_queue.push(neighbour)
                                visited.append(neighbour_index)
                                neighbour.parent = node_to_check
                        node_to_check = goal

                        # We make sure the start node has no parent.

                        start_node.parent = None
                        route = []

                        # Loop over node parents until we reach the start.
                    while node_to_check.parent:
                        # Add node to our route
                        route.append(node_to_check)

                        # Update node to be the parent of our current node
                        node_to_check = node_to_check.parent

                        # Add the start node to the route
                    route.append(node_to_check)
                    # print(route)
                    total2 = len(route) - 1
                    # Reverse and print the route
                    total1 += total2  # sums to compute the average
            total1 /= (len(self.nodes) - 1)
            total += total1
        total /= len(self.nodes)
        return total

    def get_mean_clustering(self):  # question 3 for task 3, clustering coefficient
        count = 0
        print(self.nodes)
        for i in range(len(self.nodes)):  # for all nodes
            minilist = self.nodes[i].get_neighbours()  # neighbours of node chosen
            n = len(minilist)  # number of neighbours
            possible_connection = n * (n - 1) / 2  # formula
            # print('possible connection=',possible_connection)
            if possible_connection != 0:  # cannot divide by 0 at the end
                count1 = 0  # count of edges between neighbours
                Biglist = []
                edges = []  # list to check for same edges such as 1-0 and 0-1
                for j in range(len(minilist)):
                    Biglist.append(
                        (minilist[j], self.nodes[minilist[j]].get_neighbours()))  # neighbours of these nodes
                for m in range(len(Biglist)):
                    for n in range(len(Biglist[m][1])):
                        if Biglist[m][1][n] in minilist and {Biglist[m][1][n],
                                                             Biglist[m][0]} not in edges:  # use of sets for this
                            count1 += 1
                            edges += [{Biglist[m][1][n], Biglist[m][0]}]
                count += count1 / possible_connection
        return int(count / len(self.nodes))

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
        return self

    def make_ring_network(self, N, neighbour_range=1):
        nodes = []
        edges = []
        num_nodes = N
        for node_number in range(num_nodes):
            connections = [0 for _ in range(num_nodes)]
            for distance in range(1, neighbour_range + 1):
                connections[(node_number - distance) % num_nodes] = 1
                connections[(node_number + distance) % num_nodes] = 1
            new_node = Node(np.random.random(), node_number, connections=connections)
            edges.append(connections)
            nodes.append(new_node)

        return edges, nodes

    def make_small_world_network(self, N, re_wire_prob=0.2):
        edges, nodes = self.make_ring_network(N, neighbour_range=2)
        network = Network(nodes)
        network.plot()

        # connection counting script
        # connections_before = 0
        # rewires = 0
        # for node in nodes:
        # print(node.connections)
        # for i, connection in enumerate(node.connections):
        # if connection == 1:
        # connections_before += 1

        for node_no, node in enumerate(nodes):
            connected = [edge_no for edge_no, edge in enumerate(node.connections)
                         if edge_no != node_no and edge == 1]
            for edge in connected:
                not_connected = [edge_no for edge_no, edge in enumerate(node.connections)
                                 if edge_no != node_no and edge == 0]
                if np.random.random() < re_wire_prob:
                    node.connections[edge] = 0
                    node.connections[np.random.choice(not_connected)] = 1
                    # rewires += 1

        # connection counting script
        # connections_after = 0
        # for node in nodes:
        # print(node.connections)
        # for i, connection in enumerate(node.connections):
        # if connection == 1:
        # connections_after += 1

        # print(f"Connections before: {connections_before}")
        # print(f"Number of rewires performed: {rewires}")
        # print(f"Connections after: {connections_after}")

        return nodes

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()


def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")


def independent_test():
    network = Network()
    nodes = network.make_small_world_network(20, 0.2)
    network2 = Network(nodes)
    network2.plot()


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    # get value of self for later
    self_value = population[row, col]

    # find neighbour values
    neighbour__values = find_neighbour_values(population, row, col)

    # calculate agreement by iterating through each neighbour and the summation formula
    agreement = 0
    for neighbour_value in neighbour__values:
        agreement += self_value * neighbour_value

    # add disagreement at the end
    h = external * self_value

    agreement += h

    return agreement


def ising_step(population, alpha, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1
    else:
        # find probability of flipping
        probability = np.exp([-agreement / alpha])
        random_value = np.random.random()
        if random_value < probability:
            population[row, col] *= -1


def plot_ising(im, population):
    """Create a plot of the Ising model

    Args:
        im (_type_): _description_
        population (_type_): _description_
    """
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


def find_neighbour_values(population, row, col):
    n_rows = len(population)
    n_cols = len(population[0])
    neighbours = [population[(row - 1) % n_rows, col],  # up
                  population[row, (col - 1) % n_cols],  # left
                  population[row, (col + 1) % n_cols],  # right
                  population[(row + 1) % n_rows, col]]  # down

    return neighbours


def ising_setup():
    """Sets up the starting grid for the Ising Model. Each cell has a 50/50 chance to be 1 or -1

    Returns:
        population (numpy array): the grid
    """
    population = np.zeros((100, 100))
    for row, cell in enumerate(population):
        for col, value in enumerate(cell):
            random_value = np.random.random()
            if random_value >= 0.5:
                population[row][col] = 1
            else:
                population[row][col] = -1

    return population


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def random_person_and_neighbour(grid_size=100):
    # sets the index of the person being examined
    rand_person = random.randint(0, (grid_size - 1))
    # determines whether the neighbour will the right or the left
    decide_rand_neighbour = random.randint(1, 2)

    # sets the index of the neighbour using circular boundaries

    if decide_rand_neighbour == 1:
        rand_neighbour = (rand_person - 1) % grid_size
    else:
        rand_neighbour = (rand_person + 1) % grid_size
    return rand_person, rand_neighbour


def opinion_defuant(grid, rand_person, rand_neighbour, threshold, coupling_parameter):
    # calculates opinion difference
    opinion_diff = abs(grid[rand_person] - grid[rand_neighbour])

    if opinion_diff < threshold:
        new_op_person = round(grid[rand_person] + coupling_parameter * (grid[rand_neighbour] - grid[rand_person]), 8)
        new_op_neighbour = round(grid[rand_neighbour] + coupling_parameter * (grid[rand_person] - grid[rand_neighbour]),
                                 8)
        grid[rand_person], grid[rand_neighbour] = new_op_person, new_op_neighbour
    return grid


def defuant_main(threshold, coupling_parameter, timesteps=100):
    # Creates grid of 100 people
    grid = np.random.rand(1, 100)[0]

    plt.subplot(1, 2, 2)

    for i in range(timesteps):
        for j in range(len(grid)):
            # grid_size represents the number of villagers:
            rand_person, rand_neighbour = random_person_and_neighbour(grid_size=100)

            # updates grid with new opinions
            grid = opinion_defuant(grid, rand_person, rand_neighbour, threshold, coupling_parameter)
        # creates list of times to use for scatter plot creation
        time_array = np.full((1, 100), (i + 1), dtype=int)[0]
        # graph plotting for the scatter varying with time/interactions
        plt.title(f'Beta:{coupling_parameter}, T:{threshold}, t:{i + 1}')
        plt.scatter(time_array, grid, 15, c='r')
        plt.xlabel('No. of Iterations')
        plt.ylabel('Opinions')

    # graph plotting for the histogram of opinions
    plt.subplot(1, 2, 1)
    plt.hist(grid, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], edgecolor='black')
    plt.ylabel('No. of people')
    plt.xlabel('Opinion rating')
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.grid(axis='y', alpha=0.75)
    plt.title(f'Beta:{coupling_parameter}, T:{threshold}, t:{timesteps}')
    plt.tight_layout()
    plt.show()


def test_defuant():
    # tests the model for a set grid which is changed slightly between some tests

    # The threshold and coupling_parameter (representing 'beta') are changed between tests
    print("Testing defuant model calculations...")
    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.5, 0.5
    assert (opinion_defuant(grid, 3, 4, threshold, coupling_parameter) == [0.5, 0.2, 0.8, 0.25, 0.25, 0.6, 0.7, 0.9, 1,
                                                                           0.3]), "Test 1"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.4
    assert (opinion_defuant(grid, 3, 4, threshold, coupling_parameter) == [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1,
                                                                           0.3]), "Test 2"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.4

    assert (opinion_defuant(grid, 7, 8, threshold, coupling_parameter) == [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.94,
                                                                           0.96, 0.3]), "Test 3"
    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.8
    assert (opinion_defuant(grid, 7, 8, threshold, coupling_parameter) == [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.98,
                                                                           0.92, 0.3]), "Test 4"
    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.7, 0.6
    assert (opinion_defuant(grid, 1, 2, threshold, coupling_parameter) == [0.5, 0.56, 0.44, 0.4, 0.1, 0.6, 0.7, 0.9, 1,
                                                                           0.3]), "Test 5"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.6, 0.6
    assert (opinion_defuant(grid, 2, 1, threshold, coupling_parameter) == [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1,
                                                                           0.3]), "Test 6"

    print("Tests passed")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def arg_setup():
    parser = argparse.ArgumentParser()  # use argparse

    # ising model
    parser.add_argument("-test_ising", action='store_true', default=False,
                        help="-test_ising takes boolean values only; when true, the test code will run")
    parser.add_argument("-ising_model", action='store_true', default=False,
                        help="-ising_model runs the ising model")
    parser.add_argument("-external", default=0, help="-external sets the value of 'h' for the ising model")
    parser.add_argument("-alpha",
                        help="-alpha sets the value of how tolerant a society is of those who disagree with their neighbours for the ising model")

    # networks
    parser.add_argument("-network", type=int,
                        help="size of network")  # network size argument, integer value, by default 10
    parser.add_argument("-test_networks", action='store_true',
                        default=False,
                        help="-test_networks runs tests on the networks model")  # tests if code functions well, if nothing is inputted then will not test

    # defuant model
    parser.add_argument("-test_defuant", action='store_true', default=False,
                        help="-test_defuant takes boolean values only; when true, the test code will run")
    parser.add_argument("-defuant", action='store_true', default=False,
                        help="-defuant runs the defuant model")
    parser.add_argument("-beta", default=0.5,
                        help="-beta sets the value of the coupling parameter for the defuant model")
    parser.add_argument("-threshold", default=0.5,
                        help="-threshold sets the value of the threshold for accepted opinion difference for the defuant model")
    args = parser.parse_args()

    return args


def main():
    # You should write some code for handling flags here
    args = arg_setup()

    if args.test_defuant:  # tests for defuant model if flag detected
        test_defuant()

    if args.defuant:  # runs defuant model if flag detected
        defuant_main(args.threshold, args.beta)

    if args.test_ising:  # tests for ising model if flag detected
        test_ising()

    if args.ising_model:  # runs ising model if flag detected
        ising_main(ising_setup(), args.alpha, args.external)

    if args.test_networks:  # tests for networks modelling stuff if flag detected
        test_networks()

    if args.network:  # runs networks modelling stuff if flag detected
        network = Network().make_random_network(args.network)  # uses size argument
        # prints the required outputs
        print('Mean degree=', network.get_mean_degree())
        print('Mean path length=', network.get_mean_path_length())
        print('Mean cluster coefficient=', network.get_mean_clustering())


if __name__ == "__main__":
    main()
