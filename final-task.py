import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse

# required for the animation of the network version of the defuant model
from matplotlib.animation import FuncAnimation
plt.switch_backend('Agg')


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value

    def __repr__(self):  
        # built in function so that the list with objects 
        # displays a more understandable format
        return ("Node %d has value %d" % (self.index, self.value))
    
        # alternate version that formats the node value differently
        # return(str(format(self.value,".2f")))

    def get_neighbours(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # list comprehension that displays all indexes of the neighbours
        return [i for i, connection in enumerate(self.connections) 
                if connection == 1]


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

    # question 1 for task 3, uses get neighbours function 
    # and computes average for each node
    def get_mean_degree(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        total = 0
        for node in self.nodes:
            total += len(node.get_neighbours())
        
        # return mean 
        return total / len(self.nodes)  

    # question 2 for task 3, uses breadth-first search to find the mean path 
    # from one node to all others
    def get_mean_path_length(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        total = 0

        # for every node
        for node in self.nodes:  
            total1 = 0

            # for every different node than the one looped through
            for elem in self.nodes:
                if elem != node:  
                    
                    start_node = node
                    goal = elem
                    search_queue = Queue()
                    search_queue.push(node)
                    visited = []

                    while not search_queue.is_empty():
                        node_to_check = search_queue.pop(0)

                        # when we end up on the destination
                        if node_to_check == goal: 
                            break

                        for neighbour_index in node_to_check.get_neighbours():
                            neighbour = self.nodes[neighbour_index]
                            if neighbour_index not in visited:
                                search_queue.push(neighbour)
                                visited.append(neighbour_index)
                                neighbour.parent = node_to_check
                        node_to_check = goal

                        # make sure the start node has no parent
                        start_node.parent = None
                        route = []

                    # loop over node parents until the start is reached
                    while node_to_check.parent:
                        # add node to our route
                        route.append(node_to_check)

                        # update node to be the parent of our current node
                        node_to_check = node_to_check.parent

                    # add the start node to the route
                    route.append(node_to_check)

                    total2 = len(route) - 1

                    # sums to compute the average
                    total1 += total2  
            total += total1 / (len(self.nodes) - 1)
        return total / len(self.nodes)

    # question 3 for task 3, clustering coefficient
    def get_mean_clustering(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        count = 0

        # for all nodes
        for node in self.nodes:  
            # get neighbours of node chosen
            neighbour_list = node.get_neighbours()
            # get number of neighbours
            n = len(neighbour_list)
            # calculate number of possible connections with the formula
            possible_connection = n * (n - 1) / 2

            # exclude 0 connections as cannot divide by 0 at the end
            if possible_connection != 0:
                # count of edges between neighbours
                count1 = 0  
                Biglist = []
                # list to check for same edges such as 1-0 and 0-1
                edges = []

                ######################################################
                # get better names for these variables and comments  #
                ######################################################

                # for every neighbour, add neighbours of the neighbours 
                # to the Biglist
                for neighbour in neighbour_list:
                    # neighbours of these nodes
                    Biglist.append((neighbour, 
                                    self.nodes[neighbour].get_neighbours()))

                # for every neighbour of neighbour
                for item in Biglist:
                    for thing in item[1]:
                        if thing in neighbour_list and \
                            {thing, item[0]} not in edges:  
                            # use of sets for this
                            count1 += 1
                            edges += [{thing, item[0]}]
                count += count1 / possible_connection
        return count / len(self.nodes)

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

        for index, node in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

        return self

    def make_ring_network(self, N, neighbour_range=1):
        """_summary_

        Args:
            N (_type_): _description_
            neighbour_range (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        self.nodes = []
        for node_number in range(N):
            connections = [0 for _ in range(N)]
            for distance in range(1, neighbour_range + 1):
                connections[(node_number - distance) % N] = 1
                connections[(node_number + distance) % N] = 1
            new_node = Node(np.random.random(), node_number, connections)
            self.nodes.append(new_node)

        return self

    # small world network for task 4
    def make_small_world_nw(self, N, re_wire_prob=0.2):
        """_summary_

        Args:
            N (_type_): _description_
            re_wire_prob (float, optional): _description_. Defaults to 0.2.

        Returns:
            _type_: _description_
        """
        # start with a ring network with neighbour range 2
        starting_network = self.make_ring_network(N, neighbour_range=2)
        # starting_network.plot()

        # connection counting script
        # connections_before = 0
        # rewires = 0
        # for node in starting_network.nodes:
        # print(node.connections)
        # for i, connection in enumerate(node.connections):
        # if connection == 1:
        # connections_before += 1

        # for every node in the network
        for node_no, node in enumerate(starting_network.nodes):
            # find all connected nodes if they are not itself 
            # and an edge is present
            connected = [edge_no for edge_no, edge in 
                         enumerate(node.connections) if 
                         edge_no != node_no and edge == 1]
            # for every connected edge, roll the chance for a rewire
            for edge in connected:
                # find all non-connected nodes to possibly be connected to
                non_connected = [edge_no for edge_no, edge in 
                                 enumerate(node.connections) if 
                                 edge_no != node_no and edge == 0]
                # roll the chance for a rewire
                if np.random.random() < re_wire_prob:
                    # remove the old connection
                    node.connections[edge] = 0
                    # pick a random unconnected node to connect to
                    node.connections[np.random.choice(non_connected)] = 1
                    # add to the number of rewires
                    # rewires += 1

        # connection counting script
        # connections_after = 0
        # for node in starting_network.nodes:
        # print(node.connections)
        # for i, connection in enumerate(node.connections):
        # if connection == 1:
        # connections_after += 1

        # output counts for before and after. note they should be the same
        # print(f"Connections before: {connections_before}")
        # print(f"Number of rewires performed: {rewires}")
        # print(f"Connections after: {connections_after}")

        return self

    def plot(self, showplot=True):
        """_summary_

        Args:
            showplot (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        args = arg_setup()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        if args.ring_network:
            ax.set_title("Ring Network")
        elif args.small_world:
            ax.set_title(f"Small Worlds Network \
                         (re-wire prob = {args.re_wire})")

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, 
                                color=cm.spring(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), 
                            color='black')
        if showplot:
            plt.show()

        return fig

def test_networks():
    """_summary_
    """
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
    assert (network.get_mean_path_length() == 2.777777777777778), \
        network.get_mean_path_length()

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
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), \
    network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


'''
================================================================================
This section contains code for the Ising Model - task 1 in the assignment
================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if 
    the cell at (row, col) was to flip it's value

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

    # calculate agreement by iterating through each neighbour 
    # and the summation formula
    agreement = 0
    for neighbour_value in neighbour__values:
        agreement += self_value * neighbour_value

    # add disagreement at the end
    h = external * self_value

    return agreement + h


def ising_step(population, alpha, external=0.0):
    '''
    This function will perform a single update of the Ising model

    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external 
            "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1
    else:
        # find probability of flipping
        if alpha != 0:
            probability = np.exp([-agreement / alpha])
            random_value = np.random.random()
            if random_value < probability:
                population[row, col] *= -1


def plot_ising(im, population):
    """Create a plot of the Ising model

    Args:
        im (nparray): current image of plot
        population (nparray): grid to plot
    """
    new_im = np.array([[255 if val == -1 else 1 for val in rows] 
                       for rows in population], dtype=np.int8)
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

    print("Testing external pull")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha, external=0.0):
    """_summary_

    Args:
        population (_type_): _description_
        alpha (_type_): _description_
        external (float, optional): _description_. Defaults to 0.0.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
    ax.set_title(f"External: {format(external,'.3f')}, \
                 Alpha: {format(alpha,'.3f')}")

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for _ in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)

def find_neighbour_values(population, row, col):
    """_summary_

    Args:
        population (_type_): _description_
        row (_type_): _description_
        col (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_rows = len(population)
    n_cols = len(population[0])
    # find all neighbours in the array
    neighbours = [population[(row - 1) % n_rows, col],  # up
                  population[row, (col - 1) % n_cols],  # left
                  population[row, (col + 1) % n_cols],  # right
                  population[(row + 1) % n_rows, col]]  # down

    return neighbours

def ising_setup():
    """Sets up the starting grid for the Ising Model. 
    Each cell has a 50/50 chance to be 1 or -1

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
================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
================================================================================
'''


def random_person_and_neighbour(grid_size=100):
    """_summary_

    Args:
        grid_size (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
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


def opinion_defuant(grid, rand_person, rand_neighbour, threshold, 
                    coupling_parameter):
    """_summary_

    Args:
        grid (_type_): _description_
        rand_person (_type_): _description_
        rand_neighbour (_type_): _description_
        threshold (_type_): _description_
        coupling_parameter (_type_): _description_

    Returns:
        _type_: _description_
    """
    args = arg_setup()

    # calculates opinion difference
    if not args.use_network:
        opinion_diff = abs(grid[rand_person] - grid[rand_neighbour])
    else:
        opinion_diff = abs(grid[rand_person].value - grid[rand_neighbour].value)

    # find new opinions of person and neighbours
    if opinion_diff < threshold and not args.use_network:
        new_op_person = round(grid[rand_person] + coupling_parameter * 
                              (grid[rand_neighbour] - grid[rand_person]), 8)
        
        new_op_neighbour = round(grid[rand_neighbour] + coupling_parameter * 
                                 (grid[rand_person] - grid[rand_neighbour]),8)
        
        # update the grid with the new information
        grid[rand_person], grid[rand_neighbour] = new_op_person, \
        new_op_neighbour
    
    # if the network is being used instead, grid here is the network
    elif opinion_diff < threshold and args.use_network:
        new_op_person = round(grid[rand_person].value + coupling_parameter * 
                              (grid[rand_neighbour].value -
                               grid[rand_person].value), 8)
        
        new_op_neighbour = round(grid[rand_neighbour].value + 
                                 coupling_parameter * (grid[rand_person].value 
                                - grid[rand_neighbour].value), 8)
        
        # update the network
        grid[rand_person].value, grid[rand_neighbour].value = new_op_person, \
        new_op_neighbour
    
    return grid


def defuant_main(threshold, coupling_parameter, timesteps=100):
    """_summary_

    Args:
        threshold (_type_): _description_
        coupling_parameter (_type_): _description_
        timesteps (int, optional): _description_. Defaults to 100.
    """
    # Creates grid of 100 people
    grid = np.random.rand(1, 100)[0]

    plt.subplot(1, 2, 2)

    for i in range(timesteps):
        for _ in grid:
            # grid_size represents the number of villagers:
            rand_person, rand_neighbour = \
            random_person_and_neighbour(grid_size=100)

            # updates grid with new opinions
            grid = opinion_defuant(grid, rand_person, rand_neighbour, 
                                   threshold, coupling_parameter)

        # creates list of times to use for scatter plot creation
        time_array = np.full((1, 100), (i + 1), dtype=int)[0]

        # graph plotting for the scatter varying with time/interactions
        plt.title(f'Beta:{coupling_parameter}, T:{threshold}, t:{i + 1}')
        plt.scatter(time_array, grid, 15, c='r')
        plt.xlabel('No. of Iterations')
        plt.ylabel('Opinions')

    # graph plotting for the histogram of opinions
    plt.subplot(1, 2, 1)
    plt.hist(grid, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
             edgecolor='black')
    plt.ylabel('No. of people')
    plt.xlabel('Opinion rating')
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.grid(axis='y', alpha=0.75)
    plt.title(f'Beta:{coupling_parameter}, T:{threshold}, t:{timesteps}')
    plt.tight_layout()
    plt.show()

def defuant_network(framenetwork, size, threshold, coupling_parameter):
    """_summary_

    Args:
        network (_type_): _description_
        size (_type_): _description_
        threshold (_type_): _description_
        coupling_parameter (_type_): _description_

    Returns:
        _type_: _description_
    """
    network = Network().make_small_world_nw(size)
    #print(network.nodes)

    #fig = plt.figure()

    #for node in network.nodes:
    random_node_selected = np.random.randint(0, size)
    #random_node = network.nodes[random_node_selected]

    # determines whether the neighbour will the right or the left
    decide_rand_neighbour = random.randint(1, 2)
    # sets the index of the neighbour using circular boundaries
    if decide_rand_neighbour == 1:
        rand_neighbour_selected = (random_node_selected - 1) % size
    else:
        rand_neighbour_selected = (random_node_selected + 1) % size

    #rand_neighbour = network.nodes[rand_neighbour_selected]
        
    # updates network with new opinions
    network.nodes = opinion_defuant(network.nodes, random_node_selected, 
                                    rand_neighbour_selected, threshold, 
                                    coupling_parameter)

    #print(network.nodes)
    return network

def update_animation( frame, network, size, threshold, coupling_parameter):
    """_summary_

    Args:
        frame (_type_): _description_
        network (_type_): _description_
        size (_type_): _description_
        threshold (_type_): _description_
        coupling_parameter (_type_): _description_
    """
    #print('ARGS=' network, size, threshold, coupling_parameter)
    network2 = defuant_network(network, size, threshold, coupling_parameter)

    plt.clf()
    
    return  network2.plot() 

def animate_defuant_network(size, threshold, coupling_parameter):
    """_summary_

    Args:
        size (_type_): _description_
        threshold (_type_): _description_
        coupling_parameter (_type_): _description_
    """
    network = Network().make_small_world_nw(size)
    fig = plt.figure()
    #plt.show()
    animated_thing = FuncAnimation( fig,update_animation, 
                                   fargs=(network, size, threshold,
                                          coupling_parameter), frames=120, 
                                          interval=1000/2)
    #plt.show()
    animated_thing.save('animation.mp4', writer="ffmpeg", fps=2)
    #plt.show()

def test_defuant():
    """_summary_
    """
    # tests the model for a set grid which is changed slightly between tests

    # the threshold and coupling_parameter (representing 'beta') are changed
    print("Testing defuant model calculations...")

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.5, 0.5
    assert (opinion_defuant(grid, 3, 4, threshold, coupling_parameter) == 
            [0.5, 0.2, 0.8, 0.25, 0.25, 0.6, 0.7, 0.9, 1, 0.3]), "Test 1"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.4
    assert (opinion_defuant(grid, 3, 4, threshold, coupling_parameter) ==
            [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]), "Test 2"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.4
    assert (opinion_defuant(grid, 7, 8, threshold, coupling_parameter) == 
            [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.94, 0.96, 0.3]), "Test 3"
    
    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.2, 0.8
    assert (opinion_defuant(grid, 7, 8, threshold, coupling_parameter) == 
            [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.98, 0.92, 0.3]), "Test 4"
    
    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.7, 0.6
    assert (opinion_defuant(grid, 1, 2, threshold, coupling_parameter) == 
            [0.5, 0.56, 0.44, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]), "Test 5"

    grid = [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]
    threshold, coupling_parameter = 0.6, 0.6
    assert (opinion_defuant(grid, 2, 1, threshold, coupling_parameter) == 
            [0.5, 0.2, 0.8, 0.4, 0.1, 0.6, 0.7, 0.9, 1, 0.3]), "Test 6"

    print("Tests passed")


'''
================================================================================
This section contains code for the main function - 
you should write some code for handling flags here
================================================================================
'''


def arg_setup():
    """_summary_

    Returns:
        _type_: _description_
    """
    # use argparse
    parser = argparse.ArgumentParser()  

    # ising model
    parser.add_argument("-test_ising", action='store_true', default=False,
                        help="-test_ising takes boolean values only; \
                        when true, the test code will run")
    parser.add_argument("-ising_model", action='store_true', default=False,
                        help="-ising_model runs the ising model")
    parser.add_argument("-external", default=0.0, help="-external sets the \
                        value of 'h' for the ising model", type=float)
    parser.add_argument("-alpha", type=float, default=0.01,
                        help="-alpha sets the value of how tolerant a society \
                            is of those who disagree with their neighbours for \
                            the ising model")

    # networks
    # network size argument, integer value
    parser.add_argument("-network", type=int, help="size of network")

    # tests if code functions well, if nothing is inputted then will not test
    parser.add_argument("-test_network", action='store_true', default=False, 
                        help="-test_network runs tests on the networks model")
    
    # defuant model
    parser.add_argument("-test_defuant", action='store_true', default=False,
                        help="-test_defuant takes boolean values only; \
                            when true, the test code will run")
    parser.add_argument("-defuant", action='store_true', default=False,
                        help="-defuant runs the defuant model")
    parser.add_argument("-beta", type=float, default=0.5,
                        help="-beta sets the value of the coupling parameter \
                              for the defuant model")
    parser.add_argument("-threshold", type=float, default=0.5,
                        help="-threshold sets the value of the threshold for \
                          accepted opinion difference for the defuant model")

    # ring_network stuff
    # network size argument, integer value, by default 10
    parser.add_argument("-ring_network", type=int,
                        help="-ring_network determines no. of nodes in network") 
    parser.add_argument("-small_world", type=int,
                        help="-small_world determines no. of nodes in network")
    parser.add_argument("-re_wire", default=0, type=float,
                        help="-re_wire determines probability. \
                            Should be between 0 and 1")

    # additional network code for defuant model
    parser.add_argument("-use_network", type=int,
                        help="-use_network converts the defuant model \
                            to a network and calculates accordingly")

    args = parser.parse_args()
    assert 0 <= args.re_wire <= 1, 're_wire is a probability, \
        thus must be between 0 and 1'

    return args


def main():
    """_summary_
    """
    # code to handle flags
    args = arg_setup()

    # tests for defuant model if flag detected
    if args.test_defuant:  
        test_defuant()

    # runs defuant model if flag detected
    if args.defuant:  
        if not args.use_network:
            defuant_main(args.threshold, args.beta)
        else:
            #defuant_network(args.use_network, args.threshold, args.beta)
            animate_defuant_network(args.use_network, args.threshold, args.beta)

    # tests for ising model if flag detected
    if args.test_ising:  
        test_ising()

    # runs ising model if flag detected
    if args.ising_model:  
        ising_main(ising_setup(), args.alpha, args.external)

    # tests for networks modelling stuff if flag detected
    if args.test_network: 
        test_networks()

    # runs networks modelling stuff if flag detected
    if args.network: 
        # uses size argument from flag
        network = Network().make_random_network(args.network)

        # prints the required outputs
        print('Mean degree:', network.get_mean_degree())
        print('Average path length:', network.get_mean_path_length())
        print('Clustering co-efficient:', network.get_mean_clustering())

    # runs ring networks modelling stuff if flag detected
    if args.ring_network:  
        ring_network = Network().make_ring_network(20, 3)
        ring_network.plot()

    # runs small world code if flag detected
    if args.small_world:  
        small_world_network = Network().make_small_world_nw(20, args.re_wire)
        small_world_network.plot()


if __name__ == "__main__":
    main()
