import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def make_ring_network(self, N, neighbour_range=1):
        nodes = []
        num_nodes = N
        for node_number in range(num_nodes):
            connections = [0 for _ in range(num_nodes)]
            for distance in range(1, neighbour_range + 1):
                connections[(node_number-distance)%num_nodes] = 1
                connections[(node_number+distance)%num_nodes] = 1
            new_node = Node(np.random.random(), node_number, connections=connections)
            nodes.append(new_node)

        return nodes
        

    def make_small_world_network(self, N, re_wire_prob=0.2):
        starting_network = Network
        nodes = starting_network.make_ring_network(self, N, 2)

        connections_before = 0
        rewires = 0

        for node_no, node in enumerate(nodes):
            print(node.connections)
            for old_node, connection in enumerate(node.connections):
                if connection == 1:
                    connections_before += 1
                    new_connections = [0 for _ in range(N)]
                    random_value = np.random.random()
                    if re_wire_prob > random_value:
                        rewires += 1
                        random_connection = np.random.randint(0, N)
                        new_connections[random_connection] = 1
                        node.connections = new_connections

                        #if nodes[random_node].connections

        connections_after = 0

        print()

        for node in nodes:
            print(node.connections)
            for i, connection in enumerate(node.connections):
                if connection == 1:
                    connections_after += 1

        print(f"before: {connections_before}")
        print(f"rewires: {rewires}")
        print(f"after: {connections_after}")

        return nodes

        """
        #this is purely a random network
        for node in nodes:
            new_connections = [0 for _ in range(N)]
            for i, connection in enumerate(new_connections):
                random_value = np.random.random()
                if random_value < re_wire_prob:
                    new_connections[i] = 1
            node.connections = new_connections
            print(node.connections)
        
        return nodes
        """

        """
        #this doesnt work as intended
        for node_number, node in enumerate(nodes):
            for connection_number, connection in enumerate(node.connections):
                random_value = np.random.random()
                if re_wire_prob > random_value:
                    print("rewiring")
                    new_connection = (connection + 1) % 2
                    connections[connection_number] = new_connection
                    nodes[node_number] = Node(node.value, node.index, connections)
                else:
                    continue

        print(f"Connections: {connections}")
        
        return connections, nodes
        """

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
		
        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
		
        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)
		
            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

def independent_test():
    network = Network()
    nodes = network.make_small_world_network(20, 0.2)
    network2 = Network(nodes)
    network2.plot()

if __name__=="__main__":
    independent_test()
