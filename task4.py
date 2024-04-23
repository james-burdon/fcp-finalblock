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
        edges = []
        num_nodes = N
        for node_number in range(num_nodes):
            connections = [0 for _ in range(num_nodes)]
            for distance in range(1, neighbour_range + 1):
                connections[(node_number-distance)%num_nodes] = 1
                connections[(node_number+distance)%num_nodes] = 1
            new_node = Node(np.random.random(), node_number, connections=connections)
            edges.append(connections)
            nodes.append(new_node)

        return edges, nodes
    
    def make_small_world_network(self, N, re_wire_prob=0.2):
        edges, nodes = self.make_ring_network(N, neighbour_range=2)
        network = Network(nodes)
        network.plot()

        #connection counting script
        #connections_before = 0
        #rewires = 0
        #for node in nodes:
            #print(node.connections)
            #for i, connection in enumerate(node.connections):
                #if connection == 1:
                    #connections_before += 1

        for node_no, node in enumerate(nodes):
            connected = [edge_no for edge_no, edge in enumerate(node.connections)
                         if edge_no != node_no and edge == 1]
            for edge in connected:
                not_connected = [edge_no for edge_no, edge in enumerate(node.connections) 
                                 if edge_no != node_no and edge == 0]
                if np.random.random() < re_wire_prob:
                    node.connections[edge] = 0
                    node.connections[np.random.choice(not_connected)] = 1
                    #rewires += 1

        #connection counting script
        #connections_after = 0
        #for node in nodes:
            #print(node.connections)
            #for i, connection in enumerate(node.connections):
                #if connection == 1:
                    #connections_after += 1

        #print(f"Connections before: {connections_before}")
        #print(f"Number of rewires performed: {rewires}")
        #print(f"Connections after: {connections_after}")

        return nodes

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
