
class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

    def __repr__(self):
        return ("Node %d has value %d" % (self.index, self.value))
    
    def get_neighbours(self):
        return [i for i in range(len(self.connections)) if self.connections[i]==1]

class Queue:
    def __init__(self):
        self.queue = []
    def push(self, item):
        self.queue.append(item)
    def pop(self,position):
        if len(self.queue)<1:
            return None
        return self.queue.pop(position)
    
    def is_empty(self):
        return len(self.queue)==0

class Network: 
    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

    def get_mean_degree(self):
        total=0
        for i in range(len(self.nodes)):
            total+=len(self.nodes[i].get_neighbours())
        return total/len(self.nodes)


    def get_mean_path_length(self):
        total=0
        for i in range(len(self.nodes)):
            total1=0
            for elem in self.nodes:
                if elem!=self.nodes[i]:
                    start_node = self.nodes[i]
                    goal = elem
                    search_queue = Queue()
                    search_queue.push(self.nodes[i])
                    visited = []

                    while not search_queue.is_empty():
                        node_to_check = search_queue.pop(0)
                
                        if node_to_check == goal:
                            break

                        for neighbour_index in node_to_check.get_neighbours():
                            neighbour = self.nodes[neighbour_index]
                            if neighbour_index not in visited:
                                search_queue.push(neighbour)
                                visited.append(neighbour_index)
                                neighbour.parent = node_to_check
                        node_to_check = goal
                        
                        #We make sure the start node has no parent.
                        
                        start_node.parent = None
                        route = []
                        
                        #Loop over node parents until we reach the start.
                    while node_to_check.parent:
                    
                        #Add node to our route
                        route.append(node_to_check)
                    
                        #Update node to be the parent of our current node
                        node_to_check = node_to_check.parent
                    
                        #Add the start node to the route
                    route.append(node_to_check)
                    print(route)
                    total2=len(route)-1
                        #Reverse and print the route
                    total1+=total2
            total1/=(len(self.nodes)-1)
            total+=total1
        total/=len(self.nodes)
        return total    

            


    def get_clustering(self):
        count=0
        for i in range(len(self.nodes)):
            minilist=self.nodes[i].get_neighbours()
            n=len(minilist) #number of neighbours
            possible_connection=n*(n-1)/2
            Biglist=[]
            for j in range(len(minilist)):
                Biglist.append(self.nodes[minilist[j]].get_neighbours())
            result = set(Biglist[0])
            for s in Biglist[1:]:
                result.intersection_update(s)
            result.remove(i)
            print(Biglist,result,'result')
            count+=len(result)
        return count/len(self.nodes)

            


    def make_random_network(self, N, connection_probability):
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
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

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
    assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

if __name__ == '__main__':
    test_networks()