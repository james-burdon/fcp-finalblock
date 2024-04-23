import numpy as np
import argparse
class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

    def __repr__(self): #built in funcion so that list with objects displays a more understandable format
        return ("Node %d has value %d" % (self.index, self.value))
    
    def get_neighbours(self):
        return [i for i in range(len(self.connections)) if self.connections[i]==1] #comprehension list that displays all indexes of the neighbours of node chosen

class Queue: #queue class imported from previous assignments
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

    def get_mean_degree(self): #question 1 for task 3, uses get neighbours function and computes average for each node
        total=0
        for i in range(len(self.nodes)):
            total+=len(self.nodes[i].get_neighbours())
        return total/len(self.nodes) #mean part


    def get_mean_path_length(self): #question 2 for task 3, uses breadth-first search to find the mean path from one node to all others
        total=0
        for i in range(len(self.nodes)): #for every node
            total1=0
            for elem in self.nodes:
                if elem!=self.nodes[i]: #for every different node than the one looped through
                    start_node = self.nodes[i]
                    goal = elem
                    search_queue = Queue()
                    search_queue.push(self.nodes[i])
                    visited = []

                    while not search_queue.is_empty():
                        node_to_check = search_queue.pop(0)
                
                        if node_to_check == goal: #when we end up on the destination
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
                    # print(route)
                    total2=len(route)-1
                        #Reverse and print the route
                    total1+=total2 # sums to compute the average
            total1/=(len(self.nodes)-1)
            total+=total1
        total/=len(self.nodes)
        return total    

            


    def get_clustering(self): #question 3 for task 3, clustering coefficient
        count=0
        print(self.nodes)
        for i in range(len(self.nodes)): #for all nodes
            minilist=self.nodes[i].get_neighbours() #neighbours of node chosen
            n=len(minilist) #number of neighbours
            possible_connection=n*(n-1)/2 #formula
            #print('possible connection=',possible_connection)
            if possible_connection!=0: #cannot divide by 0 at the end
                count1=0#count of edges between neighbours
                Biglist=[]
                edges=[] #list to check for same edges such as 1-0 and 0-1
                for j in range(len(minilist)):
                    Biglist.append((minilist[j],self.nodes[minilist[j]].get_neighbours())) #neighbours of these nodes
                for m in range(len(Biglist)):
                    for n in range(len(Biglist[m][1])):
                        if Biglist[m][1][n] in minilist and  {Biglist[m][1][n],Biglist[m][0]} not in edges: #use of sets for this
                            count1+=1
                            edges+=[{Biglist[m][1][n],Biglist[m][0]}]
                count+=count1/possible_connection
        return int(count/len(self.nodes))

            


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
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
        return self

def main(): #main function
    parser = argparse.ArgumentParser() #use argparse
    parser.add_argument("-network", type=int, default=10, help="size of network") #network size argument, integer value, by default 10 
    parser.add_argument("-test_networks",action='store_true', default=False) #tests if code functions well, if nothing is inputted then will not test
    args = parser.parse_args()
    network=Network()
    network=network.make_random_network(args.network) #uses size argument
    print('Mean degree=',network.get_mean_degree())
    print('Mean path length=',network.get_mean_path_length())
    print('Mean cluster coefficient=', network.get_clustering())
    if args.test_networks==True:
        test_networks()

    

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
    print('Tests passed')

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
    assert(network.get_mean_path_length()==5), network.get_mean_path_length()

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
    assert(network.get_mean_path_length()==1), network.get_mean_path_length()

    print("All tests passed")


if __name__ == '__main__':
    main()