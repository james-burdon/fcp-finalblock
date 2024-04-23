import numpy as np
import matplotlib.pyplot as plt

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

    #get value of self for later
    self_value = population[row, col]

    #find neighbour values
    neighbour__values = find_neighbour_values(population, row, col)

    #calculate agreement by iterating through each neighbour and the summation formula
    agreement = 0
    for neighbour_value in neighbour__values:
        agreement += self_value * neighbour_value

    #add disagreement at the end
    h = external * self_value

    ################################################################################################
    # if !args.h:
    #     h = 0
    # uncomment when argsetup is done as a function
    ################################################################################################

    agreement += h

    return agreement

def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col  = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1
    else:
        #find probability of flipping
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
    """
    This function will test the calculate_agreement function in the Ising model
    """

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,-10)==14), "Test 9"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 10"

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
    neighbours = [population[(row-1)%n_rows, col], #up
                  population[row, (col-1)%n_cols], #left
                  population[row, (col+1)%n_cols], #right
                  population[(row+1)%n_rows, col]] #down
    
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

if __name__=="__main__":
    test_ising()

    population = ising_setup()

    alpha = 10
    external = 0.0

    ising_main(population, alpha, external)