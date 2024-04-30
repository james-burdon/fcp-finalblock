# FCP Summative Assessment

## Group 3

James Burdon, Archie Burton and Eric Chen

## Running the program

The main file, `Complete_File.py`, is to be run from the terminal with flags to determine what the program should do. **If no flags are entered, the program does nothing.**

The flags that can be used include:

### Ising Model

- `-test_ising` to test the ising calculation functions
- `-ising_model` to run the ising model with default parameters (external = 0.0, alpha = 0.01)
- `-external` to change the value of the external pull of opinions
- `-alpha` to change the value of alpha in the model

### Defuant Model

- `-test_defuant` to test the opinion calculation functions
- `-defuant` to run the defuant model with default parameters (beta = 0.5, threshold = 0.5)
- `-beta` to change the value of beta
- `-threshold` to change the opinion difference threshold
- `-use_network` to use a network in the defuant model

### Networks

- `-network` to set the size of a network
- `-test_network` to test the calculations for mean degree, path length and clustering calculations
- `-ring_network` to set the number of nodes in a ring network
- `-small_world` to set the size of the small world network
- `-re_wire` to set the rewiring probability for the small world network (must be 0 <= re_wire <= 1)

### Example flag usage

- `-defuant -use_network 100 -beta 0.9`
  - runs a network with 100 nodes in a defuant model with a beta value (coupling parameter) of 0.9
- `-test_ising -ising_model -alpha 0.001`
  - runs the ising model calculation test, and the ising model with an alpha value of 0.001
- `-defuant -use_network 100 -ising_model -external 0.3 -beta 0.9`
  - runs the defuant model on a network of 100 nodes with a beta of 0.9, then the ising model with an external pull of 0.3

## Changes made to provided code

These edits were made mostly to help aid repeatability of code and to ease debugging.

### Node class

- `__repr__()` function added to report a more understandable output when looking at node objects via `print(network.nodes)` or similar. Commented out is an alternate output format with less rounding, used extensively whilst debugging
- `get_neighbours()` function added to list all the neighbours (connected nodes) of the specific node

### Network class

- `plot()` function now includes
  - a parameter `for_animation=None` to stop new figures being made every function call. Default value is None to ensure a figure is made for the other tasks
  - `args = arg_setup()` added to give the plotted graph the correct title

### Other functions

- `calculate_agreement()` now returns `agreement + h`, rather than `np.random.random()`
- `ising_step()` takes an extra argument alpha for more customisabe ising model outputs
- `ising_main()` adds a title to the graph plotted with `ax.set_title()`
- `main()` runs through all the possible arguments, and runs the corresponding functions

## Additions

Check comments and docstrings in the file `Complete_File.py` for more detailed explanations for each function.

- `Queue` class for the breadth-first search in task 3
- `find_neighbour_values()` finds the four neighbours of a cell on the grid for the ising model
- `ising_setup()` sets up the ising model initial grid, with a 50% chance of the cell being 1 or -1
- `random_person_and_neighbour()` finds a person and a random neighbour (left or right) and returns them for the defuant model calculations
- `opinion_defuant()` finds the opinion difference of a cell and its (previously chosen) neighbour. Checks the arguments to see if a network is used to model instead, and uses the node values instead of the cell values if applicable
- `update_network()` updates the network, calculating new opinions for a single random node in the network
- `defuant_network()` sets up the network for the defuant model, as well as tracking the mean opinion of the network. Animates the network changing over the time interval and shows plot for mean opinion over time
- `arg_setup()` sets up the arguments from the command line, whilst checking that certain flags entered are valid

## Other notes

- `make_small_world` includes a commented out edge-counting script. This script counts the number of edges the network starts with, the number of rewires that occur, and the number of edges after the rewiring. The number of edges are the same before and after the rewiring. **All lines in this function need to be uncommented for it to work**
- Individual work on tasks can be found in the named branches. Work was done on each individual branch before the merge onto main
