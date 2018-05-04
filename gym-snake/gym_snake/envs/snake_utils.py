import scipy.sparse as sparse
import numpy as np
from food import Food
import time
import pdb
import itertools

def get_implied_board(snake, screen_width, screen_height, fill = 1):
        '''
        Given a snake, generate the sparse matrix which represents the board and has the snake filled in with its fill color

        Param
            snake: object
                Instance of Snake class

            screen_width: int
                width of gridworld

            screen_height: int
                height of gridworld

            fill: float
                This number is placed at every location of the snake's body in the gridworld

        Return:
            csr matrix (sparse)
                This matrix represents the gridworld

        '''        
        pixels = np.array(snake.body)
        try:
            rows = pixels[:,0].reshape(-1)
            cols = pixels[:,1].reshape(-1)
            fill = fill + 10**-12 if fill == .5 else fill
            sparse_matrix = sparse.coo_matrix(([fill]*snake.length, (rows,cols)), 
                                 shape = (screen_width, screen_height)) #.astype(np.uint8)
        except ValueError as ve:
            snake.alive = False # This occurs if snake exits the boundary
            return sparse.csr_matrix((screen_width, screen_height))

        return sparse_matrix.tocsr()

def get_food_board(all_food, screen_width, screen_height, fill = 1):
    '''
        Given a snake, generate the sparse matrix which represents the board and has the snake filled in with its fill color

        Param
            all_food: list of list
                each list in the list is an (x,y) coord representing a food location

            screen_width: int
                width of gridworld

            screen_height: int
                height of gridworld

            fill: float
                This number is placed at every location of the snake's body in the gridworld

        Return:
            csr matrix (sparse)
                This matrix represents the gridworld

        '''        
    pixels = np.array(all_food)
    try:
        rows = pixels[:,0].reshape(-1)
        cols = pixels[:,1].reshape(-1)
        fill = fill + 10**-12 if fill == .5 else fill
        sparse_matrix = sparse.coo_matrix(([fill]*rows.shape[0], (rows,cols)), 
                             shape = (screen_width, screen_height)) #.astype(np.uint8)
    except: # occurs if no food
        return sparse.csr_matrix((screen_width, screen_height))

    return sparse_matrix.tocsr()

def check_if_snake_hit_boundary(snake, num_rows, num_cols):
    '''
    Kills snake if it hit the edge of the gridworld

        Param
            num_rows: int
                width of gridworld

            num_cols: int
                height of gridworld
    '''
    snake.alive = snake.alive and (snake.body[0][0] > 0) and (snake.body[0][0] < (num_rows-1)) and (snake.body[0][1] > 0) and (snake.body[0][1] < (num_cols-1))

def check_if_snake_self_intersected(snake):
    '''
    Kills snake if it hit itself

        Param
            snake: object
                Instance of Snake class
    '''
    snake.alive = snake.alive and not (snake.body[0] in snake.body[1:])

def check_if_snake_hit_other_snake(snake, other_snake):
    '''
    Kills snake if it hit a different snake

        Param
            snake: object
                Instance of Snake class

            other_snake: object
                Instance of Snake class
    '''
    snake.alive = False if (snake.body[0] in other_snake.body) else snake.alive

def check_which_snakes_are_still_alive(snakes, num_rows, num_cols):
    '''
    Kills all snakes which have reached a terminal condition

        Param
            snake: list
                list contains snake objects

            num_rows: int
                width of gridworld

            num_cols: int
                height of gridworld
    '''

    # Need this array to eliminate the problem of which order objects occur in the param snakes
    snake_alive = [snake.alive for idx,snake in enumerate(snakes)]

    for i,snake in enumerate(snakes):
        if snake_alive[i]:
            for j, other_snake in enumerate(snakes):
                if snake_alive[j]:
                    if i != j:
                        check_if_snake_hit_other_snake(snake, other_snake)
                    else:
                        check_if_snake_self_intersected(snake)
                        check_if_snake_hit_boundary(snake, num_rows, num_cols)

def get_boards(snakes, num_rows, num_cols):
    '''
    Get sparse grid for each snake

        Param
            snakes: list
                list contains snake objects

            num_rows: int
                width of gridworld

            num_cols: int
                height of gridworld

        Return
            boards: list
                list of sparse grids
    '''
    boards = []
    for snake in snakes:
        boards.append(get_implied_board(snake, num_rows, num_cols))

    return boards


def get_state(snakes, all_food, num_rows, num_cols, growth):
    '''
    Runs through logic

        Param
            snakes: list
                list contains snake objects

            all_food: list
                list contains food objects

            num_rows: int
                width of gridworld

            num_cols: int
                height of gridworld

            growth: int
                set new growth of food if new food is required to be made

        Return
            state: list
                A concatenation of the following lists: (1) sparse grids of full snake
                                                        (2) sparse grid of snake heads
                                                        (3) sparse grid of food

            idxs_of_alive_snakes: list
                list of ints representing which snakes are alive

    '''


    snakes = np.array(snakes)

    # get snake boards
    boards = np.array(get_boards(snakes, num_rows, num_cols))

    # check which snakes are valid
    check_which_snakes_are_still_alive(snakes, num_rows, num_cols) # kills snakes which hit other snakes, themselves, or boundary

    idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive] # enumerate alive snakes
    boards[np.array([not x.alive for x in snakes])] = sparse.csr_matrix((num_rows, num_cols)) # set dead snakes' boards as all zero

    
    # Get food sparse matrix
    # This is the quickest way to do it bc splicing sparse matrix is slow but coo to csr is fast
    all_locations = []
    for food in all_food:
        all_locations.append([food.location[0][0],food.location[0][1]])
    food_board = get_food_board(all_locations,num_rows,num_cols) 

    # Add more food if necessary
    total_board = sum(boards) + food_board

    food_to_add = len(idxs_of_alive_snakes) - len(all_food) # Make only as much food as alive snakes
    
    # Handle infinite loop that happens at the end of game
    while (total_board.nnz <= ((total_board.shape[0]-2)*(total_board.shape[0]-2))-1) and (food_to_add > 0):
        start_x_loc = np.random.randint(1, num_rows-1, size = 1)
        start_y_loc = np.random.randint(1, num_cols-1, size = 1)
        if total_board[start_x_loc, start_y_loc] == 0:
            food_board[start_x_loc, start_y_loc] = 1
            total_board[start_x_loc, start_y_loc] = 1
            all_food += [Food(start_x = start_x_loc,
                               start_y = start_y_loc,
                               growth = growth)]
            food_to_add -= 1        

    # get heads
    heads = []
    for snake in snakes:
        if snake.alive:
            heads.append(sparse.coo_matrix(([1], ([snake.body[0][0]],[snake.body[0][1]])), 
                                 shape = (num_rows, num_cols)).tocsr())
        else:
            heads.append(sparse.csr_matrix((num_rows, num_cols)))

    for i, snake in enumerate(snakes):
        if not snake.alive:
            assert boards[i].nnz == 0
            assert heads[i].nnz == 0
        else:
            assert boards[i].nnz == snake.length
            assert heads[i].nnz == 1


    # Make into (snake0 body, snake0 head,...,snakeN body, snakeN head)
    snake_boards = list(itertools.chain(*zip(boards.tolist(), heads)))

    state = snake_boards + [food_board]
    return state, idxs_of_alive_snakes

