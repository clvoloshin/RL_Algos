import scipy.sparse as sparse
import numpy as np
from food import Food
import time
import pdb

def get_implied_board(snake, screen_width, screen_height, fill = 1):
        '''
        Given a snake, generate the sparse matrix which represents the board and has the snake filled in with its grayscale color
        '''        
        pixels = np.array(snake.body)
        try:
            rows = pixels[:,0]
            cols = pixels[:,1]
            fill = fill + 10**-12 if fill == .5 else fill
            sparse_matrix = sparse.coo_matrix(([fill]*snake.length, (rows,cols)), 
                                 shape = (screen_width, screen_height)) #.astype(np.uint8)
        except ValueError as ve:
            snake.alive = False # This occurs if snake exits the boundary
            return sparse.csr_matrix((screen_width, screen_height))

        return sparse_matrix.tocsr()

def get_food_board(all_food, screen_width, screen_height, fill = 1):
    '''
    Given a snake, generate the sparse matrix which represents the board and has the snake filled in with its grayscale color
    '''        
    pixels = np.array(all_food)
    try:
        rows = pixels[:,0]
        cols = pixels[:,1]
        fill = fill + 10**-12 if fill == .5 else fill
        sparse_matrix = sparse.coo_matrix(([fill]*rows.shape[0], (rows,cols)), 
                             shape = (screen_width, screen_height)) #.astype(np.uint8)
    except: # occurs if no food
        return sparse.csr_matrix((screen_width, screen_height))

    return sparse_matrix.tocsr()

def check_if_snake_hit_boundary(snake, num_rows, num_cols):
    snake.alive = snake.alive and (snake.body[0][0] > 0) and (snake.body[0][0] < (num_rows-1)) and (snake.body[0][1] > 0) and (snake.body[0][1] < (num_cols-1))

def check_if_snake_self_intersected(snake):
    snake.alive = snake.alive and not (snake.body[0] in snake.body[1:])

def check_if_snake_hit_other_snake(snake, other_snake):
    snake.alive = False if (snake.body[0] in other_snake.body) else snake.alive

def check_which_snakes_are_still_alive(snakes, num_rows, num_cols):
    for i,snake in enumerate(snakes):
        if snake.alive:
            for j, other_snake in enumerate(snakes):
                if other_snake.alive:
                    if i != j:
                        check_if_snake_hit_other_snake(snake, other_snake)
                    else:
                        check_if_snake_self_intersected(snake)
                        check_if_snake_hit_boundary(snake, num_rows, num_cols)

def get_boards(snakes, num_rows, num_cols):
    boards = []
    for snake in snakes:
        boards.append(get_implied_board(snake, num_rows, num_cols))

    return boards


def get_state(snakes, all_food, num_rows, num_cols, growth):
    snakes = np.array(snakes)

    # gets previously alive snakes' boards
    
    boards = np.array(get_boards(snakes, num_rows, num_cols)) # kills snakes that exit board

    # check if snake is valid
    
    check_which_snakes_are_still_alive(snakes, num_rows, num_cols) # kills snakes which hit other snakes or themselves

    
    idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive]
    boards[np.array([not x.alive for x in snakes])] = sparse.csr_matrix((num_rows, num_cols)) # set dead snakes as dead

    
    all_locations = []
    for food in all_food:
        all_locations.append([food.location[0][0],food.location[0][1]])
    food_board = get_food_board(all_locations,num_rows,num_cols) 

    
    total_board = sum(boards) + food_board

    # Add more food if necessary
    food_to_add = len(idxs_of_alive_snakes) - len(all_food) # MAke only as much food as alive snakes
    
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


    return boards.tolist() + heads + [food_board], idxs_of_alive_snakes
    # else:
    #     return [], []

# def get_boards_old(snakes, num_rows, num_cols):
#     '''
#     Deprecated
#     '''
#     boards = []
#     for snake in snakes:
#         if snake.alive:
#             board = get_implied_board(snake, num_rows, num_cols)
#             if board.nnz > 0:
#                 boards.append(board)
#     return boards

# def get_state_old(snakes, all_food, num_rows, num_cols, min_amount_of_food, growth, use_grayscale):
#     '''
#     DEPRECATED
#     '''
#     snakes = np.array(snakes)

#     # gets previously alive snakes' boards
#     boards = get_boards(snakes, num_rows, num_cols)    

#     # checks if snakes are still currently alive
#     idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive]
#     check_which_snakes_are_still_alive(snakes[idxs_of_alive_snakes], boards)
    

#     idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive]

#     if len(idxs_of_alive_snakes) > 0:
#         boards = []
#         for idx in idxs_of_alive_snakes:
#             boards.append(get_implied_board(snakes[idx], num_rows, num_cols, fill = snakes[idx].to_grayscale() if use_grayscale else 1)) 

#         total_board = sum(boards)

#         for food in all_food:
#             total_board[food.location[0][0],food.location[0][1]] = .5 if use_grayscale else 1

#         # Add more food if necessary
#         food_to_add = min_amount_of_food - len(all_food)
#         while food_to_add > 0:
#             start_x_loc = np.random.randint(0, num_rows, size = 1)
#             start_y_loc = np.random.randint(0, num_cols, size = 1)
#             if total_board[start_x_loc, start_y_loc] == 0:
#                 total_board[start_x_loc, start_y_loc] = .5 if use_grayscale else 1
#                 all_food += [Food(start_x = start_x_loc,
#                                    start_y = start_y_loc,
#                                    growth = growth)]
#                 food_to_add -= 1

#         return total_board, idxs_of_alive_snakes
#     else:
#         return 0,[] # gameover
