import scipy.sparse as sparse
import numpy as np
from food import Food

def get_implied_board(snake, screen_width, screen_height, fill = 1):
        '''
        Given a snake, generate the sparse matrix which represents the board and has the snake filled in with its grayscale color
        '''        
        pixels = np.array(snake.body)
        rows = pixels[:,0]
        cols = pixels[:,1]
        fill = fill + 10**-12 if fill == .5 else fill
        try:
            sparse_matrix = sparse.coo_matrix(([fill]*snake.length, (rows,cols)), 
                                 shape = (screen_width, screen_height)) #.astype(np.uint8)
        except ValueError as ve:
            snake.alive = False # This occurs if snake exits the boundary
            return sparse.csr_matrix((screen_width, screen_height))

        return sparse_matrix.tocsr()

def check_if_snake_self_intersected(snake, board):
    snake.alive = snake.alive and not (sparse.find(board)[-1] > 1).any()

def check_if_snake_hit_other_snake(snake, board):
    snake.alive = False if (board[snake.body[0][0], snake.body[0][1]] > 0) else snake.alive

def check_which_snakes_are_still_alive(snakes, boards):
    for i,snake in enumerate(snakes):
        if snake.alive:
            for j, board in enumerate(boards):
                if i != j:
                    check_if_snake_hit_other_snake(snake, board)
                else:
                    check_if_snake_self_intersected(snake, board)



def get_boards(snakes, num_rows, num_cols):
    boards = []
    for snake in snakes:
        boards.append(get_implied_board(snake, num_rows, num_cols))

    return boards


def get_state(snakes, all_food, num_rows, num_cols, min_amount_of_food, growth):
    snakes = np.array(snakes)

    # gets previously alive snakes' boards
    boards = get_boards(snakes, num_rows, num_cols)    

    # check if snake is valid
    check_which_snakes_are_still_alive(snakes, boards)
    

    idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive]

    if len(idxs_of_alive_snakes) > 0:
        boards = get_boards(snakes, num_rows, num_cols)

        food_board = sparse.csr_matrix((num_rows,num_cols))
        for food in all_food:
            food_board[food.location[0][0],food.location[0][1]] = 1


        total_board = sum(boards) + food_board

        # Add more food if necessary
        food_to_add = min_amount_of_food - len(all_food)
        while food_to_add > 0:
            start_x_loc = np.random.randint(0, num_rows, size = 1)
            start_y_loc = np.random.randint(0, num_cols, size = 1)
            if total_board[start_x_loc, start_y_loc] == 0:
                food_board[start_x_loc, start_y_loc] = 1
                total_board[start_x_loc, start_y_loc] = 1
                all_food += [Food(start_x = start_x_loc,
                                   start_y = start_y_loc,
                                   growth = growth)]
                food_to_add -= 1

        return boards + [food_board], idxs_of_alive_snakes
    else:
        return [], []

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
