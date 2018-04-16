

def get_implied_board(food, screen_width, screen_height):
    '''
    Given a snake, generate the sparse matrix which represents the board and has the snake filled in with 1s
    '''        
    pixels = np.array(food.location)
    rows = pixels[:,0]
    cols = pixels[:,1]
   
    sparse_matrix = sparse.coo_matrix(([1]*1, (rows,cols)), 
                             shape = (screen_width, screen_height)).astype(np.uint8)   

    return sparse_matrix.tocsr()