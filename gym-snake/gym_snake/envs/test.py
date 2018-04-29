import snake_env
import pdb
import time
import numpy as np
from gym.envs.classic_control import rendering
import curses

'''
THIS IS A HACKY SCRIPT TO PLAY SNAKE BY HAND.
'''



actions = {
    'w':    '0',
    's':  '1',
    'a':  '2',
    'd': '3',
    }

reverse_action_space = {}
reverse_action_space['0'] = {'-1': 'down', '1':'up'}
reverse_action_space['1'] = {'0': 'right'}
reverse_action_space['-1'] = {'0': 'left'}

def read_single_keypress():
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK 
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR 
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1) # returns a single character
    except KeyboardInterrupt: 
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

scaler = 10
viewer = rendering.SimpleImageViewer()
env = snake_env.SnakeEnv(screen_width = 40/scaler, screen_height = 40/scaler)

key = curses.KEY_RIGHT
for j in range(10):
    env.reset()
    rgb = env.render('rgb_array') 
    upscaled=repeat_upsample(rgb,scaler,scaler)
    viewer.imshow(upscaled)
    time.sleep(.00000001)
    
    for i in range(10000):
        
        if len(env.world.idxs_of_alive_snakes):
            
            
            key = read_single_keypress()

            states_n, rewards_n, done_n = env.step([actions[key]])
            print rewards_n, done_n 
            rgb = env.render('rgb_array') 
            upscaled=repeat_upsample(rgb,scaler,scaler)
            viewer.imshow(upscaled)
            time.sleep(.00000001)


