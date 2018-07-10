import gym
import gym_snake
import pdb
import time
import numpy as np
from gym.envs.classic_control import rendering
import curses
import sys
import numpy
'''
THIS IS A HACKY SCRIPT TO PLAY SNAKE BY HAND.
'''
human_colors = {'maroon':(128,0,0), 'dark_red':(139,0,0), 'brown':(165,42,42), 'firebrick':(178,34,34), 'crimson':(220,20,60), 'red':(255,0,0), 'tomato':(255,99,71), 'coral':(255,127,80), 'indian_red':(205,92,92), 'light_coral':(240,128,128), 'dark_salmon':(233,150,122), 'salmon':(250,128,114), 'light_salmon':(255,160,122), 'orange_red':(255,69,0), 'dark_orange':(255,140,0), 'orange':(255,165,0), 'gold':(255,215,0), 'dark_golden_rod':(184,134,11), 'golden_rod':(218,165,32), 'pale_golden_rod':(238,232,170), 'dark_khaki':(189,183,107), 'khaki':(240,230,140), 'olive':(128,128,0), 'yellow':(255,255,0), 'yellow_green':(154,205,50), 'dark_olive_green':(85,107,47), 'olive_drab':(107,142,35), 'lawn_green':(124,252,0), 'chart_reuse':(127,255,0), 'green_yellow':(173,255,47), 'dark_green':(0,100,0), 'green':(0,128,0), 'forest_green':(34,139,34), 'lime':(0,255,0), 'lime_green':(50,205,50), 'light_green':(144,238,144), 'pale_green':(152,251,152), 'dark_sea_green':(143,188,143), 'medium_spring_green':(0,250,154), 'spring_green':(0,255,127), 'sea_green':(46,139,87), 'medium_aqua_marine':(102,205,170), 'medium_sea_green':(60,179,113), 'light_sea_green':(32,178,170), 'dark_slate_gray':(47,79,79), 'teal':(0,128,128), 'dark_cyan':(0,139,139), 'aqua':(0,255,255), 'cyan':(0,255,255), 'light_cyan':(224,255,255), 'dark_turquoise':(0,206,209), 'turquoise':(64,224,208), 'medium_turquoise':(72,209,204), 'pale_turquoise':(175,238,238), 'aqua_marine':(127,255,212), 'powder_blue':(176,224,230), 'cadet_blue':(95,158,160), 'steel_blue':(70,130,180), 'corn_flower_blue':(100,149,237), 'deep_sky_blue':(0,191,255), 'dodger_blue':(30,144,255), 'light_blue':(173,216,230), 'sky_blue':(135,206,235), 'light_sky_blue':(135,206,250), 'midnight_blue':(25,25,112), 'navy':(0,0,128), 'dark_blue':(0,0,139), 'medium_blue':(0,0,205), 'blue':(0,0,255), 'royal_blue':(65,105,225), 'blue_violet':(138,43,226), 'indigo':(75,0,130), 'dark_slate_blue':(72,61,139), 'slate_blue':(106,90,205), 'medium_slate_blue':(123,104,238), 'medium_purple':(147,112,219), 'dark_magenta':(139,0,139), 'dark_violet':(148,0,211), 'dark_orchid':(153,50,204), 'medium_orchid':(186,85,211), 'purple':(128,0,128), 'thistle':(216,191,216), 'plum':(221,160,221), 'violet':(238,130,238), 'magenta_/_fuchsia':(255,0,255), 'orchid':(218,112,214), 'medium_violet_red':(199,21,133), 'pale_violet_red':(219,112,147), 'deep_pink':(255,20,147), 'hot_pink':(255,105,180), 'light_pink':(255,182,193), 'pink':(255,192,203), 'antique_white':(250,235,215), 'beige':(245,245,220), 'bisque':(255,228,196), 'blanched_almond':(255,235,205), 'wheat':(245,222,179), 'corn_silk':(255,248,220), 'lemon_chiffon':(255,250,205), 'light_golden_rod_yellow':(250,250,210), 'light_yellow':(255,255,224), 'saddle_brown':(139,69,19), 'sienna':(160,82,45), 'chocolate':(210,105,30), 'peru':(205,133,63), 'sandy_brown':(244,164,96), 'burly_wood':(222,184,135), 'tan':(210,180,140), 'rosy_brown':(188,143,143), 'moccasin':(255,228,181), 'navajo_white':(255,222,173), 'peach_puff':(255,218,185), 'misty_rose':(255,228,225), 'lavender_blush':(255,240,245), 'linen':(250,240,230), 'old_lace':(253,245,230), 'papaya_whip':(255,239,213), 'sea_shell':(255,245,238), 'mint_cream':(245,255,250), 'slate_gray':(112,128,144), 'light_slate_gray':(119,136,153), 'light_steel_blue':(176,196,222), 'lavender':(230,230,250), 'floral_white':(255,250,240), 'alice_blue':(240,248,255), 'ghost_white':(248,248,255), 'honeydew':(240,255,240), 'ivory':(255,255,240), 'azure':(240,255,255), 'snow':(255,250,250), 'black':(0,0,0), 'dim_gray':(105,105,105), 'gray':(128,128,128), 'dark_gray':(169,169,169), 'silver':(192,192,192), 'light_gray':(211,211,211), 'gainsboro':(220,220,220), 'white_smoke':(245,245,245), 'white':(255,255,255)}
def get_closest_color(color, human_colors = human_colors):
    color = [256*c for c in color]

    def eucl_distance(x, y):
        # sum sqare difference. No sqrt since monotone increasing
        return sum([(z[0]-z[1])**2 for z in zip(x,y)])

    distances = []
    for hc in human_colors:
        possibility = human_colors[hc]
        distances.append(eucl_distance(color, possibility))

    return human_colors.keys()[np.argmin(distances)]


which_game = 0
while not which_game:
    which_game = raw_input("Press a for single snake, b for multi-snake, q for exit: ")
    if which_game == 'q':
        sys.exit()
    elif which_game == 'a':
        which_game = 0
        break
    elif which_game == 'b':
        which_game = 1
        break
    else:
        which_game = 0

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

print '*'*20
print 'Instructions: '
print 'Press w for up. s for down. a for left. d for right'
print '*'*20

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

np.random.seed(5)
scaler = 10
viewer = rendering.SimpleImageViewer()
env = gym.make('snake-v%s' % which_game) # Make the gym environment


key = curses.KEY_RIGHT
for j in range(10):
    env.reset()

    print '*'*20
    for idx, c in enumerate([get_closest_color(snake.color) for snake in env.world.snakes]):
        print 'Snake %s is %s' % (idx, c)
    print '*'*20
    
    rgb = env.render('rgb_array') 
    upscaled=repeat_upsample(rgb,scaler,scaler)
    viewer.imshow(upscaled)
    time.sleep(.00000001)
    
    for i in range(10000):
        
        if len(env.world.idxs_of_alive_snakes):
            
            try:
                if (0 in env.world.idxs_of_alive_snakes):
                    key1 = read_single_keypress()
                else:
                    key1 = 'w' #arbitary. Snake is dead.
                
                if (which_game == 1) and (1 in env.world.idxs_of_alive_snakes):
                    key2 = read_single_keypress()
                elif (which_game == 1):
                    key2 = 'w' #arbitary. Snake is dead.
                else:
                    pass #single snake game


                if which_game == 0:
                    states_n, rewards_n, done_n = env.step([actions[key1]])
                if which_game == 1:
                    states_n, rewards_n, done_n = env.step([actions[key1], actions[key2]])

                print rewards_n, done_n 
                rgb = env.render('rgb_array') 
                upscaled=repeat_upsample(rgb,scaler,scaler)
                viewer.imshow(upscaled)
                time.sleep(.00000001)
            except:
                pdb.set_trace()


