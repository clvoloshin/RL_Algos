from moviepy.editor import ImageSequenceClip
import numpy as np
import os

class Monitor(object):
    def __init__(self, save_dir):
        self.images = {}
        self.save_dir = save_dir

        if not(os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

    def add(self, image, epoch, game_number, scaled_to_1 = True):
        if epoch not in self.images:
            self.images[epoch] = {}
        if game_number not in self.images[epoch]:
            self.images[epoch][game_number] = []

        if scaled_to_1:
            self.images[epoch][game_number].append((image*255).astype(np.uint8))
        else:
            self.images[epoch][game_number].append(image)

    def make_gif(self, filename, epoch, game_number, fps=24, scale = 1.0):
        clip = ImageSequenceClip(self.images[epoch][game_number], fps=fps).resize(scale)
        clip.write_gif(filename, fps=fps)
        return clip

    def make_gifs(self, epoch, fps=24, scale = 1.0):
        if epoch in self.images:
            for game_number in self.images[epoch]:
                self.make_gif(os.path.join(self.save_dir,'%s_%s.gif' % (epoch, game_number)), epoch, game_number, fps=fps, scale=scale)
            del self.images[epoch]                


