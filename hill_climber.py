import os
import json
import random
import numpy as np
from math import log2
import datetime as dt
from time import perf_counter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops, ImageFilter


class HillClimbingArtist(object):
    """
    Documentation...
    """
    BLACK = 0
    WHITE = 255

    def __init__(self):
        # image params
        self.path_target = None
        self.im_target: Image.Image = None
        self.num_pixels = None
        self.num_climb_pixels = None
        self.x_max = None
        self.y_max = None
        self.color_mode = None
        self.channels_to_climb = None
        self.color_dims = None
        self.climb_dims = None

        # self.im_generated: Image.Image = None
        self.im_climbed: Image.Image = None

        self.np_target = None
        self.np_target_climb = None
        self.np_target_dont_climb = None

        self.im_target_climb = None
        self.im_target_dont_climb = None

        # climbing gif settings
        self._record_im_generated_gif: bool = False
        self._gif_path: str = ''
        self._progress_ims: list = []

        # generation params
        self.set_seed()
        self.max_iterations = 10_000
        self.target_difference_percent = 0
        self.start_radius = 100
        self.end_radius = 5
        self.radius_percent_variance = 0.2
        self.alpha_func = 'loglike'
        self.alpha_stretch = 80
        self.shape_center_choice_method = 'uniform'
        self.use_edge_attraction = False  # whether to make shape placement more likely in more different pixels
        self.edge_probability_weight = 0.7  # ignored if self.use_edge_attraction == False
        self.color_epsilon = 0
        self.epsilon = 0

        # result data
        self.climbing_seconds = None

        self.generate_difference_percent = None
        self.good_guesses = None
        self.iteration_epochs = None

        self.difference_list = None
        self.guess_difference_list = None
        self.good_guess_percent_list = None

    @property
    def im_generated(self):
        return ImageChops.add(self.im_climbed, self.im_target_dont_climb).convert('RGB')

    def save_climbing_gif(self, path: str):
        """Generates gif of climbing and saves to path.
        Must be called before climb()"""
        self._record_im_generated_gif = True
        self._gif_path = path

    def set_seed(self, seed: int = None):
        if seed is not None: self.seed = seed
        else: self.seed = random.randrange(1000)
        random.seed(self.seed)
        np.random.seed(self.seed)

    @property
    def params(self) -> dict:
        return {
            'seed': self.seed
            ,'max_iterations': self.max_iterations
            ,'target_difference_percent': self.target_difference_percent
            ,'start_radius': self.start_radius
            ,'end_radius': self.end_radius
            ,'radius_percent_variance': self.radius_percent_variance
            ,'alpha_func': self.alpha_func
            ,'alpha_stretch': self.alpha_stretch
            ,'shape_center_choice_method': self.shape_center_choice_method
            ,'use_edge_attraction': self.use_edge_attraction
            ,'edge_probability_weight': self.edge_probability_weight
        }

    def export_params(self, path: str):
        with open(path, 'w') as f:
            f.write(json.dumps(self.params, indent=4))

    def import_params(self, path: str):
        with open(path, 'r') as f:
            params = json.loads(f.read())

        self.set_seed(params['seed'])
        self.max_iterations = params['max_iterations']
        self.target_difference_percent = params['target_difference_percent']
        self.start_radius = params['start_radius']
        self.end_radius = params['end_radius']
        self.radius_percent_variance = params['radius_percent_variance']
        self.alpha_func = params['alpha_func']
        self.alpha_stretch = params['alpha_stretch']
        self.shape_center_choice_method = params['shape_center_choice_method']
        self.use_edge_attraction = params['use_edge_attraction']
        self.edge_probability_weight = params['edge_probability_weight']

    @property
    def metadata(self) -> dict:
        return {
            'path_target': self.path_target
            , 'im_target_format': self.im_target.format
            , 'im_target_size': self.im_target.size
            , 'im_target_mode': self.im_target.mode
            , 'num_pixels': self.num_pixels
            , 'x_max': self.x_max
            , 'y_max': self.y_max
            , 'climbing_seconds': self.climbing_seconds
            , 'generated_difference_percent': self.generate_difference_percent
            , 'good_guesses': self.good_guesses
            , 'iteration_epochs': self.iteration_epochs
        }

    def save_metadata(self, path: str):
        with open(path, mode='w') as f:
            f.write(json.dumps(self.metadata, indent=4))

    def load_target_image(self, path_target: str):
        self.im_target = Image.open(path_target)
        if self.color_mode is not None:
            self.im_target = self.im_target.convert(self.color_mode)
        else:
            self.color_mode = self.im_target.mode
        self.num_pixels = np.ones(self.im_target.size).size
        self.x_max = self.im_target.size[0]
        self.y_max = self.im_target.size[1]

        self.np_target = np.array(self.im_target)

    def find_img_diff_pct(self, im1: Image.Image, im2: Image.Image, scale: float = 1) -> float:
        assert im1.size == im2.size
        return np.asarray(ImageChops.difference(im1, im2)).mean() / (self.WHITE * scale)

    def modify_im(self, im, percent_done: float) -> [Image.Image, float]:
        """general function to return modification of im"""
        alpha = self.alpha(percent_done,
                           func=self.alpha_func,
                           stretch=self.alpha_stretch)

        xy = self.choose_xy(alpha, how=self.shape_center_choice_method)
        fill_color = self.choose_color(alpha=alpha, color_mode=self.color_mode) #, xy=xy)

        # draw shape
        im_compare = ImageChops.duplicate(im)
        draw = ImageDraw.Draw(im_compare)
        draw.ellipse(xy=xy, fill=fill_color)
        # pie_start = random.randint(0, 360)
        # pie_min_angle = int(alpha * 35)
        # pie_max_angle = int(alpha * 40 + 50)
        # draw.pieslice(xy=xy, start=pie_start,
        #               end=(pie_start+random.randint(2, 65)),
        #               fill=fill_color)
        return im_compare, xy

    def choose_color(self, alpha: float, color_mode: str = 'L', color_range_step: int = 1, xy=None):
        # todo: implement chosing using center of circle color
        # if xy is not None:
        #     if random.random() < self.color_epsilon:
        #         c = int(np.asarray(self.im_target)[int((xy[0][1] + xy[1][1]) / 2) - 1][int((xy[0][0] + xy[1][0]) / 2) - 1])
        #     else:
        #         c = random.randrange(self.BLACK, self.WHITE + 1, color_range_step)
        # else:
        #     if color_mode == 'L':
        #         c = random.randrange(self.BLACK, self.WHITE + 1, color_range_step)
        #     else:
        #         raise NotImplementedError(f'{color_mode} color mode not implemented')
        c = tuple(random.randrange(self.BLACK, self.WHITE, color_range_step)
                  if d else 0 for d in self.channels_to_climb)
        return c

    def choose_xy(self, alpha: float, how: str = 'uniform') -> tuple:
        """returns xy bounding box"""
        radius_mean = int(self.start_radius -
                          (self.start_radius - self.end_radius) * alpha)
        # circle_radius = circle_radius_target
        radius = int(random.uniform(1 - self.radius_percent_variance,
                                    1 + self.radius_percent_variance) * radius_mean)
        start_point, stop_point = None, None
        if how == 'uniform':
            x = random.randint(0, self.x_max)
            y = random.randint(0, self.y_max)
            start_point = (x - radius, y - radius)
            stop_point = (x + radius, y + radius)
        elif how == 'edge_attraction':
            self.edge_attraction()
        else:
            raise NotImplementedError(f'{how} xy choice not defined')

        return start_point, stop_point

    def edge_attraction(self):
        raise NotImplementedError('Not supported yet')
        # index_choice = np.arange(self.num_pixels)
        # blank_im = Image.new('L', self.im_target.size, self.WHITE)

        # # weight circle start point using difference image
        # diff_im = ImageChops.difference(self.im_target, im_generated)
        #
        # # blur difference by avg circle radius
        # diff_im = diff_im.filter(ImageFilter.BoxBlur(circle_radius_mean))
        # diff_array = np.array(ImageChops.blend(blank_im, diff_im, alpha=self.edge_probability_weight)).flatten()
        # diff_probs = diff_array / diff_array.sum()
        #
        # diff_array = np.array(ImageChops.blend(blank_im, diff_im, alpha=self.edge_probability_weight))
        # diff_array = diff_array / diff_array.sum()
        # center_indx = np.random.choice(index_choice, p=diff_array.flatten())
        # circle_center_x = center_indx % self.x_max
        # circle_center_y = center_indx // self.x_max

    def reset_default_channels_to_climb(self):
        ndims = self.np_target.ndim
        if ndims == 2:
            self.channels_to_climb = np.array([True])
        else:
            self.channels_to_climb = np.array(tuple(True for _ in range(ndims)))

    def climb_setup(self):
        assert self.im_target is not None
        if self.channels_to_climb is None:
            self.reset_default_channels_to_climb()
        self.color_dims = self.channels_to_climb.size
        self.climb_dims = self.channels_to_climb.sum()
        self.num_climb_pixels = np.prod(self.im_target.size) * self.climb_dims

        self.np_target_climb = self.np_target.copy()
        self.np_target_dont_climb = self.np_target.copy()
        if self.channels_to_climb.size != 1:
            self.np_target_climb[:,:,~self.channels_to_climb] = 0
            self.np_target_dont_climb[:,:,self.channels_to_climb] = 0
        else:
            self.np_target_dont_climb = np.zeros(
                self.np_target_dont_climb.shape,
                dtype=self.np_target.dtype
            )

        self.im_target_climb = Image.fromarray(self.np_target_climb)
        self.im_target_dont_climb = Image.fromarray(self.np_target_dont_climb)

        # blank starting image
        self.im_climbed = Image.new(
            mode=self.color_mode,
            size=self.im_target.size,
            # todo: allow white starting background
            color=tuple(self.BLACK for _ in range(self.color_dims))
        )

    def climb(self, print_output: bool = True, display_output: bool = False):
        self.climb_setup()

        if display_output:
            plt.ion()

        imshow_args = {}
        if self.color_mode == 'L':
            imshow_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 255}

        diff_generated_init = self.find_img_diff_pct(
            ImageChops.add(self.im_climbed, self.im_target_dont_climb),
            self.im_target,
            scale=(self.num_climb_pixels / self.num_pixels)
        )
        diff_generated = diff_generated_init

        # print output either 20 times or at least every 2500 epochs
        print_freq = min(int(self.max_iterations / 20), 2500)
        if print_freq == 0: print_freq = 1  # avoid div by 0 with few iterations

        diff_list = []
        guess_diff_list = []
        good_guess_pct_list = []

        good_guesses = 0
        epoch = 0

        t0 = perf_counter()

        while epoch < self.max_iterations and diff_generated > self.target_difference_percent:
            percent_done = epoch / self.max_iterations

            # generate random modification
            im_compare, xy = self.modify_im(im=self.im_climbed, percent_done=percent_done)

            # append to progress im list
            if (self._record_im_generated_gif and
                ((int(log2(epoch + 1)) == log2(epoch + 1) and epoch < 1_024) or
                 (epoch % 1_000 == 0 and epoch >= 1_024))):
                self._progress_ims.append(
                    ImageChops.add(im_compare, self.im_target_dont_climb)
                              .resize((self.im_target.size[0] // 3,
                                       self.im_target.size[1] // 3))
                              .convert('P')
                )

            # difference of new pic with random shape
            diff_compare = self.find_img_diff_pct(
                ImageChops.add(im_compare, self.im_target_dont_climb),
                self.im_target,
                scale=(self.num_climb_pixels / self.num_pixels)
            )

            # let epsilon vary with shape top left corner
            # dist = np.sqrt((xy[0][0] / self.x_max)**2 + (xy[0][1] / self.y_max)**2) / np.sqrt(2)
            dist = 1

            if diff_compare < diff_generated: # * (1 + self.epsilon * dist):
                self.im_climbed = ImageChops.duplicate(im_compare)
                diff_generated = diff_compare
                good_guesses += 1

            if epoch % print_freq == 0 and print_output:
                print(f'{epoch}: difference {diff_generated}')

            epoch += 1

            diff_list.append(diff_generated)
            guess_diff_list.append(diff_compare)
            good_guess_pct_list.append(good_guesses / (epoch + 1))

            if display_output and epoch % 100 == 0:
                fig, ax = plt.subplots(2, 1, num=1, width_ratios=[1], height_ratios=[3, 1])
                im_to_show = np.array(ImageChops.add(im_compare, self.im_target_dont_climb).convert('RGB'))
                ax[0].imshow(im_to_show, **imshow_args)

                x = np.arange(epoch)
                ax[1].plot(x, good_guess_pct_list, label='good guess %')
                ax[1].plot(x, guess_diff_list, label='guess difference')
                ax[1].set_yscale('log')
                ax[1].set_xlabel('Epoch')
                fig.legend()

                plt.show()
                plt.pause(0.0001)
                plt.clf()

        t1 = perf_counter()
        self.climbing_seconds = t1 - t0

        self.generate_difference_percent = diff_generated
        self.good_guesses = good_guesses
        self.iteration_epochs = epoch

        self.difference_list = diff_list
        self.guess_difference_list = guess_diff_list
        self.good_guess_percent_list = good_guess_pct_list

        if print_output:
            print(f'{self.iteration_epochs}: difference {self.generate_difference_percent}')
            print(f'Number of good guesses: {self.good_guesses}')
            print(f'Good guess percent: {self.good_guesses / self.iteration_epochs}')
            print(f'Total time: {round(self.climbing_seconds / 60, 2)} min')
            print(f'Average of {round(self.climbing_seconds / self.iteration_epochs, 5)} sec / guess')

        if self._record_im_generated_gif:
            self._progress_ims[0].save(
                self._gif_path, save_all=True,
                append_images=self._progress_ims[1:],
                optimize=True, duration=40, loop=0
            )

        if display_output:
            plt.ioff()

    def plot_climb_stats(self, save: bool = False, dir: str = None):
        assert self.im_generated is not None
        x = np.arange(self.max_iterations)

        fig, ax = plt.subplots()

        ax.plot(x, self.good_guess_percent_list, label='good guess %')
        ax.plot(x, self.guess_difference_list, label='guess difference')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        fig.legend()

        if save:
            if dir is None: fname = f'{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}.jpg'
            else: fname = dir
            plt.savefig(fname)
        else:
            plt.show()

    def save(self, dir: str, im_name: str = None,
             include_plot: bool = False,
             include_params: bool = True,
             include_metadata: bool = True):
        save_time = dt.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        if im_name is not None: im_path = os.path.join(dir, f'{im_name}_{save_time}.jpg')
        else: im_path = os.path.join(dir, f'{save_time}.jpg')
        self.im_generated.save(im_path)
        if include_plot: self.plot_climb_stats(save=True, dir=os.path.join(dir, f'{save_time}_climb_stats.jpg'))
        if include_params: self.export_params(os.path.join(dir, f'{save_time}_params.json'))
        if include_metadata: self.save_metadata(os.path.join(dir, f'{save_time}_metadata.json'))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        return x

    def loglike(self, x: float, stretch: int = 80) -> float:
        return np.log((x * stretch) + 1) / np.log(stretch + 1)

    def alpha(self, x: float, func: str, stretch: int = 80) -> float:
        """x in range [0, 1] returns range [0, 1]"""
        assert 0 <= x <= 1
        if func == 'linear':
            return self.linear(x)
        elif func == 'sigmoid':
            return self.sigmoid(x)
        elif func == 'loglike':
            return self.loglike(x, stretch=stretch)
        else:
            raise NotImplementedError(f'{func} function not yet defined')


def main():
    target_image_path = 'images/Jimi_Hendrix_1967_uncropped.jpg'

    hca = HillClimbingArtist()

    hca.max_iterations = 10_000
    hca.set_seed(2023)
    hca.start_radius = 100
    hca.end_radius = 18
    hca.alpha_stretch = 80

    hca.color_mode = 'HSV'
    hca.channels_to_climb = np.array((True, False, False))

    hca.load_target_image(target_image_path)

    # hca.save_climbing_gif('images/anothertest.gif')
    hca.climb(print_output=True, display_output=True)

    hca.im_generated.show()
    ImageChops.difference(hca.im_target, hca.im_generated).show()

    # hca.save(dir='images', im_name='jimi', include_plot=True)

    hca.plot_climb_stats()


if __name__ == '__main__':
    main()
