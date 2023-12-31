import os
import json
import random
import numpy as np
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
        self.x_max = None
        self.y_max = None
        self.color_mode = 'L'

        self.im_generated: Image.Image = None

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

        # result data
        self.climbing_seconds = None

        self.generate_difference_percent = None
        self.good_guesses = None
        self.iteration_epochs = None

        self.difference_list = None
        self.guess_difference_list = None
        self.good_guess_percent_list = None

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
        if self.color_mode is not None: self.im_target = self.im_target.convert(self.color_mode)
        self.num_pixels = np.ones(self.im_target.size).size
        self.x_max = self.im_target.size[0]
        self.y_max = self.im_target.size[1]

    def find_img_diff_pct(self, im1: Image.Image, im2: Image.Image) -> float:
        assert im1.size == im2.size
        return np.asarray(ImageChops.difference(im1, im2)).mean() / self.WHITE

    def modify_im(self, im, percent_done: float) -> Image.Image:
        """general function to return modification of im"""
        alpha = self.alpha(percent_done,
                           func=self.alpha_func,
                           stretch=self.alpha_stretch)

        xy = self.choose_xy(alpha, how=self.shape_center_choice_method)
        fill_color = self.choose_color(alpha=alpha, color_mode=self.color_mode)

        # draw shape
        im_compare = ImageChops.duplicate(im)
        draw = ImageDraw.Draw(im_compare)
        draw.ellipse(xy=xy, fill=fill_color)

        return im_compare

    def choose_color(self, alpha: float, color_mode: str = 'L', color_range_step: int = 1):
        c = None
        if color_mode == 'L':
            c = random.randrange(self.BLACK, self.WHITE + 1, color_range_step)
        else:
            raise NotImplementedError(f'{color_mode} color mode not implemented')
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

    def climb(self, print_output: bool = True):
        assert self.im_target is not None
        # blank starting image
        self.im_generated = Image.new(self.color_mode, self.im_target.size, self.WHITE)
        diff_generated_init = self.find_img_diff_pct(self.im_generated, self.im_target)
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
            im_compare = self.modify_im(im=self.im_generated, percent_done=percent_done)
            # difference of new pic with random shape
            diff_compare = self.find_img_diff_pct(self.im_target, im_compare)

            if diff_compare < diff_generated:
                self.im_generated = ImageChops.duplicate(im_compare)
                diff_generated = diff_compare
                good_guesses += 1

            if epoch % print_freq == 0 and print_output:
                print(f'{epoch}: difference {diff_generated}')

            epoch += 1

            diff_list.append(diff_generated)
            guess_diff_list.append(diff_compare)
            good_guess_pct_list.append(good_guesses / (epoch + 1))

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
    hca.max_iterations = 100_000
    hca.set_seed(2023)
    # print(hca.params)

    hca.load_target_image(target_image_path)

    hca.climb()

    hca.im_generated.show()
    ImageChops.difference(hca.im_target, hca.im_generated).show()

    hca.save(dir='images', im_name='jimi', include_plot=True)

    hca.plot_climb_stats()


if __name__ == '__main__':
    main()
