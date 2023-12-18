import random
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops

"""
Basic implementation of main hill climber code
"""

BLACK = 0
WHITE = 255

target_image_path = 'images/Jimi_Hendrix_1967_uncropped.jpg'
im_target = Image.open(target_image_path).convert('L')
print(im_target.format, im_target.size, im_target.mode)

X_MAX = im_target.size[0]
Y_MAX = im_target.size[1]

CIRCLE_START_RADIUS = 100
CIRCLE_END_RADIUS = 4
CIRCLE_PERCENT_VARIANCE = 0.2

MAX_ITERATIONS = 10_000

SEED = 2023
random.seed(SEED)
np.random.seed(SEED)


def find_img_diff_pct(im1: Image.Image, im2: Image.Image) -> float:
    """finds difference between images and returns float in [0, 1] where 0 for same image"""
    assert im1.size == im2.size
    num_pixels = np.zeros(im1.size).size
    return np.asarray(ImageChops.difference(im1, im2)).sum() / (num_pixels * WHITE)


def loglike(x: float, stretch: int = 40) -> float:
    return np.log((x * stretch) + 1) / np.log(stretch + 1)


def alpha_func(x: float) -> float:
    """x in range [0, 1] returns range [0, 1]"""
    assert 0 <= x <= 1
    # return linear(x)
    # return sigmoid(x)
    return loglike(x)


def main():
    # blank starting image
    im_generated = Image.new('L', im_target.size, WHITE)
    # initial image difference
    diff_generated_init = find_img_diff_pct(im_generated, im_target)
    diff_generated = diff_generated_init
    diff_compare = diff_generated

    num_pixels = np.ones(im_target.size).size

    diff_list = []
    guess_diff_list = []
    good_guess_pct_list = []

    good_guesses = 0
    epoch = 0

    t0 = perf_counter()

    while epoch < MAX_ITERATIONS:
        # We want to use progressively smaller circles as the number of iterations increases since we want
        # to create progressively more detail.
        # The alpha parameter [0, 1] controls where we are between CIRCLE_START_RADIUS and CIRCLE_END_RADIUS.
        # As (epoch / MAX_ITERATIONS) goes from 0 -> 1 the circle_radius_target will decrease
        # from CIRCLE_START_RADIUS to CIRCLE_END_RADIUS.
        # The loglike function makes circle_radius_target spend more iterations on smaller circles
        # since we require more small circles than big to flush out details.
        alpha = alpha_func(epoch / MAX_ITERATIONS)  # increases from 0 to 1
        circle_radius_target = int(CIRCLE_START_RADIUS - (CIRCLE_START_RADIUS - CIRCLE_END_RADIUS) * alpha)
        # add a bit of variance to the circle radius
        circle_radius = int(random.uniform(1 - CIRCLE_PERCENT_VARIANCE,
                                           1 + CIRCLE_PERCENT_VARIANCE) * circle_radius_target)

        # uniform center choice
        circle_center_x = random.randint(0, X_MAX)
        circle_center_y = random.randint(0, Y_MAX)

        circle_start_point = (circle_center_x - circle_radius, circle_center_y - circle_radius)
        circle_stop_point = (circle_center_x + circle_radius, circle_center_y + circle_radius)

        shape_color = random.randrange(BLACK, WHITE+1, 1)

        # copy current working image
        im_compare = ImageChops.duplicate(im_generated)
        # draw a random shape on it
        draw = ImageDraw.Draw(im_compare)
        draw.ellipse((circle_start_point, circle_stop_point), fill=shape_color)

        # find difference of new im_compare with target
        diff_compare = find_img_diff_pct(im_target, im_compare)

        if epoch % int(MAX_ITERATIONS / 20) == 0:
        # if epoch % 2500 == 0:
            print(f'epoch: {epoch}, difference: {diff_generated}, alpha: {alpha}')

        # if new im_compare is more similar to target
        if diff_compare < diff_generated:
            # switch working image to the new im_compare
            im_generated = ImageChops.duplicate(im_compare)
            diff_generated = diff_compare
            good_guesses += 1
        # else just keep the working image without the random shape

        epoch += 1

        diff_list.append(diff_generated)
        guess_diff_list.append(diff_compare)
        good_guess_pct_list.append(good_guesses / epoch)

    t1 = perf_counter()

    print(f'epoch: {epoch}, difference: {diff_generated}, alpha: {alpha}')
    im_generated.show()
    diff_im = ImageChops.difference(im_target, im_generated)
    diff_im.show()
    print(f'good guesses: {good_guesses}')
    print(f'good guess percent: {good_guesses / epoch}')
    print(f'Total time: {round((t1 - t0) / 60, 2)} min, '
          f'Average of {round((t1 - t0) / epoch, 5)} sec / guess')

    x = np.arange(MAX_ITERATIONS)

    fig, ax = plt.subplots()

    ax.plot(x, good_guess_pct_list, label='correct guess %')
    ax.plot(x, guess_diff_list, label='guess difference')

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    fig.legend()

    plt.show()


if __name__ == '__main__':
    main()
