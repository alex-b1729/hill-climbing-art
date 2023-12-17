import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops

def find_img_diff_pct(im1: Image.Image, im2: Image.Image) -> float:
    assert im1.size == im2.size
    num_pixles = np.zeros(im1.size).size
    return np.asarray(ImageChops.difference(im1, im2)).sum() / (num_pixles * 255)

# target image
target_image_path = 'images/Jimi_Hendrix_1967_uncropped.jpg'
# target_image_path = 'images/Train_wreck_at_Montparnasse_1895.jpg'
# target_image_path = 'images/Hindenburg_disaster.jpg'
# target_image_path = 'images/Statue_of_Liberty_12.jpg'
# target_image_path = 'images/pawns.jpg'
im_target = Image.open(target_image_path).convert('L')
print(im_target.format, im_target.size, im_target.mode)
# im.show()

X_MAX = im_target.size[0]
Y_MAX = im_target.size[1]

BLACK = 0
WHITE = 255

CIRCLE_START_RADIUS = 100
CIRCLE_END_RADIUS = 4
CIRCLE_PERCENT_VARIANCE = 0.2

MAX_ITERATIONS = 100_000
TARGET_DIFF_PERCENT = 0

SEED = 2023
random.seed(SEED)

# white starting image
im_generated = Image.new('L', im_target.size, BLACK)
diff_generated_init = find_img_diff_pct(im_generated, im_target)
diff_generated = diff_generated_init
diff_compare = diff_generated

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x

def loglike(x):
    stretch = 80 # > 0
    return np.log((x * stretch) + 1) / np.log(stretch + 1)

def alpha_func(x):
    """x in range [0, 1] returns range [0, 1]"""
    # return linear(x)
    # return sigmoid(x)
    return loglike(x)


diff_list = []
guess_diff_list = []
good_guess_pct_list = []

good_guesses = 0
epoch = 0
while epoch < MAX_ITERATIONS and diff_generated > TARGET_DIFF_PERCENT:
    alpha = alpha_func(epoch / MAX_ITERATIONS)  # increases from 0 to 1
    circle_radius_target = int(CIRCLE_START_RADIUS - (CIRCLE_START_RADIUS - CIRCLE_END_RADIUS) * alpha)
    # circle_radius = circle_radius_target
    circle_radius = int(random.uniform(1 - CIRCLE_PERCENT_VARIANCE, 1 + CIRCLE_PERCENT_VARIANCE) * circle_radius_target)
    color_range_step = 1  # not implementing changing colors for now

    circle_center_x = random.randint(0, X_MAX)
    circle_center_y = random.randint(0, Y_MAX)
    circle_start_point = (circle_center_x - circle_radius, circle_center_y - circle_radius)
    circle_stop_point = (circle_center_x + circle_radius, circle_center_y + circle_radius)

    shape_color = random.randrange(BLACK, WHITE+1, 1)

    # draw random shape
    im_compare = ImageChops.duplicate(im_generated)
    draw = ImageDraw.Draw(im_compare)
    draw.ellipse((circle_start_point, circle_stop_point), fill=shape_color)

    # difference of new pic with random shape
    diff_compare = find_img_diff_pct(im_target, im_compare)

    # if epoch % int(MAX_ITERATIONS / 20) == 0:
    if epoch % 2500 == 0:
        print(f'{epoch}: {diff_generated}, alpha: {alpha}')

    if diff_compare < diff_generated:
        im_generated = ImageChops.duplicate(im_compare)
        diff_generated = diff_compare
        good_guesses += 1

    if epoch != 0:
        diff_list.append(diff_generated)
        guess_diff_list.append(diff_compare)
        good_guess_pct_list.append(good_guesses / (epoch + 1))

    epoch += 1

print(f'{epoch}: {diff_generated}, alpha: {alpha}')
im_generated.show()
diff_im = ImageChops.difference(im_target, im_generated)
ImageChops.invert(diff_im).show()
print(f'good guesses: {good_guesses}')
print(f'good guess percent: {good_guesses / epoch}')

x = np.arange(MAX_ITERATIONS-1)

fig, ax = plt.subplots()

# ax.plot(x, diff_list, label='gudifference')
ax.plot(x, good_guess_pct_list, label='correct guess %')
ax.plot(x, guess_diff_list, label='guess difference')

ax.set_yscale('log')
fig.legend()

plt.show()

# diff = find_img_diff_pct(im, start_im)
# print(diff)
#
# draw = ImageDraw.Draw(start_im)
# draw.ellipse(((-100, -100), (100, 100)), fill=BLACK)
#
# diff = find_img_diff_pct(im, start_im)
# print(diff)
#
# start_im.show()

# ImageChops.difference(im, start_im).show()

# sz = (10, 10)
# blk = 0 #(0, 0, 0)
# wht = 255 # (255, 255, 255)
# im1 = Image.new('L', sz, blk)
# draw1 = ImageDraw.Draw(im1)
# draw1.ellipse(((0, 0), (2, 2)), fill=10)
# im1.show()
# npim = np.asarray(im1)
# print(npim)




