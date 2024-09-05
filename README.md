# Hill Climbing Art - with Python

Implements a [Hill Climbing algorithm](https://en.wikipedia.org/wiki/Hill_climbing) to replicate a 
target) image by drawing random circles. 

![Jimi Hendrix Hill Climbing Art](./images/jimi_2023-12-23-131801.jpg)
[Original image source](https://commons.wikimedia.org/wiki/File:Jimi_Hendrix_1967_uncropped.jpg)

## Overview
This hill climbing algo draws random circles on a blank canvas. 
Circles that make the generated image more similar to the target image are kept while those that make the 
generated image less similar are discarded.  

## Examples

I generated the image above of Jimi Hendrix using the following code. 
Running this took just under 7 minutes my 2015 MacBook, about 0.004 seconds / epoch. 
The `hca.color_mode = 'L'` sets the image to grayscale. 
See documentation for [Pillow modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes). 

```python
target_image_path = 'images/Jimi_Hendrix_1967_uncropped.jpg'

hca = HillClimbingArtist()

hca.max_iterations = 100_000
hca.color_mode = 'L'

hca.load_target_image(target_image_path)

hca.climb()

hca.im_generated.show()
```

## Future improvements
- [ ] Shapes besides circles
- [x] Colored images
- [ ] Choose the average circle color to be the color of the pixel at the circle center point. 
