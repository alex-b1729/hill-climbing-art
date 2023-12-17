# Hill Climbing Art - with Python

## Implementation brainstorming
- Allow new circles that make the generated image more different, as long as difference is within a given limit. 
- Choose average circle size using variance of the difference image array. 
- Choose the average circle color to be the color of the pixel at the circle center point, with some variance. 