# mandelbrot

This repository contains a few CUDA files related to the generation of Mandelbrot-type fractals. The programs have been tested on my 6900 XT via [SCALE](https://docs.scale-lang.com/).
The base `mandelbrot.cu` generates images zooming into the point `-0.8212-0.2006i`, overall applying a ~13000x zoom from the original image.
`mandelbrot-double.cu` zooms into `-0.8212007965493-0.200572441411i`, overall providing a ~6.7*10^12x zoom from the original image.
`multibrot` and `negabrot` both image the [Multibrot set](https://en.wikipedia.org/wiki/Mandelbrot_set#Multibrot_sets) for various d. In `multibrot`, d ranges from 1 to 6, while it ranges from -1 to -5 in `negabrot`.

All the programs dump their generated images as PNGs in a folder. From there, it's possible to make a video from the images using `ffmpeg`.

Generated Multibrot fractals:

https://github.com/user-attachments/assets/8d72ee8a-9561-4529-aa4c-7d86bc1bed3b

https://github.com/user-attachments/assets/46643e77-2f37-4c58-acc6-9b0692a096f2

