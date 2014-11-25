##Overview
The scripts in this directory are designed to remove the camera motion from All-22 video of NFL games, although is likely to be generic enough to work in some other cases (e.g. other sports or different NFL footage). 

The main program is rectify_images.py, which uses several functions and classes from alignlib.py. The calling syntax is:

[Video Filename] [First Frame #] [Last Frame #] [Frames to Skip (1 for no skip)] [Output Images Prefix] [Print Original Images (Y/N)]

Most of these should be self explanatory - the output images prefix defines the prefix to prepend the individual output frames with, and selecting 'Y' for print original images will output the individual frames of the original video in addition to the rectified frames.

There is also the convenience script rectify_with_animation.sh, which has the same calling syntax as rectify_images.py. This script will remove the camera motion from the frames, then stitch them together back into an animated GIF. It will then delete the intermediate frames to save space and clutter.

##Requirements
PhDVision depends on several packages. Most crucial is the dependence on the OpenCV library, which is written in C++. You'll need to download the library itself as well as the python bindings. Additionally, PhDVision requires the following Python modules:
*numpy
*scipy
*scikit-image
*matplotlib
*bottleneck

An OpenCV installation can be non-trivial, so hopefully soon the scikit-image library will be mature enough to fully replace it in the code.
