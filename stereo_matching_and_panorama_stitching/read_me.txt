
To run stereo_matching.py, enter:
    python stereo_matching.py

To run panorama_stitching.py, enter:
    python panorama_stitching.py

Python program panorama_stitching.py will end displaying a picture named 'panorama'.
Press any key to close that window

Configuration:
    * Python 2.7.12
    * opencv3 3.2.0 (installed via Homebrew)

*: Best RMS distance with gaussian filter:
      sigma = 1
   Best RMS distance with bilateral filter:11.0878808191
      d=5
      sigmaColor=80
      sigmaSpace=80

***************************** Issues *********************************

Stereo_matching.py:
    1. the color of depth_l is the inverse of depth_r, no idea so far (fixed)
    2. bilateral filter does not give very promising pic (fixed)
    3. the RMS distance with joint bilateral filter is higher than that of gaussian (fixed)

Panorama_stitching.py:
    1. numpy.linalg.lstsq failed to output the correct result (numpy.linalg.solve works well so far)


