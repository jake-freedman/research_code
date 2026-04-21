import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hsv_to_rgb, to_hex

BLACK = '#000000'

VIOLET1 = '#9480FF'
GREEN1 = '#44DE9B'
RED1 = '#F26767'
GRAY1 = '#675D5D'
BLUE1 = '#3e54a3'
ORANGE1 = '#f9a619'
YELLOW1 = '#F9F293'
TEAL1 = '#43e8d8'
PINK1 = '#eea1cd'

GREEN2 = '#93C572'
LIGHTBLUE2 = '#b2cbf2'
DARKBLUE2 = '#475c6c'
ORANGE2 = '#FBD8A2'
LIGHTGRAY2 = '#cccccc'
DARKGRAY2 = '#777777'
RED2 = '#e5a3a3'
TAN2 = '#EED7A1'
BEIGE2 = '#d9b99b'
PINK2 = '#e6b8d0'
BLUE2 = "#2522d4"
VIOLET2 = '#C2B7E9'
DARKGREEN2 = '#8DB591'

# plot settings
axes_width_mm = 150.0
axes_height_mm = 80.0
left_mm, right_mm = 20.0, 5.0
bottom_mm, top_mm = 12.0, 5.0
fig_w = (left_mm + axes_width_mm + right_mm)
fig_h = (bottom_mm + axes_height_mm + top_mm)
spine_linewidth = 2.0
tick_width = 2.0
tick_direction = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize = 8.0