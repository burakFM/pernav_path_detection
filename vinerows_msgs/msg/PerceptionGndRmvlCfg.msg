# Custom message for perception auto configuration

std_msgs/Header header
float64 nominal_row_spacing_m        # nominal row spacing to use in VSN
float64 min_height_diff_m            # height disparity to use for vinerow detection
float64 non_ground_threshold_m       # height above which all points can be kept
float64 img_dft_quality              # quality measure for above values
