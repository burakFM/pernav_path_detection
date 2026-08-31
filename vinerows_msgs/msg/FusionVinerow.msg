std_msgs/Header header

geometry_msgs/Point center
geometry_msgs/Vector3 direction
float64 width
float64[8] variance     # the variance is the variance of the center [3] + the direction [3] + width (in this order)
bool[7] vine_present
