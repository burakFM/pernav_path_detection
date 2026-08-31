std_msgs/Header header

# Latitude (degrees). Positive is north of equator; negative is south.
float64 latitude

# Longitude (degrees). Positive is east of prime meridian, negative west.
float64 longitude

# Altitude (meters). Positive is above reference (e.g., sea level).
float64 altitude

# Direction (degrees from north)
float64 heading

# Valid message
bool gps_valid

# Vehicle pose wrt local enu (meters)
float32 local_x_enu
float32 local_y_enu
