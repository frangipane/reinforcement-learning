#!/bin/bash

# CONVERT video, e.g. mp4 to gif, using ffmpeg

# See https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality

# MODIFY THESE VARIABLES
src="input.flv"
dest="output.gif"
palette="/tmp/palette.png"


ffmpeg -i $src -vf palettegen -y $palette
ffmpeg -i $src -i $palette -lavfi paletteuse -y $dest
