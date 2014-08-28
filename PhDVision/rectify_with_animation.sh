#!/bin/sh

if [ $# != 6 ]; then
    echo "Syntax: [Video Filename] [First Frame #] [Last Frame #] [Frames to Skip (1 for no skip)] [Output Images Prefix] [Print Original Images (Y/N)]"
    exit 1
fi

python rectify_images.py $1 $2 $3 $4 $5 $6
if [ `ls ${5}_rectified_[0-9]*.jpg | wc -l` -eq 0 ]; then
    rm -f ${5}_orig_[0-9]*.jpg
    exit 1
fi

delay=`echo ${4} | awk '{print 3.5*$1}'`

convert -delay ${delay} -loop 0 ${5}_rectified_[0-9]*.jpg ${5}_rectified_animation.gif

if [ `echo ${6} | awk '{print tolower($0)}'` == "y" ]; then
    convert -delay ${delay} -loop 0 ${5}_orig_[0-9]*.jpg ${5}_orig_animation.gif
    convert ${5}_orig_animation.gif'[0]' -coalesce \( ${5}_rectified_animation.gif'[0]' -coalesce \) +append -channel A -evaluate set 0 +channel ${5}_orig_animation.gif -coalesce -delete 0 null: \( ${5}_rectified_animation.gif -coalesce \) -gravity East  -layers Composite    ${5}_composite_animation.gif
fi

rm ${5}_orig_[0-9]*.jpg
rm ${5}_rectified_[0-9]*.jpg

exit 0
