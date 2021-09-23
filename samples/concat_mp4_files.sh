#!/bin/bash
# usage: ./concat_mp4_files.sh <directory containing mp4 files>
for f in $1/*.mp4 ;
    do echo file \'$f\' >> list.txt;
done && \
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4 && rm list.txt
