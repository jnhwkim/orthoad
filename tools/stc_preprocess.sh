#!/bin/bash

source ./script/setup.sh STC

mkdir -p ./archive

unzip ${DATA}/training.zip -d ./archive/
unzip ${DATA}/Testing/frames_part1.zip -d ./archive/
unzip ${DATA}/Testing/frames_part2.zip -d ./archive/
unzip ${DATA}/Testing/test_frame_mask.zip -d ./archive/
unzip ${DATA}/Testing/test_pixel_mask.zip -d ./archive/

mkdir -p ./converted/test
mv ./archive/frames_part1/* ./converted/test/
mv ./archive/frames_part2/* ./converted/test/

python ./tools/stc_vid2frames.py
