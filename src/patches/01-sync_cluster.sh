#!/usr/bin/bash

rsync -azP --exclude=.git --exclude=`git -C . ls-files --exclude-standard -oi --directory` . euler:/cluster/work/sachan/vilem/mt-metric-estimation/