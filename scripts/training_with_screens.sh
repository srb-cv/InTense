#!/bin/bash
# pass a txt file containing a set of experiments
if [ $# -eq 1 ]
    then screen_name=$1
elif [ $# -eq 2 ]
    then screen_name=$2
elif [ $# -eq 3 ]
    then free_gpu_id=$3
    screen_name=$2
    echo $screen_name
else
    echo "either one or two parameters should be passed"
    exit 0
fi  
count=0
screen -S $screen_name -d -m
while read line
do
    count=`expr $count + 1`
    # create now window using screen command
    screen -S $screen_name -X screen $count
    screen -S $screen_name -p $count -X stuff "cd /home/varshney/magus/projects/intense
    "
    screen -S $screen_name -p $count -X stuff "conda activate py10t20
    "
    screen -S $screen_name -p $count -X stuff "$line
    "
done < scripts/$1