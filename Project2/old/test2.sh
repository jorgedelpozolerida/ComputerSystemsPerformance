#!/bin/bash

while true; do
    timestamp=$(date +%s.%N)
    power=$(powerstat -d 0 | awk '{print $1}')
    memory=$(free -m | awk '/Mem:/ {print $3}')
    echo "$timestamp,$power W,$memory MiB"
    sleep 0.1
done



