#!/bin/bash

echo "timestamp;power;memory"
while true; do
  power=$(sensors | grep "Package id" | awk '{print $4}' | sed 's/+//g')
  timestamp=$(date +%Y-%m-%d_%H-%M-%S.%N)
  memory=$(free | awk 'FNR == 2 {print $3/1024}')
  echo "$timestamp;$power;$memory"
  sleep 0.1
done