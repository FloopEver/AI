#!/bin/bash
rm result.txt
for (( round=1; round<=1000; round+=1 )) 
do
echo $round >> result.txt
sh build.sh
done