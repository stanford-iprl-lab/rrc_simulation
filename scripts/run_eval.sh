#!/bin/bash

for i in {1..4}
do
    echo "running evaluate on output dir output/run-${i}"
    rrc_evaluate ./output/rl/run-$i
done
