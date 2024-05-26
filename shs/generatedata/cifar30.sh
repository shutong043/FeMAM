#!/bin/bash

# åºç¡è·¯å¾
base_dir="/home/shutoche/PycharmProjects/FeMAM_master"

# å¾ªç¯æ¬¡æ°
loop1=1
loop2=1

# å®ä¹åæ°åè¡¨
num_client=("50")
mark=("new_design")
num_class_per_cluster=("[30,30,30,30,30]")
use_class_partition=("True")
hierarchical_data=("False")
acc_queue_length=("50")
std_threshold=("1")
structure_length=("5")
data_name=("cifar100")
global_epoch=("700")
device=("0")
for ((i=0; i<$loop1; i++)); do
for ((j=0; j<$loop2; j++)); do
    python3 $base_dir/generatedata.py --mark ${mark[$i]} --num-client ${num_client[$i]} --use-class-partition ${use_class_partition[$i]} --hierarchical-data ${hierarchical_data[$i]} --data-name ${data_name[$i]} --global-epoch ${global_epoch[$i]} --num-class-per-cluster ${num_class_per_cluster[$i]} --device ${device[$i]} --acc-queue-length ${acc_queue_length[$i]} --std-threshold ${std_threshold[$i]} --structure-length ${structure_length[$i]}  > /dev/null
done
done



