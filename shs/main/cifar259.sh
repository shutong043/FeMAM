#!/bin/bash

# 基础路径
base_dir="/home/shutoche/PycharmProjects/FeMAM_master"
# 循环次数
loop1=1
loop2=1

# 定义参数列表
mark=("new_design_259")
hierarchical_dis=("0")
use_class_partition=("True")
hierarchical_data=("True")
data_name=("cifar100")
global_epoch=("700")
num_client=("50")
acc_queue_length=("50")
std_threshold=("1")
structure_length=("5")
device=("0")
for ((i=0; i<$loop1; i++)); do
for ((j=0; j<$loop2; j++)); do
    python3 $base_dir/main.py --mark ${mark[$i]} --num-client ${num_client[$i]} --use-class-partition ${use_class_partition[$i]} --hierarchical-data ${hierarchical_data[$i]} --data-name ${data_name[$i]} --global-epoch ${global_epoch[$i]} --device ${device[$i]} --acc-queue-length ${acc_queue_length[$i]} --hierarchical-dis ${hierarchical_dis[$i]} --std-threshold ${std_threshold[$i]} --structure-length ${structure_length[$i]}  > /dev/null
done
done

