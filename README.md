This repository contains the implementation for the paper "Multi-Level Additive Modeling for Fine-grained Non-IID Federated Learning." 

Please first place the original cifar100 and tinyimagent dataset in ./dump_items dataset.

Then run the sh files in shs/generatedata to produce the data partition setting you desired. For example, cifar0.1.sh means generate dirichlet distribution dataset with parameter 0.1, i.e., bash shs/generatedata/cifar0.1.sh.

Finally run the corresponding sh file in shs/main to start training. For example, cifar0.1.sh means run under dirichlet distribution with parameter 0.1, i.e., bash shs/main/cifar0.1.sh. 
