from loaddata import *
from server import Server
import time
from metric import save_meters
from clients import client
import argparse
import ast
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='./dump_items', type=str, help='the path for generated non-IID settings and experiment results,please put original cifar100 and tinyimagenet dataset in this path.')
parser.add_argument('--mark', default='FeMAM', type=str, help='Rename this variable to distinguish between different experiment outcomes')
parser.add_argument('--data-name', default='cifar100', type=str, help='choose the data between tinyimagenet and cifar100')
parser.add_argument('--local-epoch', default=2, type=int, help='number of local epoch, default is 2')
parser.add_argument('--model_name', default='resnet', type=str, help='use resnet as the model')
parser.add_argument('--hierarchical-data', default=False, type=str2bool, help='whether use fine-grained data partition')
parser.add_argument('--hierarchical-dis', default=6, type=int, help='choose detailed generate settings for fine-grained data partition')
parser.add_argument('--num-client', default=50, type=int, help='choose the number of clients')
parser.add_argument('--use-class-partition', default=False, type=str2bool, help='whether to partition data by class')
parser.add_argument('--num-class-per-cluster', default='[10]', type=str, help='choose the number of class per cluster in dataset,for example,[3,3,3,3,3] means five clusters, three class per cluster')
parser.add_argument('--cluster-num', default='[1,5,5,5,5,5,5,5]', type=str, help='number of clusters for each level')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--global-epoch', default=700, type=int, help='number of global epoch')
parser.add_argument('--init-epoch', default=400, type=int, help='number of rounds for the first level')
parser.add_argument('--batch-size', default=32, type=int, help='number of batch size')
parser.add_argument('--cluster-keys', default=2, type=int, help='model parameters for cluster, for example, 2 means use the parameter of the last 2 layer as clustering samples')
parser.add_argument('--use-diff', default=False, type=str2bool, help='whether use gradient as clustering samples')
parser.add_argument('--add-layer', default=True, type=str2bool, help='whether add layer, if not FeMAM reduced to one level')
parser.add_argument('--acc-queue-length', default=50, type=int, help='number of communication rounds to judge convergence')
parser.add_argument('--std-threshold', default=1, type=float, help='the threshold for convergence')
parser.add_argument('--structure-length', default=5, type=float, help='The number of level for FeMAM')
parser.add_argument('--alpha', default=0.1, type=float, help='choose the paramter of dirichlet distribution for data')
parser.add_argument('--device', default=0, type=int, help='gpu device number')
if __name__=='__main__':
    args = parser.parse_args()
    args.cluster_num=ast.literal_eval(args.cluster_num)
    args.num_class_per_cluster=ast.literal_eval(args.num_class_per_cluster)
    data,client_by_class, class_by_client,gd_cluster=load_data(args)
    args.client_by_class=client_by_class
    args.class_by_client=class_by_client
    args.gd_cluster=gd_cluster

    model=load_model(args,data['client' + str(0)]['train'])
    clients=[]
    for i in range(args.num_client):
        clients.append(client(i,args,data['client' + str(i)],model))
    server=Server(args)
    server.init_model(model)
    server.time.append(time.time())
    for epoch in range(args.global_epoch):
        print('epoch:', epoch)

        server.model_distribute(epoch,clients)
        add_layer=server.eval(clients)
        if add_layer:
            server.model_distribute(epoch, clients)
        server.train(clients)
        server.aggregate(clients)
        server.model_distribute(epoch,clients)
        server.test(clients)
    server.meters.update_clients(clients)
    save_meters(args,server.meters)
    server.time.append(time.time())



