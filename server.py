import copy
import numpy as np
from loaddata import generate_dataloader
from metric import server_meters
from sklearn.cluster import KMeans

class Server():
    def __init__(self,args):
        self.time=[]
        self.args=args
        self.add=self.args.add_layer
        self.structure= {0:{}}
        self.num_layers=1
        self.eval_loss_queue = [999]*args.acc_queue_length
        self.eval_acc=[]
        self.test_acc = []
        self.mean_test_acc=0
        self.meters=server_meters(args)
        self.process_mask=[0,True]*self.args.num_client
    def init_model(self, model):
        self.structure[0][0] = [np.arange(self.args.num_client), copy.deepcopy(model)]
        self.keys = self.structure[0][0][1].state_dict().keys()
    def model_distribute(self,epoch,clients):
        self.epoch=epoch
        for client in clients:
            client.load_model(self.structure)
    def train(self,clients):
        for client in clients:
            client.train()

    def eval(self, clients):
        acc = []
        loss=[]
        for client in clients:
            eval_loader = generate_dataloader(client.args.data_name, client.data['eval'], batch_size=32)
            meter = client.eval(eval_loader,'eval')
            acc.append(meter.accuracy_score)
            loss.append(meter.new_loss)
        self.eval_loss_queue.pop(0)
        self.eval_loss_queue.append(np.sum(np.array(loss)) / len(clients))
        self.eval_acc.append(np.array(acc))
        print('eval_accuracy:', acc)
        print('overall_eval_accuracy:', np.sum(np.array(acc)) / self.args.num_client)
        print('eval_loss:', loss)
        print('overall_eval_loss:', np.sum(np.array(loss)) / self.args.num_client)
        self.meters.update_optimal(copy.deepcopy(self.structure), np.sum(np.array(loss)) / self.args.num_client,
                                   copy.deepcopy(self.epoch), copy.deepcopy(self.mean_test_acc),
                                   copy.deepcopy(self.test_acc))
        self.meters.update_structure_list(copy.deepcopy(self.structure))
        self.meters.update(self.epoch, acc, np.sum(np.array(acc)) / self.args.num_client, loss,
                           np.sum(np.array(loss)) / self.args.num_client)
        if self.add and self.epoch >= self.args.init_epoch:
            return self.add_layer(clients)
        return False
    def test(self,clients):
        self.test_acc = []
        for client in clients:
            test_loader = generate_dataloader(client.args.data_name, client.data['test'], batch_size=32)
            meter = client.eval(test_loader,'test')
            self.test_acc.append(meter.accuracy_score)
        print('test_accuracy:',  self.test_acc)
        self.mean_test_acc=np.sum(np.array( self.test_acc)) / self.args.num_client
        print('overall_test_accuracy:',  self.mean_test_acc)

    def add_layer(self,clients):
        redistribute = False
        if np.std(self.eval_loss_queue)<self.args.std_threshold:
            redistribute=True
            print('eval_loss_queue:',self.eval_loss_queue)
            self.eval_loss_queue = [999] * self.args.acc_queue_length

            if self.num_layers>1:
                for client in clients:
                    client.process_mask.append(None)
                    client.process_mask.append(len(client.eval_loss))
                    indexes = [index for index, value in enumerate(client.eval_loss) if value == 999]
                    index1 = indexes[-1]
                    a = client.eval_loss[index1 + 1:]
                    client.eval_loss.append(999)
                    if len(a) > self.args.acc_queue_length:
                        a = a[-self.args.acc_queue_length:]
                    for i,item in enumerate(reversed(client.process_mask)):
                        if item==True:
                            index2=client.process_mask[len(client.process_mask)-1-i-1]
                            index2_1=client.process_mask[len(client.process_mask)-1-i+1]
                            break
                    b = client.eval_loss[index2 + 1:index2_1]
                    if len(b) > self.args.acc_queue_length:
                        b = b[-self.args.acc_queue_length:]
                    surpass = (sum(b) / len(b) - sum(a) / len(a) > 0)
                    if surpass:
                        client.process_mask[-2]=True
                    else:
                        client.process_mask[-2]=False
                        for level,value in self.structure[self.num_layers - 1].items():
                            value[0]=value[0][value[0]!=client.id]
                    self.process_mask[client.id]=client.process_mask
                new_structure={}
                i=0
                for key,value in self.structure[self.num_layers - 1].items():
                    if self.structure[self.num_layers - 1][key][0].size==0:
                        continue
                    else:
                        new_structure[i]=self.structure[self.num_layers - 1][key]
                        i=i+1
                if len(new_structure)==0:
                    del self.structure[self.num_layers - 1]
                else:
                    self.structure[self.num_layers-1]=new_structure
            else:
                for client in clients:
                    client.process_mask.append(len(client.eval_loss))
                    client.eval_loss.append(999)
                    self.process_mask[client.id] = client.process_mask

            if self.num_layers >= self.args.structure_length:
                for i in range(5):
                    self.model_distribute(self.epoch,clients)
                    self.eval(clients)
                    self.test(clients)
                    self.meters.update_final(copy.deepcopy(self.structure),
                                               copy.deepcopy(self.epoch), copy.deepcopy(self.mean_test_acc),
                                               copy.deepcopy(self.test_acc))
                return redistribute
            self.meters.update_mask(self.process_mask)
            self.num_layers += 1
            self.structure[self.num_layers - 1] = {}
            self.structure[self.num_layers - 1][0] = [[], copy.deepcopy(self.structure[0][0][1])]
            for client in clients:
                client.stop_layer = False
                self.structure[self.num_layers - 1][0][0].append(client.id)
            self.structure[self.num_layers-1][0][0]=np.array(self.structure[self.num_layers-1][0][0])
        return redistribute

    def aggregate_id(self,clients):
        clusters=[self.args.cluster_num[self.num_layers-1]]
        if not self.args.add_layer and self.epoch<3:
            clusters=[1]
        layers = [list(self.structure.keys())[-1]]
        self.diff_sample=[]
        aggregate_id=[]
        for client in clients:
            self.diff_sample.append(client.diff_sample)
            aggregate_id.append(client.id)
        aggregate_id=np.array(aggregate_id)
        for layer,cluster_num in zip(layers,clusters):
            kmeans = KMeans(n_clusters=cluster_num, init='random', n_init=10, max_iter=300, tol=1e-4)
            a = []
            for sample in self.diff_sample:
                a.append(sample[layer])
            if self.num_layers>=self.args.structure_length:
                kmeans.n_clusters = len(a)
            kmeans.fit(a)
            labels = kmeans.labels_
            self.structure[layer] = {}
            for i in set(labels):
                self.structure[layer][i] = [aggregate_id[np.squeeze(np.argwhere(labels == i), axis=-1)], None]

        for layer in list(self.structure.keys()):
            group_ids = []
            groups = []
            for group_id, group in self.structure[layer].items():
                group_ids.append(str(group_id))
                groups.append(group[0].tolist())
            print(group_ids)
            print(groups)

    def aggregate_model(self,clients):
        def weight_sum_param(models, weights):
            new_model = copy.deepcopy(models[0])
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            new_param = {}
            for key in self.keys:
                for (model, weight) in zip(models, weights):
                    if key not in new_param:
                        new_param[key] = model.state_dict()[key] * weight
                    else:
                        new_param[key] = new_param[key] + model.state_dict()[key] * weight
            new_model.load_state_dict(new_param)
            return new_model
        layers = [list(self.structure.keys())[-1]]
        for layer in layers:
            for group_id, group in self.structure[layer].items():
                model = []
                weight = []
                label = group[0]
                for client in clients:
                    if client.id in label:
                        model.append(client.structure[layer].cpu())
                        weight.append(client.num_data)
                new_model = weight_sum_param(model, weight)
                self.structure[layer][group_id][1] = new_model

    def aggregate(self, clients):
        self.aggregate_id(clients)
        self.aggregate_model(clients)


