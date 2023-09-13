# import argparse
import os

import numpy as np
import torch
# from DataSetMaster import dataset
# from MainLauncher import path_operator, Queue

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default='MNISTUSPS')  # MNISTUSPS ReverseMNIST Office HAR MTFL
# parser.add_argument("--batch_size", default=512, type=int)  # MNISTUSPS ReverseMNIST Office HAR MTFL
# args = parser.parse_args()
# args.MnistTrain = True


# def get_dist():
#     train_loader, test_loader, class_num = dataset.get_dataloader(
#         dataset=args.dataset, batch_size=args.batch_size, path=path_operator.get_dataset_path, args=args)
#
#     loader = test_loader
#     num_data = [10]
#     with torch.no_grad():
#         dist_list = [[] for i in range(len(num_data))]
#         for j, data_t in enumerate(loader, 0):
#             # get all inputs
#             inputs_t, _, labels_t, _ = data_t
#             inputs_t, labels_t = inputs_t.cuda(), labels_t.cuda()
#             for i in range(len(inputs_t)):
#                 if i % 1000 == 0:
#                     print(i)
#                 aa = torch.mul(inputs_t - inputs_t[i], inputs_t - inputs_t[i])
#                 # dist = torch.sqrt(torch.sum(aa, dim=(2, 3)))
#                 # dist_m = dist[:, 0]
#                 # print(aa.shape)
#                 dist_m = torch.sqrt(torch.sum(aa, dim=tuple(torch.arange(1, len(aa.shape)))))
#                 dist_m[i] = 1000
#                 sorted_dist = np.sort(dist_m.cpu().numpy())
#                 for jj in range(len(num_data)):
#                     dist_list[jj].append(sorted_dist[num_data[jj]])
#         for ii in range(len(num_data)):
#             np.savetxt('../' + args.dataset + '_' + str(num_data[ii]) + 'th_neighbor.txt', np.array(dist_list[ii]))
from _Utils.DirectoryOperator import DirectoryOperator


def get_dist_release(loader, dist_path):
    if not os.path.exists(dist_path):
        # loader = test_loader
        num_data = [10]
        with torch.no_grad():
            dist_list = [[] for i in range(len(num_data))]
            for j, data_t in enumerate(loader, 0):
                # get all inputs
                fea0, fea1, class_labels0, class_labels1, mask, is_pair, idx = data_t
                inputs_t = fea0.cuda()
                # inputs_t = torch.cat([fea0,fea1]).cuda()
                # labels_t = torch.cat([class_labels0,class_labels1]).cuda()
                # inputs_t, _, labels_t, _ = data_t
                # inputs_t, labels_t = inputs_t.cuda(), labels_t.cuda()
                for i in range(len(inputs_t)):
                    if i % 1000 == 0:
                        print(i)
                    aa = torch.mul(inputs_t - inputs_t[i], inputs_t - inputs_t[i])
                    # dist = torch.sqrt(torch.sum(aa, dim=(2, 3)))
                    # dist_m = dist[:, 0]
                    # print(aa.shape)
                    dist_m = torch.sqrt(torch.sum(aa, dim=tuple(torch.arange(1, len(aa.shape)))))
                    dist_m[i] = 1000
                    sorted_dist = np.sort(dist_m.cpu().numpy())
                    for jj in range(len(num_data)):
                        dist_list[jj].append(sorted_dist[num_data[jj]])
                inputs_t = fea1.cuda()
                for i in range(len(inputs_t)):
                    if i % 1000 == 0:
                        print(i)
                    aa = torch.mul(inputs_t - inputs_t[i], inputs_t - inputs_t[i])
                    # dist = torch.sqrt(torch.sum(aa, dim=(2, 3)))
                    # dist_m = dist[:, 0]
                    # print(aa.shape)
                    dist_m = torch.sqrt(torch.sum(aa, dim=tuple(torch.arange(1, len(aa.shape)))))
                    dist_m[i] = 1000
                    sorted_dist = np.sort(dist_m.cpu().numpy())
                    for jj in range(len(num_data)):
                        dist_list[jj].append(sorted_dist[num_data[jj]])
            for ii in range(len(num_data)):
                DirectoryOperator(dist_path).make_fold()
                np.savetxt(dist_path, np.array(dist_list[ii]))

    dist = torch.from_numpy(
        np.loadtxt(
            dist_path
        ).astype(np.float32)
    )
    return dist


# if __name__ == '__main__':
#     get_dist()
#     # Queue.dequeue()
