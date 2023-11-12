import os
import warnings
from DataSetMaster import dataset_master
import Net
import argparse
import numpy as np
import torch
import random
from _MainLauncher import path_operator, Queue
from _Utils.ConfigOperator import ConfigOperator
from figures.ScatterMaster import visualize_image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset", default='NoisyMNIST30000')
    # 'MNISTUSPS', 'NoisyMNIST30000', '2view-caltech101-8677sample', 'cub_googlenet_doc2vec_c10'
    # 'NoisyMNIST',
    #  'Caltech101-all', 'Caltech101-20',
    # 'Reuters_dim10', 'Scene15',
    # ,
    # ,

    # # Vector (MNIST-USPS, Scene15, ...)
    parser.add_argument("--normalize_type",
                        default='dim_wise')  # dim_wise, sample_wise, rescale_dim_wise, rescale_sample_wise, None
    parser.add_argument('--withTanh', default=0, type=int, help='bool')
    parser.add_argument("--DimensionalityReduction", default=0, type=int)
    # # Image (MNISTUSPS)
    parser.add_argument("--MnistTrain", default=1, type=int, help='bool')  # dataset==MNISTUSPS
    parser.add_argument("--MaxSampleNum", default=5000, type=int)  # dataset==MNISTUSPS
    parser.add_argument("--DataSource", default='')  # dataset==MNISTUSPS
    parser.add_argument('--Image01NoTanh', default=1, type=int, help='bool')
    parser.add_argument('--Image01AndTanh', action='store_true')

    # Network
    parser.add_argument('--UseCovNet', default=1, type=int, help='bool')
    parser.add_argument('--GroupWiseLayer', default='000111')
    parser.add_argument('--BatchNormType', default='11001')
    parser.add_argument('--ElActivationType',
                        default='Normalize')  # 'Normalize', 'BnNormalize', 'BnReNormalize', 'None', 'BnRe'
    parser.add_argument('--Dropout', default=0.2, type=float)
    parser.add_argument('--McDecoder', action='store_true')
    parser.add_argument("--representation_dim", help="", default=0, type=int)
    parser.add_argument('--ActivationType', default='None')  # None Sigmoid Tanh

    # Training
    parser.add_argument("--batch_size", help="batch size", default=256, type=int)
    parser.add_argument("--train_epoch", help="training epochs", default=150, type=int)
    parser.add_argument("--seed", help="random seed", default=2023, type=int)
    parser.add_argument("--resume", default='')
    parser.add_argument('--CodeTest', action='store_true')
    parser.add_argument('--RunInManuel', action='store_true')
    # # Optimizer
    parser.add_argument("--LearnRate", help="", default=1e-3, type=float)
    parser.add_argument("--LearnRateDecayType", default='None')  # Exp, Cosine
    parser.add_argument("--LearnRateWarm", default=0, type=int)
    parser.add_argument("--WeightDecay", default=0, type=float)
    parser.add_argument("--betas_a", help="", default=0.9, type=float)
    parser.add_argument("--betas_v", help="", default=0.999, type=float)
    # # Loss
    parser.add_argument("--WarmAll", default=50, type=int)
    parser.add_argument("--WarmContras", default=0, type=int)

    parser.add_argument("--Reconstruction", help="", default=1.0, type=float)
    parser.add_argument("--ReconstructionEpoch", default=999, type=int)

    parser.add_argument("--SoftAssignmentTemperatureBalance", help="", default=0.1, type=float)
    parser.add_argument("--InfoBalanceLoss", help="", default=0.04, type=float)
    parser.add_argument("--InfoFairLoss", help="", default=0.20, type=float)

    parser.add_argument("--SoftAssignmentTemperatureHot", help="", default=0.1, type=float)
    parser.add_argument("--OneHot", help="", default=0.04, type=float)

    parser.add_argument("--SoftAssignmentTemperatureSelfCons", help="", default=0.2, type=float)
    parser.add_argument("--loss_self_cons", help="", default=0.0, type=float)
    parser.add_argument("--SelfConsLossType", default='Assignment')  # Assignment LogAssignment

    parser.add_argument("--Decenc", help="", default=0.0, type=float)

    parser.add_argument("--noise_type", default='Drop')  # None, Drop, Add
    parser.add_argument("--noise_weight", default=0.1, type=float)

    parser.add_argument("--loss_imsat", default=0, type=float)
    parser.add_argument('--prop_eps', type=float, help='imsat', default=0.25)

    parser.add_argument("--loss_correspondence", help="", default=0.0, type=float)
    parser.add_argument("--loss_sim_cluster", help="", default=0.0, type=float)
    parser.add_argument("--loss_sim_instance", help="", default=0.0, type=float)  # 0.05
    parser.add_argument("--loss_sim_contras", help="", default=0.02, type=float)  # 0.01
    parser.add_argument("--loss_sim_contras_temperature", help="", default=0.2, type=float)  # 0.2
    parser.add_argument("--loss_sim_contras_clu", help="", default=0.0, type=float)  #
    parser.add_argument("--loss_sim_contras_gau", help="", default=0.0, type=float)  # 0.0
    parser.add_argument("--loss_sim_distrib", help="", default=0.0, type=float)
    # # MC
    parser.add_argument('--aligned_prop', type=float, default=1.00, help='1-unaligned rate in our paper')  
    parser.add_argument('--complete_prop', type=float, default=1.00, help='1-missing rate in our paper') 
    # if aligned_prop<1 and complete_prop <1: the exact proportion of complete, unaligned, and missing samples are aligned_prop, (1-aligned_prop)*complete_prop, and (1-aligned_prop)*(1-complete_prop), respectively.
    
    # Evaluate
    parser.add_argument("--VisualFreq", help="", default=10, type=int)
    parser.add_argument("--VisualRandom", help="", default=0, type=int)
    parser.add_argument("--EvalMulti", default=1, type=int, help='bool')
    parser.add_argument("--Rev", default=0, type=int, help='bool')
    parser.add_argument("--EvalMean", default=0, type=int, help='bool')
    parser.add_argument("--EvalSingel0", default=0, type=int, help='bool')
    parser.add_argument("--EvalSingel1", default=0, type=int, help='bool')
    parser.add_argument("--EvalOriMean", default=0, type=int, help='bool')
    parser.add_argument("--EvalOriScoreMean", default=0, type=int, help='bool')
    parser.add_argument("--EvalOriPredMean", default=0, type=int, help='bool')  # Stable and zero cast
    parser.add_argument('--DrawMax', type=int, default=1000000000)
    parser.add_argument("--DrawTSNE", action='store_true')
    parser.add_argument("--DrawUmap", action='store_true')
    parser.add_argument("--reFill", default='KnnMean')  # Copy, KnnMean, Center
    parser.add_argument("--reAlign", default='KnnMean')  # Copy, KnnMean, KnnMapMean, Ignore
    parser.add_argument("--reAlignK", default=1, type=int)

    parser.add_argument("--EvalCla", default=0, type=int, help='bool')

    parser.add_argument("--Desc0", default='')
    parser.add_argument("--QuickConfig", default='')
    parser.add_argument("--Desc", default='')
    parser.add_argument("--ShowReconstruct", action='store_true')
    parser.add_argument("--ShowClustering", action='store_true')
    parser.add_argument("--SingleView", default=-1, type=int, )

    args = parser.parse_args()

    if args.RunInManuel:
        warnings.warn('RunInManuel')
        # args.dataset = 'NoisyMNIST30000'
        # args.dataset = 'MNISTUSPS'
        # args.dataset = 'cub_googlenet_doc2vec_c10'
        # args.train_epoch = 1
        args.dataset = '2view-caltech101-8677sample'
        # args.dataset = 'Scene15'
        # args.GroupWiseLayer = '111111'
        # args.withTanh = False
        # args.normalize_type = 'None'
        # args.MaxSampleNum = 5000
        # args.Image01NoTanh = True
        # args.Image01AndTanh = True
        # args.train_epoch = 1
        # args.loss_imsat = 0.01
        # args.loss_self_cons = 0.01
        # args.WarmAll = 0
        # args.InfoBalanceLoss = 0
        # args.InfoFairLoss = 0
        # args.OneHot = 0
        # args.VisualFreq = 1
        # args.batch_size = 128
        # args.ActivationType = "None"
        # args.QuickConfig = 'QuickConfig/Unsup.yaml'
        # args.reFill = 'KnnMean'
        # args.reAlign = 'KnnMapMean'
        # args.reFill = 'KnnMapMean'
        # args.reFill = 'Center'
        # args.reAlign = 'Copy'
        # args.aligned_prop = 0.5
        # args.complete_prop = 1
        # args.SingleView = 0
        # args.UseCovNet = True
        # args.loss_sim_contras = 0.02
        # args.loss_sim_contras_clu = 0.5
        # args.loss_sim_contras_gau = 1
        # args.EvalCla = 1
        # args.EvalMean = 1
        # args.EvalSingel0 = 1
        # args.EvalOriMean = 1
        # args.normalize_type = 'sample_wise'
        # args.withTanh = 0
        # args.EvalOriScoreMean = 1
        # args.EvalOriPredMean = 1
        # args.CodeTest = True
        # args.ShowReconstruct = True
        # args.QuickConfig = 'X50C50'
        # args.QuickConfig = 'A100'
        # args.seed = 2022
        args.QuickConfig = 'C100'
        # args.DrawMax = 3000
        args.seed = 1215
        args.ShowClustering = False
        # args.resume = '/xlearning/pengxin/Temp/Python/Checkpoints/Epoch005.checkpoint'
        args.resume = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22/ --QuickConfig {} --dataset 2view-caltech101-8677sample --seed {}/Checkpoints/Epoch149.checkpoint'.format(args.QuickConfig,args.seed)
        # args.QuickConfig = 'C100'
        # # args.DrawMax = 3000
        # args.seed = 1999
        # args.ShowClustering = True
        # # args.resume = '/xlearning/pengxin/Temp/Python/Checkpoints/Epoch005.checkpoint'
        # args.resume = 'D:/VirtualMachine/Codes/230904/SMAIL_RunSet_Visual/ --QuickConfig C100 --VisualFreq 5 --VisualRandom 1 --dataset NoisyMNIST30000 --seed 1999 --train_epoch 100/Checkpoints/Epoch099.checkpoint'
        # args.resume = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22/ --QuickConfig A100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed 9116/Checkpoints/Epoch120.checkpoint'
        # args.resume = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22/ --QuickConfig A100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed 9116/Checkpoints/Epoch149.checkpoint'
    if args.QuickConfig:
        qcs = args.QuickConfig.split(',')
        for i in range(len(qcs)):
            qcs[i] = 'QuickConfig/' + qcs[i] + '.yaml'
            print(qcs[i])
        for k, val in ConfigOperator(cuda=None, yaml_list=qcs).config.items():
            args.__setattr__(k, val)
    if args.aligned_prop == 1:
        args.complete_prop = 1
    if args.SingleView != -1:
        if args.loss_sim_contras or args.InfoFairLoss:
            warnings.warn('We use SingleView while you set args.loss_sim_contras or args.InfoFairLoss.'
                          'Thus we reset them automatically')
        args.loss_sim_contras = 0
        args.InfoFairLoss = 0
        args.EvalMulti = 0
    if args.UseCovNet:
        if 'MNIST' in args.dataset:
            net = Net.NetCov
            args.BatchNormType = "00000"
            args.Dropout = 0
            args.batch_size = 64
            args.normalize_type = 'sample_wise'
            args.Rev = 1
        else:
            net = Net.Net
            args.UseCovNet = False
            warnings.warn('We use FFN while you set args.UseCovNet == True, '
                          'due to that the dataset {} cannot use CovNet'.format(args.dataset))
    else:
        net = Net.Net

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print('=======Arguments=======')
    print(path_operator.python_path)
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))

    if args.train_epoch >= 0:

        train_loader, test_loader, class_num = dataset_master.get_dataloader_mouxin(
            args=args, neg_prop=0, aligned_prop=args.aligned_prop, complete_prop=args.complete_prop,
            is_noise=False)
        # visualize_image(test_loader.dataset.data[0][:100].reshape([10,10,28,28]), show=True)
        # visualize_image(test_loader.dataset.datasets[0].data.numpy()[500:600].reshape([10,10,28,28]), show=True)
        if args.dataset == 'MNISTUSPS':
            in_dims = [np.product(d.data.shape[1:]) for d in test_loader.dataset.datasets]
        else:
            in_dims = [d.shape[1] for d in test_loader.dataset.data]
        print('in_dims == {}'.format(in_dims))

        # if args.dataset == 'Scene15':
        #     args.ActivationType = 'None'
        #     args.normalize_type = 'dim_wise'
        #     args.withTanh = 0
        #     warnings.warn('You are force editing the preprocessing of Scene15')

        # if 'MNIST' in args.dataset
        if args.aligned_prop == 0:
            if args.reAlign == 'KnnMapMean':
                args.reAlign = 'KnnMean'
                warnings.warn("We set args.reAlign = 'KnnMean' due to KnnMapMean not viabel for args.aligned_prop == 0")
            if args.reFill == 'KnnMapMean':
                args.reFill = 'KnnMean'
                warnings.warn("We set args.reFill = 'KnnMean' due to KnnMapMean not viabel for args.aligned_prop == 0")
        if in_dims[0] != in_dims[1]:
            if args.UseCovNet:
                crit_dim = [2, 3]
            else:
                crit_dim = [0, 5]
            for c_dim in crit_dim:
                if args.GroupWiseLayer[c_dim] == '0':
                    args.GroupWiseLayer = args.GroupWiseLayer[:c_dim] + '1' + args.GroupWiseLayer[c_dim + 1:]
                    warnings.warn("args.GroupWiseLayer[c_dim] == '0' and in_dims[0] != in_dims[1] "
                                  "=> args.GroupWiseLayer == {}".format(args.GroupWiseLayer))
        net = net(class_num=class_num,
                  in_dims=in_dims,
                  group_num=len(in_dims),
                  args=args
                  ).cuda()
        print(net)
        # from torchsummary import summary
        # summary(net, [20], batch_size=args.batch_size)
        net.run(epochs=args.train_epoch,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                args=args)
    print('Ending...')
    Queue.dequeue()
