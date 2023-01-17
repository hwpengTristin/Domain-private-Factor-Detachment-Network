"""
Part of our code is based on Github's open source project (https://github.com/davidsandberg/facenet).

We present an efficient Disentangled Spectrum Variations Networks (DSVNs) for NIR-VIS HFR.
Two key strategies are introduced to the DSVNs for disentangling spectrum variations between
two domains: Spectrum-adversarial Discriminative Feature Learning (SaDFL) and Step-wise Spectrum
Orthogonal Decomposition (SSOD). The SaDFL consists of Identity-Discriminative subnetwork (IDNet)
and Auxiliary Spectrum Adversarial subnetwork (ASANet). Both IDNet and ASANet can jointly enhance
the domain-invariant feature representations via an adversarial learning. The SSOD is built by
stacking multiple modularized mirco-block DSV, and thereby enjoys the benefits of disentangling
spectrum variation step by step.Nir_Vis_reconstruct_images

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc

import facenet
import time
import scipy.io
import torch
import cv2
import numpy as np

import scipy.io as sio

import os

from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def Nir_Vis_evaluate_Rank_1(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160):

    try:
        'DATA Path'
        currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
        fileNameDict = {'nirTrain': 'nir_train_1.txt', 'visTrain': 'vis_train_1.txt', 'nirProbe': 'nir_probe_1.txt',
                        'visGallery': 'vis_gallery_1.txt'}
        '文件名路径'
        filePath = './../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
        '数据路径'
        dataPath = './../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
        start_time = time.time()
        from datetime import datetime
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        for tup in fileNameDict.keys():
            # tup = 'nirTrain'
            A = []
            fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
            filename = os.path.join(filePath, fileName)
            with open(filename) as fid:
                content = fid.read()
            content = content.split('\n')
            if content[-1] == '':
                content = content[:-1]
            '替换.jpg为.bmp'
            # if content[0][-3:] == 'jpg':
            for index in range(len(content)):
                content[index] = content[index][:-3] + 'png'
            '逐个图片读取，并标上类标'
            for idx, dataName in enumerate(content):
                # print(str(tup) + ':' + str(idx + 1))
                dataname = os.path.join(dataPath, dataName)
                A.append(dataname)

            # f = open('imglist_10k.txt')
            # A = []
            # for line in f:
            # 	line = line.strip()
            # 	A.append(line)
            # f.close()

            res = []
            # labs = []
            labs = []
            count = 0
            for i in range(len(A)):
                line = A[i]
                # print (line)

                # TODO linux
                line = line.replace('\\', '/')
                path = line
                # a=line.split('/')
                labs.append(int(line.split('/')[-2]))

                img_list = []
                image_size =Image_size
                img = misc.imread(os.path.expanduser(path), mode='RGB')
                aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)
                images = np.stack(img_list)

                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                feas = sess.run(embeddings, feed_dict=feed_dict)



                res.append(feas)
                count += 1
                # if count == 10:
                # 	break
                # if count %10 == 0:
                    # print (count)
            res = np.array(res)
            res = np.reshape(res, [len(A), feas.shape[1]])
            labs = np.array(labs)

            currentWorkPath = os.path.abspath(os.curdir)

            NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
            PkgPath = os.path.join(NewPaperTools_path, subdir)
            if not os.path.exists(PkgPath):
                try:
                    os.makedirs(PkgPath)
                except:
                    print('File exists:',PkgPath)

            sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

        'DATA Path'
        dataPath = PkgPath

        fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                        'visGallery': 'NIR_VIS_visGallery_feas.mat'}


        for tup in fileNameDict.keys():
            fileName = fileNameDict[tup]
            filename = os.path.join(dataPath, fileName)
            'read data'
            data = sio.loadmat(filename)
            if tup == 'nirTrain':
                nirTrain = data['data']
                nirTrainLabel = data['label']
            elif tup == 'visTrain':
                visTrain = data['data']
                visTrainLabel = data['label']
            elif tup == 'nirProbe':
                nirProbe = data['data']
                nirProbeLabel = data['label']
            elif tup == 'visGallery':
                visGallery = data['data']
                visGalleryLabel = data['label']

        nirTrainLabel=nirTrainLabel[0]
        visTrainLabel = visTrainLabel[0]
        nirProbeLabel = nirProbeLabel[0]
        visGalleryLabel = visGalleryLabel[0]

        nirTrain = torch.FloatTensor(nirTrain)
        visTrain = torch.FloatTensor(visTrain)
        nirProbe = torch.FloatTensor(nirProbe)
        visGallery = torch.FloatTensor(visGallery)

        nirTrain = nirTrain.cuda()
        visTrain = visTrain.cuda()
        nirProbe = nirProbe.cuda()
        visGallery = visGallery.cuda()
        print('feature extracting time cost:', time.time() - start_time)

        try:
            JB_rank1, Cos_rank1, Eucl_rank1 =-1,-1,-1
            Euclidean=False
            if Euclidean==True:
                'training Euclidean'
                print('=========================training Euclidean============================')
                print(visTrain.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visTrainLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirTrainLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirTrainLabel)  # average CMC
                CMC = np.array(CMC)
                # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('train: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)

                'testing Euclidean'
                print('========================testing Euclidean=============================')
                print(visGallery.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirProbeLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirProbeLabel)  # average CMC
                CMC=np.array(CMC)
                # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                Eucl_rank1=CMC[0]
                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('test: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)
            else:
                Eucl_rank1=-1

            #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
            Cosine=True

            if Cosine==True:
                'training Cosine'
                # print('=========================training Cosine============================')
                # print(visTrain.shape)
                # start_time = time.time()
                # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
                # VR_score_list=[]
                # VR_label_list=[]
                # for i in range(len(nirTrainLabel)):
                #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                #     if CMC_tmp[0] == -1:
                #         continue
                #     CMC = CMC + CMC_tmp
                #     # VR
                #     VR_score_list=VR_score_list+VR_score
                #     VR_label_list = VR_label_list + VR_label
                # CMC = CMC.float()
                # CMC = CMC / len(nirTrainLabel)  # average CMC
                # CMC.numpy()
                # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
                #
                # # ROC
                # ROC_curve(VR_label_list, VR_score_list)
                # print('train: Rank@1:%f' % (CMC[0]))
                # print('time cost:', time.time() - start_time)

                'testing Cosine'
                print('========================testing Cosine=============================')
                print('visGallery:',visGallery.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirProbeLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirProbeLabel)  # average CMC
                CMC = np.array(CMC)
                # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                Cos_rank1=CMC[0]
                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('test: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)
            else:
                Cos_rank1=-1

            'JB classifier'
            print('=======================JB classifier==============================')
            train_feature= torch.cat((nirTrain,visTrain)).cpu()


            nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
            visTrainLabel = np.reshape(visTrainLabel, [1, -1])
            train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

            PCAReduction_=False
            PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
            V_PercentageOrDimension_ = 128

            G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
            G = torch.FloatTensor(G)
            A = torch.FloatTensor(A)
            G=G.cuda()
            A=A.cuda()

            if PCAReduction_==True:
                visGallery=visGallery.cpu()
                nirProbe = nirProbe.cpu()
                visGallery = (np.mat(visGallery) * n_eigVect)
                nirProbe = (np.mat(nirProbe) * n_eigVect)

                visGallery=np.array(visGallery)
                nirProbe = np.array(nirProbe)

                visGallery=np.real(visGallery)
                nirProbe = np.real(nirProbe)
                visGallery = torch.FloatTensor(visGallery)
                nirProbe = torch.FloatTensor(nirProbe)

                nirProbe = nirProbe.cuda()
                visGallery = visGallery.cuda()


                print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
                print('feature dimention',visGallery.shape[1])


            'JB'
            start_time = time.time()
            print('nirProbe:',nirProbe.shape)
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

            VR_score_list=[]
            VR_label_list=[]
            # print(query_label)
            for i in range(len(nirProbeLabel)):
                CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
                # print(i, CMC_tmp[0])
                if i%1000==0 and i>0:
                    current_CMC=CMC.float()
                    current_CMC=current_CMC/ (i+1)  # average CMC
                    current_CMC=np.array(current_CMC)
                    print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            JB_rank1=CMC[0]

            # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        except:
            print('error occur !!!')
    except:
        print('error occur during processing!!!')
        JB_rank1, Cos_rank1=-1,-1
    return JB_rank1, Cos_rank1


def Nir_Vis_evaluate_Rank_1_Select(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160,protocols='1'):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_'+protocols+'.txt', 'visTrain': 'vis_train_'+protocols+'.txt', 'nirProbe': 'nir_probe_'+protocols+'.txt',
                    'visGallery': 'vis_gallery_'+protocols+'.txt'}
    '文件名路径'
    filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)

        # f = open('imglist_10k.txt')
        # A = []
        # for line in f:
        # 	line = line.strip()
        # 	A.append(line)
        # f.close()

        res = []
        # labs = []
        labs = []
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }


            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
                # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        labs = np.array(labs)

        currentWorkPath = os.path.abspath(os.curdir)

        NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
        PkgPath = os.path.join(NewPaperTools_path, subdir)
        if not os.path.exists(PkgPath):
            os.makedirs(PkgPath)

        sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    dataPath = PkgPath

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}


    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=False
        PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 128

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1

def Nir_Vis_evaluate_Rank_1_JB_PCA(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160,Dimension = 128):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_1.txt', 'visTrain': 'vis_train_1.txt', 'nirProbe': 'nir_probe_1.txt',
                    'visGallery': 'vis_gallery_1.txt'}
    '文件名路径'
    filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)

        # f = open('imglist_10k.txt')
        # A = []
        # for line in f:
        # 	line = line.strip()
        # 	A.append(line)
        # f.close()

        res = []
        # labs = []
        labs = []
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }

            feas = sess.run(embeddings, feed_dict=feed_dict)
            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
                # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        labs = np.array(labs)

        currentWorkPath = os.path.abspath(os.curdir)

        NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
        PkgPath = os.path.join(NewPaperTools_path, subdir)
        if not os.path.exists(PkgPath):
            try:
                os.makedirs(PkgPath)
            except:
                print('File exists:',PkgPath)

        sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    dataPath = PkgPath

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}


    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:
        JB_rank1, Cos_rank1, Eucl_rank1 =-1,-1,-1
        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=True
        PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = Dimension

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery=visGallery.cpu()
            nirProbe = nirProbe.cpu()
            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)
    except:
        print('error occur !!!')
    return JB_rank1, Cos_rank1

def Nir_Vis_evaluate_Rank_1_Select_test(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160,protocols='1'):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_'+protocols+'.txt', 'visTrain': 'vis_train_'+protocols+'.txt', 'nirProbe': 'nir_probe_'+protocols+'.txt',
                    'visGallery': 'vis_gallery_'+protocols+'.txt'}
    '文件名路径'
    filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)

        # f = open('imglist_10k.txt')
        # A = []
        # for line in f:
        # 	line = line.strip()
        # 	A.append(line)
        # f.close()

        res = []
        # labs = []
        labs = []
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }


            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
                # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        labs = np.array(labs)

        currentWorkPath = os.path.abspath(os.curdir)

        NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
        PkgPath = os.path.join(NewPaperTools_path, subdir)
        if not os.path.exists(PkgPath):
            os.makedirs(PkgPath)

        sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    dataPath = PkgPath

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}


    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)



    Euclidean=False
    if Euclidean==True:
        'training Euclidean'
        print('=========================training Euclidean============================')
        print(visTrain.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visTrainLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirTrainLabel)):
            CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirTrainLabel)  # average CMC
        CMC = np.array(CMC)
        # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('train: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)

        'testing Euclidean'
        print('========================testing Euclidean=============================')
        print(visGallery.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirProbeLabel)):
            CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC=np.array(CMC)
        # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        Eucl_rank1=CMC[0]
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)
    else:
        Eucl_rank1=-1

    #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
    Cosine=True

    if Cosine==True:
        'training Cosine'
        # print('=========================training Cosine============================')
        # print(visTrain.shape)
        # start_time = time.time()
        # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
        # VR_score_list=[]
        # VR_label_list=[]
        # for i in range(len(nirTrainLabel)):
        #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
        #     if CMC_tmp[0] == -1:
        #         continue
        #     CMC = CMC + CMC_tmp
        #     # VR
        #     VR_score_list=VR_score_list+VR_score
        #     VR_label_list = VR_label_list + VR_label
        # CMC = CMC.float()
        # CMC = CMC / len(nirTrainLabel)  # average CMC
        # CMC.numpy()
        # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        #
        # # ROC
        # ROC_curve(VR_label_list, VR_score_list)
        # print('train: Rank@1:%f' % (CMC[0]))
        # print('time cost:', time.time() - start_time)

        'testing Cosine'
        print('========================testing Cosine=============================')
        print(visGallery.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirProbeLabel)):
            CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        Cos_rank1=CMC[0]
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)
    else:
        Cos_rank1=-1

    'JB classifier'
    print('=======================JB classifier==============================')
    train_feature= torch.cat((nirTrain,visTrain)).cpu()


    nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
    visTrainLabel = np.reshape(visTrainLabel, [1, -1])
    train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

    PCAReduction_=False
    PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
    V_PercentageOrDimension_ = 128

    G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
    G = torch.FloatTensor(G)
    A = torch.FloatTensor(A)
    G=G.cuda()
    A=A.cuda()

    if PCAReduction_==True:
        visGallery = (np.mat(visGallery) * n_eigVect)
        nirProbe = (np.mat(nirProbe) * n_eigVect)

        visGallery=np.array(visGallery)
        nirProbe = np.array(nirProbe)

        visGallery=np.real(visGallery)
        nirProbe = np.real(nirProbe)
        visGallery = torch.FloatTensor(visGallery)
        nirProbe = torch.FloatTensor(nirProbe)

        nirProbe = nirProbe.cuda()
        visGallery = visGallery.cuda()


        print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
        print('feature dimention',visGallery.shape[1])


    'JB'
    start_time = time.time()
    print(nirProbe.shape)
    CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

    VR_score_list=[]
    VR_label_list=[]
    # print(query_label)
    for i in range(len(nirProbeLabel)):
        CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        # VR
        VR_score_list=VR_score_list+VR_score
        VR_label_list = VR_label_list + VR_label
        # print(i, CMC_tmp[0])
        if i%1000==0 and i>0:
            current_CMC=CMC.float()
            current_CMC=current_CMC/ (i+1)  # average CMC
            current_CMC=np.array(current_CMC)
            print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
    CMC = CMC.float()
    CMC = CMC / len(nirProbeLabel)  # average CMC
    CMC = np.array(CMC)
    JB_rank1=CMC[0]

    # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
    # ROC
    ROC_curve(VR_label_list, VR_score_list)
    print('test: Rank@1:%f' % (CMC[0]))
    # JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1

def Nir_Vis_evaluate_Rank_1(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160):

    try:
        'DATA Path'
        currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
        fileNameDict = {'nirTrain': 'nir_train_1.txt', 'visTrain': 'vis_train_1.txt', 'nirProbe': 'nir_probe_1.txt',
                        'visGallery': 'vis_gallery_1.txt'}
        '文件名路径'
        filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
        '数据路径'
        dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
        start_time = time.time()
        from datetime import datetime
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        for tup in fileNameDict.keys():
            # tup = 'nirTrain'
            A = []
            fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
            filename = os.path.join(filePath, fileName)
            with open(filename) as fid:
                content = fid.read()
            content = content.split('\n')
            if content[-1] == '':
                content = content[:-1]
            '替换.jpg为.bmp'
            # if content[0][-3:] == 'jpg':
            for index in range(len(content)):
                content[index] = content[index][:-3] + 'png'
            '逐个图片读取，并标上类标'
            for idx, dataName in enumerate(content):
                # print(str(tup) + ':' + str(idx + 1))
                dataname = os.path.join(dataPath, dataName)
                A.append(dataname)

            # f = open('imglist_10k.txt')
            # A = []
            # for line in f:
            # 	line = line.strip()
            # 	A.append(line)
            # f.close()

            res = []
            # labs = []
            labs = []
            count = 0
            for i in range(len(A)):
                line = A[i]
                # print (line)

                # TODO linux
                line = line.replace('\\', '/')
                path = line
                # a=line.split('/')
                labs.append(int(line.split('/')[-2]))

                img_list = []
                image_size =Image_size
                img = misc.imread(os.path.expanduser(path), mode='RGB')
                aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)
                images = np.stack(img_list)

                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                feas = sess.run(embeddings, feed_dict=feed_dict)



                res.append(feas)
                count += 1
                # if count == 10:
                # 	break
                # if count %10 == 0:
                    # print (count)
            res = np.array(res)
            res = np.reshape(res, [len(A), feas.shape[1]])
            labs = np.array(labs)

            currentWorkPath = os.path.abspath(os.curdir)

            NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
            PkgPath = os.path.join(NewPaperTools_path, subdir)
            if not os.path.exists(PkgPath):
                try:
                    os.makedirs(PkgPath)
                except:
                    print('File exists:',PkgPath)

            sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

        'DATA Path'
        dataPath = PkgPath

        fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                        'visGallery': 'NIR_VIS_visGallery_feas.mat'}


        for tup in fileNameDict.keys():
            fileName = fileNameDict[tup]
            filename = os.path.join(dataPath, fileName)
            'read data'
            data = sio.loadmat(filename)
            if tup == 'nirTrain':
                nirTrain = data['data']
                nirTrainLabel = data['label']
            elif tup == 'visTrain':
                visTrain = data['data']
                visTrainLabel = data['label']
            elif tup == 'nirProbe':
                nirProbe = data['data']
                nirProbeLabel = data['label']
            elif tup == 'visGallery':
                visGallery = data['data']
                visGalleryLabel = data['label']

        nirTrainLabel=nirTrainLabel[0]
        visTrainLabel = visTrainLabel[0]
        nirProbeLabel = nirProbeLabel[0]
        visGalleryLabel = visGalleryLabel[0]

        nirTrain = torch.FloatTensor(nirTrain)
        visTrain = torch.FloatTensor(visTrain)
        nirProbe = torch.FloatTensor(nirProbe)
        visGallery = torch.FloatTensor(visGallery)

        nirTrain = nirTrain.cuda()
        visTrain = visTrain.cuda()
        nirProbe = nirProbe.cuda()
        visGallery = visGallery.cuda()
        print('feature extracting time cost:', time.time() - start_time)

        try:
            JB_rank1, Cos_rank1, Eucl_rank1 =-1,-1,-1
            Euclidean=False
            if Euclidean==True:
                'training Euclidean'
                print('=========================training Euclidean============================')
                print(visTrain.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visTrainLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirTrainLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirTrainLabel)  # average CMC
                CMC = np.array(CMC)
                # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('train: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)

                'testing Euclidean'
                print('========================testing Euclidean=============================')
                print(visGallery.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirProbeLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirProbeLabel)  # average CMC
                CMC=np.array(CMC)
                # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                Eucl_rank1=CMC[0]
                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('test: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)
            else:
                Eucl_rank1=-1

            #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
            Cosine=True

            if Cosine==True:
                'training Cosine'
                # print('=========================training Cosine============================')
                # print(visTrain.shape)
                # start_time = time.time()
                # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
                # VR_score_list=[]
                # VR_label_list=[]
                # for i in range(len(nirTrainLabel)):
                #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                #     if CMC_tmp[0] == -1:
                #         continue
                #     CMC = CMC + CMC_tmp
                #     # VR
                #     VR_score_list=VR_score_list+VR_score
                #     VR_label_list = VR_label_list + VR_label
                # CMC = CMC.float()
                # CMC = CMC / len(nirTrainLabel)  # average CMC
                # CMC.numpy()
                # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
                #
                # # ROC
                # ROC_curve(VR_label_list, VR_score_list)
                # print('train: Rank@1:%f' % (CMC[0]))
                # print('time cost:', time.time() - start_time)

                'testing Cosine'
                print('========================testing Cosine=============================')
                print('visGallery:',visGallery.shape)
                start_time = time.time()
                CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
                VR_score_list=[]
                VR_label_list=[]
                for i in range(len(nirProbeLabel)):
                    CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC = CMC + CMC_tmp
                    # VR
                    VR_score_list=VR_score_list+VR_score
                    VR_label_list = VR_label_list + VR_label
                CMC = CMC.float()
                CMC = CMC / len(nirProbeLabel)  # average CMC
                CMC = np.array(CMC)
                # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

                Cos_rank1=CMC[0]
                # ROC
                ROC_curve(VR_label_list, VR_score_list)
                print('test: Rank@1:%f' % (CMC[0]))
                print('time cost:', time.time() - start_time)
            else:
                Cos_rank1=-1

            'JB classifier'
            print('=======================JB classifier==============================')
            train_feature= torch.cat((nirTrain,visTrain)).cpu()


            nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
            visTrainLabel = np.reshape(visTrainLabel, [1, -1])
            train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

            PCAReduction_=False
            PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
            V_PercentageOrDimension_ = 128

            G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
            G = torch.FloatTensor(G)
            A = torch.FloatTensor(A)
            G=G.cuda()
            A=A.cuda()

            if PCAReduction_==True:
                visGallery=visGallery.cpu()
                nirProbe = nirProbe.cpu()
                visGallery = (np.mat(visGallery) * n_eigVect)
                nirProbe = (np.mat(nirProbe) * n_eigVect)

                visGallery=np.array(visGallery)
                nirProbe = np.array(nirProbe)

                visGallery=np.real(visGallery)
                nirProbe = np.real(nirProbe)
                visGallery = torch.FloatTensor(visGallery)
                nirProbe = torch.FloatTensor(nirProbe)

                nirProbe = nirProbe.cuda()
                visGallery = visGallery.cuda()


                print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
                print('feature dimention',visGallery.shape[1])


            'JB'
            start_time = time.time()
            print('nirProbe:',nirProbe.shape)
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

            VR_score_list=[]
            VR_label_list=[]
            # print(query_label)
            for i in range(len(nirProbeLabel)):
                CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
                # print(i, CMC_tmp[0])
                if i%1000==0 and i>0:
                    current_CMC=CMC.float()
                    current_CMC=current_CMC/ (i+1)  # average CMC
                    current_CMC=np.array(current_CMC)
                    print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            JB_rank1=CMC[0]

            # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        except:
            print('error occur !!!')
    except:
        print('error occur during processing!!!')
        JB_rank1, Cos_rank1=-1,-1
    return JB_rank1, Cos_rank1


def Nir_Vis_reconstruct_images(sess,images_placeholder,reconstImg,phase_train_placeholder,Image_size=160,protocols='1',persentage_save_img=0.1,rootPath='CASIA_NIR_VIS'):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_'+protocols+'.txt', 'visTrain': 'vis_train_'+protocols+'.txt', 'nirProbe': 'nir_probe_'+protocols+'.txt',
                    'visGallery': 'vis_gallery_'+protocols+'.txt'}
    '文件名路径'
    filePath = './../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = './../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0'

    NeutralPath='./../Datasets/CASIA NIR-VIS 2.0/NIR_VIS_protocols_'+protocols+'_neutralface_160_5/'

    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    if not os.path.exists('./PlotReconstImg'):
        os.makedirs('./PlotReconstImg')

    rootPath_ = os.path.join('./PlotReconstImg', rootPath)
    if not os.path.exists(rootPath_):
        os.makedirs(rootPath_)

    subdir_ = os.path.join(rootPath_, subdir)
    if not os.path.exists(subdir_):
        os.makedirs(subdir_)

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)

        res = []
        # labs = []
        labs = []
        count = 0
        saveImgNum=int(persentage_save_img*len(A))

        subset_dir = os.path.join(subdir_, tup)

        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

        for i in range(saveImgNum):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }

            reconstImg_array = sess.run(reconstImg, feed_dict=feed_dict)
            reconstImg_array = reconstImg_array[0, :, :, :]

            count += 1

            if tup=='nirTrain' or tup=='visTrain':
                NeutralClass_dir = os.path.join(NeutralPath, line.split('/')[-2])
                items=os.listdir(NeutralClass_dir)
                for item in items:
                    if 'VIS' in item:
                        NeutralImg_dir=os.path.join(NeutralClass_dir,item)
                        imgNeutral = misc.imread(os.path.expanduser(NeutralImg_dir), mode='RGB')
                        alignedNeutral = misc.imresize(imgNeutral, (image_size, image_size), interp='bilinear')
                        prewhitenedNeutral = facenet.prewhiten(alignedNeutral)
            else:
                prewhitenedNeutral=np.zeros([image_size,image_size,3])




            prewhitened=cv2.normalize(prewhitened,prewhitened,0,255,cv2.NORM_MINMAX)
            reconstImg_array=cv2.normalize(reconstImg_array,reconstImg_array,0,255,cv2.NORM_MINMAX)
            prewhitenedNeutral=cv2.normalize(prewhitenedNeutral,prewhitenedNeutral,0,255,cv2.NORM_MINMAX)

            r,g,b=cv2.split(prewhitened)
            prewhitened=cv2.merge([b,g,r])
            r,g,b=cv2.split(reconstImg_array)
            reconstImg_array=cv2.merge([b,g,r])
            r,g,b=cv2.split(prewhitenedNeutral)
            prewhitenedNeutral=cv2.merge([b,g,r])

            img_concat = np.hstack((prewhitened, reconstImg_array,prewhitenedNeutral))
            fileName=line.split('/')[-4]+'_'+line.split('/')[-3]+'_'+line.split('/')[-2]+'_'+line.split('/')[-1]
            ImgSave_dir = os.path.join(subset_dir, fileName)

            cv2.imwrite(ImgSave_dir, img_concat)


def Nir_Vis_evaluate_Rank_1_Select_matchConfidence_without_MD(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160,protocols='1'):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_'+protocols+'.txt', 'visTrain': 'vis_train_'+protocols+'.txt', 'nirProbe': 'nir_probe_'+protocols+'.txt',
                    'visGallery': 'vis_gallery_'+protocols+'.txt'}
    '文件名路径'
    filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)
        if tup=='nirProbe':
            nirProbe_PathList=A
        if tup=='visGallery':
            visGallery_PathList=A

        # f = open('imglist_10k.txt')
        # A = []
        # for line in f:
        # 	line = line.strip()
        # 	A.append(line)
        # f.close()

        res = []
        # labs = []
        labs = []
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }


            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
                # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        labs = np.array(labs)

        currentWorkPath = os.path.abspath(os.curdir)

        NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
        PkgPath = os.path.join(NewPaperTools_path, subdir)
        if not os.path.exists(PkgPath):
            try:
                os.makedirs(PkgPath)
            except:
                print('File exists:',PkgPath)

        sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    dataPath = PkgPath

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}


    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    Euclidean=False
    if Euclidean==True:
        'training Euclidean'
        print('=========================training Euclidean============================')
        print(visTrain.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visTrainLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirTrainLabel)):
            CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirTrainLabel)  # average CMC
        CMC = np.array(CMC)
        # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('train: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)

        'testing Euclidean'
        print('========================testing Euclidean=============================')
        print(visGallery.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirProbeLabel)):
            CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC=np.array(CMC)
        # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        Eucl_rank1=CMC[0]
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)
    else:
        Eucl_rank1=-1

    #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
    Cosine=True

    if Cosine==True:
        'training Cosine'
        # print('=========================training Cosine============================')
        # print(visTrain.shape)
        # start_time = time.time()
        # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
        # VR_score_list=[]
        # VR_label_list=[]
        # for i in range(len(nirTrainLabel)):
        #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
        #     if CMC_tmp[0] == -1:
        #         continue
        #     CMC = CMC + CMC_tmp
        #     # VR
        #     VR_score_list=VR_score_list+VR_score
        #     VR_label_list = VR_label_list + VR_label
        # CMC = CMC.float()
        # CMC = CMC / len(nirTrainLabel)  # average CMC
        # CMC.numpy()
        # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        #
        # # ROC
        # ROC_curve(VR_label_list, VR_score_list)
        # print('train: Rank@1:%f' % (CMC[0]))
        # print('time cost:', time.time() - start_time)


        'testing Cosine'
        print('========================testing Cosine=============================')
        print(visGallery.shape)
        start_time = time.time()
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
        VR_score_list=[]
        VR_label_list=[]
        for i in range(len(nirProbeLabel)):
            CMC_tmp,VR_score,VR_label = evaluate_visualMatch(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel,nirProbe_PathList[i],visGallery_PathList,subdir)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

        Cos_rank1=CMC[0]
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
        print('time cost:', time.time() - start_time)
    else:
        Cos_rank1=-1

    'JB classifier'
    print('=======================JB classifier==============================')
    train_feature= torch.cat((nirTrain,visTrain)).cpu()


    nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
    visTrainLabel = np.reshape(visTrainLabel, [1, -1])
    train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

    PCAReduction_=False
    PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
    V_PercentageOrDimension_ = 128

    G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
    G = torch.FloatTensor(G)
    A = torch.FloatTensor(A)
    G=G.cuda()
    A=A.cuda()

    if PCAReduction_==True:
        visGallery = (np.mat(visGallery) * n_eigVect)
        nirProbe = (np.mat(nirProbe) * n_eigVect)

        visGallery=np.array(visGallery)
        nirProbe = np.array(nirProbe)

        visGallery=np.real(visGallery)
        nirProbe = np.real(nirProbe)
        visGallery = torch.FloatTensor(visGallery)
        nirProbe = torch.FloatTensor(nirProbe)

        nirProbe = nirProbe.cuda()
        visGallery = visGallery.cuda()


        print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
        print('feature dimention',visGallery.shape[1])


    'JB'
    start_time = time.time()
    print(nirProbe.shape)
    CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

    VR_score_list=[]
    VR_label_list=[]
    # print(query_label)
    for i in range(len(nirProbeLabel)):
        CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        # VR
        VR_score_list=VR_score_list+VR_score
        VR_label_list = VR_label_list + VR_label
        # print(i, CMC_tmp[0])
        if i%1000==0 and i>0:
            current_CMC=CMC.float()
            current_CMC=current_CMC/ (i+1)  # average CMC
            current_CMC=np.array(current_CMC)
            print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
    CMC = CMC.float()
    CMC = CMC / len(nirProbeLabel)  # average CMC
    CMC = np.array(CMC)
    JB_rank1=CMC[0]

    # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
    # ROC
    ROC_curve(VR_label_list, VR_score_list)
    print('test: Rank@1:%f' % (CMC[0]))

    return JB_rank1, Cos_rank1


def Nir_Vis_evaluate_Rank_1_Select_matchConfidence(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160,protocols='1',rootPath='Root'):


    'DATA Path'
    currentWorkPath = os.path.dirname(os.path.dirname(os.getcwd()))
    fileNameDict = {'nirTrain': 'nir_train_'+protocols+'.txt', 'visTrain': 'vis_train_'+protocols+'.txt', 'nirProbe': 'nir_probe_'+protocols+'.txt',
                    'visGallery': 'vis_gallery_'+protocols+'.txt'}
    '文件名路径'
    filePath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols'
    '数据路径'
    dataPath = '/home/hwpeng/Database/CASIA NIR-VIS 2.0/NIR-VIS-2.0'
    start_time = time.time()
    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    for tup in fileNameDict.keys():
        # tup = 'nirTrain'
        A = []
        fileName = fileNameDict[tup]  # 'nir_train_1.txt' 'nir_probe_1'  'vis_gallery_1'  'vis_train_1'
        filename = os.path.join(filePath, fileName)
        with open(filename) as fid:
            content = fid.read()
        content = content.split('\n')
        if content[-1] == '':
            content = content[:-1]
        '替换.jpg为.bmp'
        # if content[0][-3:] == 'jpg':
        for index in range(len(content)):
            content[index] = content[index][:-3] + 'png'
        '逐个图片读取，并标上类标'
        for idx, dataName in enumerate(content):
            # print(str(tup) + ':' + str(idx + 1))
            dataname = os.path.join(dataPath, dataName)
            A.append(dataname)
        if tup=='nirProbe':
            nirProbe_PathList=A
        if tup=='visGallery':
            visGallery_PathList=A

        # f = open('imglist_10k.txt')
        # A = []
        # for line in f:
        # 	line = line.strip()
        # 	A.append(line)
        # f.close()

        res = []
        # labs = []
        labs = []
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }


            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
                # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        labs = np.array(labs)

        currentWorkPath = os.path.abspath(os.curdir)

        NewPaperTools_path = os.path.join(currentWorkPath, 'NewPaperTools')
        PkgPath = os.path.join(NewPaperTools_path, subdir)
        if not os.path.exists(PkgPath):
            try:
                os.makedirs(PkgPath)
            except:
                print('File exists:',PkgPath)

        sio.savemat(PkgPath + '/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    dataPath = PkgPath

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat', 'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}


    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)


            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]

            match_confidence_matrix=np.zeros([np.shape(visGalleryLabel)[0],np.shape(visGalleryLabel)[0]])
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_visualMatch(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel,nirProbe_PathList[i],visGallery_PathList,subdir,rootPath)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label

                # match confidence plot data collect
                ProbeClassNum=sum(nirProbeLabel==nirProbeLabel[i])
                MCM_rowIndex=np.argwhere(visGalleryLabel == nirProbeLabel[i])[0,0] #each gallery contains only one image
                match_confidence_matrix[MCM_rowIndex,:]=match_confidence_matrix[MCM_rowIndex,:]+(1/ProbeClassNum)*np.array(VR_score)

            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            Cos_rank1=CMC[0]

            #mean distance Positive pairs and mean distance Negative pairs
            VR_score_np=np.array(VR_score_list)
            VR_label_list = np.array(VR_label_list)
            positive_index = np.argwhere(VR_label_list == True)
            negative_index = np.argwhere(VR_label_list == False)
            VR_score_positive=VR_score_np[positive_index]
            VR_score_negative = VR_score_np[negative_index]
            MD_positive_pairs=round(np.sum(VR_score_positive)/np.shape(VR_score_positive)[0],4)
            MD_negative_pairs = round(np.sum(VR_score_negative) / np.shape(VR_score_negative)[0], 4)
            print('MD_positive_pairs',MD_positive_pairs,'MD_negative_pairs',MD_negative_pairs)

            #match confidence plot
            rootPath_ = os.path.join('./PlotMatchConfidence', rootPath)
            subdir_ = os.path.join(rootPath_, subdir)
            fname='CosRank1:'+str(round(CMC[0],4))+'_MD_positive_pairs_'+str(MD_positive_pairs)+'_MD_negative_pairs_'+str(MD_negative_pairs)
            output_path = os.path.join(subdir_, fname + ".png")
            print('output_path',output_path)
            match_confidence_matrix_=match_confidence_matrix*255
            cv2.imwrite(output_path, match_confidence_matrix_)

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=False
        PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 128

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('visualMatch subdir:',subdir)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1

def evaluate_visualMatch(qf, ql, gf, gl,nirProbe_Path,visGallery_PathList,subdir,rootPath):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    score_ori=score.copy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index=np.reshape(query_index,[-1])
    score_GT = score_ori[good_index][0]

    CMC_tmp = compute_mAP_visualMatch(index, good_index)
    plot_query_topK(score,CMC_tmp[0],query_index[0,0],score_GT, subdir,rootPath, nirProbe_Path, visGallery_PathList, index, topk=5, width=200, high=200)
    # VR score and label
    VR_index=np.array(range(0,len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label=list(VR_label)
    VR_score=list(score)

    return CMC_tmp,VR_score,VR_label

def compute_mAP_visualMatch(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # find good_index index
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1

    return cmc

def plot_query_topK(score,CMC0,query_index,score_GT,subdir,rootPath,nirProbe_Path,visGallery_PathList,index,topk = 5,width=200, high = 200):

    if not os.path.exists('./PlotMatchConfidence'):
        os.makedirs('./PlotMatchConfidence')

    rootPath_ = os.path.join('./PlotMatchConfidence', rootPath)
    if not os.path.exists(rootPath_):
        os.makedirs(rootPath_)

    subdir_=os.path.join(rootPath_, subdir)
    if not os.path.exists(subdir_):
        os.makedirs(subdir_)

    true_pairs_dir = os.path.join(subdir_, 'true_pairs')
    false_pairs_dir = os.path.join(subdir_, 'false_pairs')
    if not os.path.exists(true_pairs_dir):
        os.makedirs(true_pairs_dir)
    if not os.path.exists(false_pairs_dir):
        os.makedirs(false_pairs_dir)
    score_=list(score)
    # fname_0 = os.path.splitext(os.path.basename(nirProbe_Path))[0]
    Probe_Path_str=nirProbe_Path.replace('//','_')
    Probe_Path_str = Probe_Path_str.replace('\\', '_')
    Probe_Path_str = Probe_Path_str.replace('/', '_')


    nirProbe_Path = nirProbe_Path.replace('\\', '/')
    im_input_probe = cv2.imread(nirProbe_Path, -1)
    im_input_probe = cv2.resize(im_input_probe, (width, high))
    im_input_probe = im_input_probe[:, :, 0:3]
    for i in range(topk):
        visGallery_PathList[index[i]] = visGallery_PathList[index[i]].replace('\\', '/')
        im_input_gallery = cv2.imread(visGallery_PathList[index[i]], -1)
        im_input_gallery = cv2.resize(im_input_gallery, (width, high))
        im_input_gallery = im_input_gallery[:, :, 0:3]
        try:
            if i == 0:
                img_concat = np.hstack((im_input_probe, im_input_gallery))
            else:
                img_concat = np.hstack((img_concat, im_input_gallery))
        except:
            print('img_concat error !!!')
            pass
    if CMC0 == 1:
        score_record = str(round(score_[index[0]], 4)) + '_' + str(round(score_[index[1]], 4)) + '_' + str(round(score_[index[2]], 4)) + '_' + str(round(score_[index[3]], 4)) + '_' + str(round(score_[index[4]], 4))+ '_score_GT' + str(round(score_GT, 4))
        decribe = 'ImageOrder_probe_top1-top' + str(topk)
        fname = Probe_Path_str + '_' + decribe + '_ScorePairs_' + score_record
        output_path = os.path.join(true_pairs_dir, fname + ".png")
        cv2.imwrite(output_path, img_concat)
    else:

        score_record = str(round(score_[index[0]], 4)) + '_' + str(round(score_[index[1]], 4)) + '_' + str(round(score_[index[2]], 4)) + '_' + str(round(score_[index[3]], 4)) + '_' + str(round(score_[index[4]], 4))+ '_score_GT' + str(round(score_GT, 4))
        decribe = 'ImageOrder_probe_top1-top' + str(topk)+'_GroundTrue'
        fname = Probe_Path_str + '_' + decribe + '_ScorePairs_' + score_record
        visGallery_PathList[query_index] = visGallery_PathList[query_index].replace('\\', '/')
        im_input_gallery_GT = cv2.imread(visGallery_PathList[query_index], -1)
        im_input_gallery_GT = cv2.resize(im_input_gallery_GT, (width, high))
        im_input_gallery_GT = im_input_gallery_GT[:, :, 0:3]
        img_concat = np.hstack((img_concat, im_input_gallery_GT))

        output_path = os.path.join(false_pairs_dir, fname + ".png")
        cv2.imwrite(output_path, img_concat)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths



def Nir_Vis_evaluate_Rank_1_Oulu(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160):
    start_time = time.time()

    fileNameDict = {
        'nirTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/NIR/',
        'visTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/VIS/',
        'nirProbe': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/NIR/',
        'visGallery': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/VIS/'}

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        res = []
        # labs = []
        # labs = np.empty([len(A), 1], dtype=object)
        labs=[]
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            # labs[count, 0] = int(line.split('/')[-2])
            labs.append(int(line.split('/')[-2]))
            img_list = []
            image_size = Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
            # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        # print (res.shape)
        # print (labs.shape)
        sio.savemat('./NewPaperTools/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    currentWorkPath = os.path.abspath(os.curdir)
    dataPath = os.path.join(currentWorkPath, 'NewPaperTools')

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat',
                    'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}

    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=False
        PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 128

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = visGallery.cpu()
            visGallery = visGallery.numpy()
            nirProbe = nirProbe.cpu()
            nirProbe = nirProbe.numpy()

            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1

def Nir_Vis_reconstruct_images_Oulu(sess,images_placeholder,reconstImg,phase_train_placeholder,Image_size=160,persentage_save_img=0.1,rootPath='Oulu_NIR_VIS'):


    fileNameDict = {
        'nirTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/NIR/',
        'visTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/VIS/',
        'nirProbe': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/NIR/',
        'visGallery': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/VIS/'}

    NeutralPath = './../Datasets/Oulu_40Class/Oulu_20class_train/NIR_VIS_neutralface'

    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    if not os.path.exists('./PlotReconstImg'):
        os.makedirs('./PlotReconstImg')

    rootPath_ = os.path.join('./PlotReconstImg', rootPath)
    if not os.path.exists(rootPath_):
        os.makedirs(rootPath_)

    subdir_ = os.path.join(rootPath_, subdir)
    if not os.path.exists(subdir_):
        os.makedirs(subdir_)

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        labs = []
        count = 0
        saveImgNum=int(persentage_save_img*len(A))

        subset_dir = os.path.join(subdir_, tup)

        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

        for i in range(saveImgNum):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }

            reconstImg_array = sess.run(reconstImg, feed_dict=feed_dict)
            reconstImg_array = reconstImg_array[0, :, :, :]

            count += 1

            if tup=='nirTrain' or tup=='visTrain':
                NeutralClass_dir = os.path.join(NeutralPath, line.split('/')[-2])
                items=os.listdir(NeutralClass_dir)
                for item in items:
                    if 'VIS' in item:
                        NeutralImg_dir=os.path.join(NeutralClass_dir,item)
                        imgNeutral = misc.imread(os.path.expanduser(NeutralImg_dir), mode='RGB')
                        alignedNeutral = misc.imresize(imgNeutral, (image_size, image_size), interp='bilinear')
                        prewhitenedNeutral = facenet.prewhiten(alignedNeutral)
            else:
                prewhitenedNeutral=np.zeros([image_size,image_size,3])




            prewhitened=cv2.normalize(prewhitened,prewhitened,0,255,cv2.NORM_MINMAX)
            reconstImg_array=cv2.normalize(reconstImg_array,reconstImg_array,0,255,cv2.NORM_MINMAX)
            prewhitenedNeutral=cv2.normalize(prewhitenedNeutral,prewhitenedNeutral,0,255,cv2.NORM_MINMAX)

            r,g,b=cv2.split(prewhitened)
            prewhitened=cv2.merge([b,g,r])
            r,g,b=cv2.split(reconstImg_array)
            reconstImg_array=cv2.merge([b,g,r])
            r,g,b=cv2.split(prewhitenedNeutral)
            prewhitenedNeutral=cv2.merge([b,g,r])

            img_concat = np.hstack((prewhitened, reconstImg_array,prewhitenedNeutral))
            fileName=line.split('/')[-4]+'_'+line.split('/')[-3]+'_'+line.split('/')[-2]+'_'+line.split('/')[-1]
            ImgSave_dir = os.path.join(subset_dir, fileName)

            cv2.imwrite(ImgSave_dir, img_concat)


def Nir_Vis_evaluate_Rank_1_Oulu_PCA(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160):
    start_time = time.time()

    fileNameDict = {
        'nirTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/NIR/',
        'visTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_train/VIS/',
        'nirProbe': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/NIR/',
        'visGallery': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/VIS/'}

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        res = []
        # labs = []
        # labs = np.empty([len(A), 1], dtype=object)
        labs=[]
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            # labs[count, 0] = int(line.split('/')[-2])
            labs.append(int(line.split('/')[-2]))
            img_list = []
            image_size = 160
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
            # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        # print (res.shape)
        # print (labs.shape)
        sio.savemat('./NewPaperTools/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    currentWorkPath = os.path.abspath(os.curdir)
    dataPath = os.path.join(currentWorkPath, 'NewPaperTools')

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat',
                    'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}

    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=True
        PercenOrDim_ = 'Percentage' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 0.96

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = visGallery.cpu()
            visGallery = visGallery.numpy()
            nirProbe = nirProbe.cpu()
            nirProbe = nirProbe.numpy()

            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1



def Nir_Vis_evaluate_Rank_1_Oulu_DataAug(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160, PCA_Process=False):
    start_time = time.time()

    fileNameDict = {
        'nirTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/DataAugmentation/Oulu_20class_train/NIR/',
        'visTrain': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/DataAugmentation/Oulu_20class_train/VIS/',
        'nirProbe': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/NIR/',
        'visGallery': '/home/hwpeng/Database/OuluCasIA/OuluCasia_40Class/Oulu_40Class/Oulu_20class_test/VIS/'}

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        res = []
        # labs = []
        # labs = np.empty([len(A), 1], dtype=object)
        labs=[]
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            # labs[count, 0] = int(line.split('/')[-2])
            labs.append(int(line.split('/')[-2]))
            img_list = []
            image_size = 160
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
            # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        # print (res.shape)
        # print (labs.shape)
        sio.savemat('./NewPaperTools/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    currentWorkPath = os.path.abspath(os.curdir)
    dataPath = os.path.join(currentWorkPath, 'NewPaperTools')

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat',
                    'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}

    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            # print('=========================training Cosine============================')
            # print(visTrain.shape)
            # start_time = time.time()
            # CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            # VR_score_list=[]
            # VR_label_list=[]
            # for i in range(len(nirTrainLabel)):
            #     CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
            #     if CMC_tmp[0] == -1:
            #         continue
            #     CMC = CMC + CMC_tmp
            #     # VR
            #     VR_score_list=VR_score_list+VR_score
            #     VR_label_list = VR_label_list + VR_label
            # CMC = CMC.float()
            # CMC = CMC / len(nirTrainLabel)  # average CMC
            # CMC.numpy()
            # # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
            #
            # # ROC
            # ROC_curve(VR_label_list, VR_score_list)
            # print('train: Rank@1:%f' % (CMC[0]))
            # print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=PCA_Process
        PercenOrDim_ = 'Percentage' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 0.96

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = visGallery.cpu()
            visGallery = visGallery.numpy()
            nirProbe = nirProbe.cpu()
            nirProbe = nirProbe.numpy()

            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1



def Nir_Vis_evaluate_Rank_1_BUAA(sess,images_placeholder,embeddings,phase_train_placeholder,Image_size=160):
    start_time = time.time()

    fileNameDict = {
        'nirTrain': '/home/hwpeng/Database/BUAA_NIRVIS/BUAA_VISNIR_train_test/BUAA_VISNIR_train/NIR/',
        'visTrain': '/home/hwpeng/Database/BUAA_NIRVIS/BUAA_VISNIR_train_test/BUAA_VISNIR_train/VIS/',
        'nirProbe': '/home/hwpeng/Database/BUAA_NIRVIS/BUAA_VISNIR_train_test/BUAA_VISNIR_test/NIR/',
        'visGallery': '/home/hwpeng/Database/BUAA_NIRVIS/BUAA_VISNIR_train_test/BUAA_VISNIR_test/VIS/'}

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        res = []
        # labs = []
        # labs = np.empty([len(A), 1], dtype=object)
        labs=[]
        count = 0
        for i in range(len(A)):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            # labs[count, 0] = int(line.split('/')[-2])
            labs.append(int(line.split('/')[-2]))
            img_list = []
            image_size = Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            feas = sess.run(embeddings, feed_dict=feed_dict)

            res.append(feas)
            count += 1
            # if count == 10:
            # 	break
            # if count %10 == 0:
            # print (count)
        res = np.array(res)
        res = np.reshape(res, [len(A), feas.shape[1]])
        # print (res.shape)
        # print (labs.shape)
        sio.savemat('./NewPaperTools/' + 'NIR_VIS_' + tup + '_feas.mat', {'data': res, 'label': labs})

    'DATA Path'
    currentWorkPath = os.path.abspath(os.curdir)
    dataPath = os.path.join(currentWorkPath, 'NewPaperTools')

    fileNameDict = {'nirTrain': 'NIR_VIS_nirTrain_feas.mat', 'visTrain': 'NIR_VIS_visTrain_feas.mat',
                    'nirProbe': 'NIR_VIS_nirProbe_feas.mat',
                    'visGallery': 'NIR_VIS_visGallery_feas.mat'}

    for tup in fileNameDict.keys():
        fileName = fileNameDict[tup]
        filename = os.path.join(dataPath, fileName)
        'read data'
        data = sio.loadmat(filename)
        if tup == 'nirTrain':
            nirTrain = data['data']
            nirTrainLabel = data['label']
        elif tup == 'visTrain':
            visTrain = data['data']
            visTrainLabel = data['label']
        elif tup == 'nirProbe':
            nirProbe = data['data']
            nirProbeLabel = data['label']
        elif tup == 'visGallery':
            visGallery = data['data']
            visGalleryLabel = data['label']

    nirTrainLabel=nirTrainLabel[0]
    visTrainLabel = visTrainLabel[0]
    nirProbeLabel = nirProbeLabel[0]
    visGalleryLabel = visGalleryLabel[0]

    nirTrain = torch.FloatTensor(nirTrain)
    visTrain = torch.FloatTensor(visTrain)
    nirProbe = torch.FloatTensor(nirProbe)
    visGallery = torch.FloatTensor(visGallery)

    nirTrain = nirTrain.cuda()
    visTrain = visTrain.cuda()
    nirProbe = nirProbe.cuda()
    visGallery = visGallery.cuda()
    print('feature extracting time cost:', time.time() - start_time)

    try:

        Euclidean=False
        if Euclidean==True:
            'training Euclidean'
            print('=========================training Euclidean============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC = np.array(CMC)
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Euclidean'
            print('========================testing Euclidean=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate_Euclidean(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC=np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Eucl_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Eucl_rank1=-1

        #### The similarity calculated by Euclidean or Cosine has the same performance (because xx^T=1 and yy^T=1). Euclidean: (x-y)(x-y)^T=xx^T+yy^T+2xy^T=2+2xy^T  Cosine: xy^T/((xx^T)+yy^T)=xy^T
        Cosine=True

        if Cosine==True:
            'training Cosine'
            print('=========================training Cosine============================')
            print(visTrain.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visTrainLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirTrainLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirTrain[i], nirTrainLabel[i], visTrain, visTrainLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirTrainLabel)  # average CMC
            CMC.numpy()
            # print('train: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('train: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)

            'testing Cosine'
            print('========================testing Cosine=============================')
            print(visGallery.shape)
            start_time = time.time()
            CMC = torch.IntTensor(len(visGalleryLabel)).zero_()
            VR_score_list=[]
            VR_label_list=[]
            for i in range(len(nirProbeLabel)):
                CMC_tmp,VR_score,VR_label = evaluate(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                # VR
                VR_score_list=VR_score_list+VR_score
                VR_label_list = VR_label_list + VR_label
            CMC = CMC.float()
            CMC = CMC / len(nirProbeLabel)  # average CMC
            CMC = np.array(CMC)
            # print('test: Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))

            Cos_rank1=CMC[0]
            # ROC
            ROC_curve(VR_label_list, VR_score_list)
            print('test: Rank@1:%f' % (CMC[0]))
            print('time cost:', time.time() - start_time)
        else:
            Cos_rank1=-1

        'JB classifier'
        print('=======================JB classifier==============================')
        train_feature= torch.cat((nirTrain,visTrain)).cpu()


        nirTrainLabel=np.reshape(nirTrainLabel,[1,-1])
        visTrainLabel = np.reshape(visTrainLabel, [1, -1])
        train_label = np.column_stack((nirTrainLabel, visTrainLabel)).reshape(-1)

        PCAReduction_=False
        PercenOrDim_ = 'Dimension' #PercentageOrDimension = 'Percentage' 'Dimension'
        V_PercentageOrDimension_ = 128

        G,A,n_eigVect=Joint_Bayesianl_Classify_ReID_G_A(train_feature, train_label, PCAReduction=PCAReduction_, OneHot=False,PercenOrDim=PercenOrDim_, V_PercentageOrDimension=V_PercentageOrDimension_)
        G = torch.FloatTensor(G)
        A = torch.FloatTensor(A)
        G=G.cuda()
        A=A.cuda()

        if PCAReduction_==True:
            visGallery = visGallery.cpu()
            visGallery = visGallery.numpy()
            nirProbe = nirProbe.cpu()
            nirProbe = nirProbe.numpy()

            visGallery = (np.mat(visGallery) * n_eigVect)
            nirProbe = (np.mat(nirProbe) * n_eigVect)

            visGallery=np.array(visGallery)
            nirProbe = np.array(nirProbe)

            visGallery=np.real(visGallery)
            nirProbe = np.real(nirProbe)
            visGallery = torch.FloatTensor(visGallery)
            nirProbe = torch.FloatTensor(nirProbe)

            nirProbe = nirProbe.cuda()
            visGallery = visGallery.cuda()


            print('para setting PCAReduction_,PercenOrDim_,V_PercentageOrDimension_',PCAReduction_,PercenOrDim_,V_PercentageOrDimension_)
            print('feature dimention',visGallery.shape[1])


        'JB'
        start_time = time.time()
        print(nirProbe.shape)
        CMC = torch.IntTensor(len(visGalleryLabel)).zero_()

        VR_score_list=[]
        VR_label_list=[]
        # print(query_label)
        for i in range(len(nirProbeLabel)):
            CMC_tmp, VR_score, VR_label = evaluate_JB(nirProbe[i], nirProbeLabel[i], visGallery, visGalleryLabel, G, A)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            # VR
            VR_score_list=VR_score_list+VR_score
            VR_label_list = VR_label_list + VR_label
            # print(i, CMC_tmp[0])
            if i%1000==0 and i>0:
                current_CMC=CMC.float()
                current_CMC=current_CMC/ (i+1)  # average CMC
                current_CMC=np.array(current_CMC)
                print('Current step:',i+1,'/',len(nirProbeLabel),'  rank-1:',current_CMC[0],'  time cost:',time.time() - start_time)
        CMC = CMC.float()
        CMC = CMC / len(nirProbeLabel)  # average CMC
        CMC = np.array(CMC)
        JB_rank1=CMC[0]

        # print('test: JB Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        # ROC
        ROC_curve(VR_label_list, VR_score_list)
        print('test: Rank@1:%f' % (CMC[0]))
    except:
        JB_rank1, Cos_rank1, Eucl_rank1=-1.0,-1.0,-1.0
    return JB_rank1, Cos_rank1


def Nir_Vis_reconstruct_images_BUAA(sess,images_placeholder,reconstImg,phase_train_placeholder,Image_size=160,persentage_save_img=0.1,rootPath='Oulu_NIR_VIS'):


    fileNameDict = {
        'nirTrain': './../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_train/NIR/',
        'visTrain': './../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_train/VIS/',
        'nirProbe': './../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_test/NIR/',
        'visGallery': './../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_test/VIS/'}

    NeutralPath = './../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_train/NIR_VIS_neutralface'

    from datetime import datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    if not os.path.exists('./PlotReconstImg'):
        os.makedirs('./PlotReconstImg')

    rootPath_ = os.path.join('./PlotReconstImg', rootPath)
    if not os.path.exists(rootPath_):
        os.makedirs(rootPath_)

    subdir_ = os.path.join(rootPath_, subdir)
    if not os.path.exists(subdir_):
        os.makedirs(subdir_)

    for tup in fileNameDict.keys():
        A=[]
        tup_dir=fileNameDict[tup]
        class_dirs=get_image_paths(tup_dir)
        class_dirs = sorted(class_dirs)
        for class_dir in class_dirs:
            image_paths = get_image_paths(class_dir)
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                A.append(image_path)

        labs = []
        count = 0
        saveImgNum=int(persentage_save_img*len(A))

        subset_dir = os.path.join(subdir_, tup)

        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

        for i in range(saveImgNum):
            line = A[i]
            # print (line)

            # TODO linux
            line = line.replace('\\', '/')
            path = line
            # a=line.split('/')
            labs.append(int(line.split('/')[-2]))

            img_list = []
            image_size =Image_size
            img = misc.imread(os.path.expanduser(path), mode='RGB')
            aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }

            reconstImg_array = sess.run(reconstImg, feed_dict=feed_dict)
            reconstImg_array = reconstImg_array[0, :, :, :]

            count += 1

            if tup=='nirTrain' or tup=='visTrain':
                NeutralClass_dir = os.path.join(NeutralPath, line.split('/')[-2])
                items=os.listdir(NeutralClass_dir)
                for item in items:
                    if 'VIS' in item:
                        NeutralImg_dir=os.path.join(NeutralClass_dir,item)
                        imgNeutral = misc.imread(os.path.expanduser(NeutralImg_dir), mode='RGB')
                        alignedNeutral = misc.imresize(imgNeutral, (image_size, image_size), interp='bilinear')
                        prewhitenedNeutral = facenet.prewhiten(alignedNeutral)
            else:
                prewhitenedNeutral=np.zeros([image_size,image_size,3])




            prewhitened=cv2.normalize(prewhitened,prewhitened,0,255,cv2.NORM_MINMAX)
            reconstImg_array=cv2.normalize(reconstImg_array,reconstImg_array,0,255,cv2.NORM_MINMAX)
            prewhitenedNeutral=cv2.normalize(prewhitenedNeutral,prewhitenedNeutral,0,255,cv2.NORM_MINMAX)

            r,g,b=cv2.split(prewhitened)
            prewhitened=cv2.merge([b,g,r])
            r,g,b=cv2.split(reconstImg_array)
            reconstImg_array=cv2.merge([b,g,r])
            r,g,b=cv2.split(prewhitenedNeutral)
            prewhitenedNeutral=cv2.merge([b,g,r])

            img_concat = np.hstack((prewhitened, reconstImg_array,prewhitenedNeutral))
            fileName=line.split('/')[-4]+'_'+line.split('/')[-3]+'_'+line.split('/')[-2]+'_'+line.split('/')[-1]
            ImgSave_dir = os.path.join(subset_dir, fileName)

            cv2.imwrite(ImgSave_dir, img_concat)




def Joint_Bayesianl_Classify_ReID_G_A(train_fea,train_labels, PCAReduction=False, OneHot=False,PercenOrDim='Percentage',V_PercentageOrDimension=0.96):
    if PCAReduction == True:
        dataMat = train_fea

        print('PCA 降维 procedure ing...')
        lowDDataMat, n_eigVect = pca_reduction(dataMat, valueOfPercentageOrDimension=V_PercentageOrDimension, ColAsSample=False, PercentageOrDimension=PercenOrDim) #PercentageOrDimension = 'Percentage' 'Dimension'

        train_fea=lowDDataMat
        train_fea = np.real(np.array(train_fea))
    else:
        n_eigVect=np.mat(np.eye(train_fea.shape[1]))

    train_fea = np.mat(train_fea).T
    temp =[]
    for i in range(train_labels.shape[0]):
        val = train_labels[i]
        temp.append(val)
    'order label'
    settemp = np.array(list(set(temp)))
    for k in range(len(temp)):
        temp[k] = list(np.where(temp[k] == settemp)[0])[0]
    train_labels = np.mat(temp)


    # trainFeature, testFeature = JB.normalization(galleryFea, probeFea)

    G, A = classify_ReID_G_A(train_fea,train_labels)
    return G, A, n_eigVect


def classify_ReID_G_A(train_fea,train_labels):
    train_fea = np.mat(train_fea)

    Sw, Sb = scatter_mat(train_fea, train_labels.getA())
    G, A = Su_Se(Sw, Sb, train_fea, train_labels.getA())

    return G, A




def normalization(trainFeature, testFeature):  # centering
    m_trainFeature = np.mat(np.tile(np.mean(trainFeature, axis=1), (1, trainFeature.shape[1])))  # 均值矩阵
    trainFeature = np.mat(trainFeature) - m_trainFeature
    m_testFeature = np.mat(np.tile(np.mean(testFeature, axis=1), (1, testFeature.shape[1])))  # 均值矩阵
    testFeature = np.mat(testFeature) - m_testFeature
    return trainFeature, testFeature


def scatter_mat(trainFeature,
                labelTrain):  # 求解类内离散度矩阵Sw 和 类间离散度矩阵Sb； trainFeature(d x n矩阵)：列代表样本  labelTrain(1 x n矩阵)：存储样本类标

    # Function - -实现类内、类间、混合散布矩阵距离计算
    # trainFeature - -多类构成的样本集合（一个列向量表示一个样本）
    # labelTrain - -一个N维行向量，第i个元素包含trainFeature中第i个向量的label（总共有c个类标）
    # Sw - -类内散布矩阵，类内距离的平方形式
    # Sb - -类间散布矩阵，类间距离的平方形式
    # St - -混合散布矩阵
    print('Joint Bayesian procedure ing...')
    [L, N] = np.shape(trainFeature)  # 设X有L * N维
    c = labelTrain.max()
    # Sw
    m = np.mat(np.zeros([L, c]))  # np.mat(np.zeros(L, c))
    P = np.array([])
    Sw = np.array([0])
    for i in range(1, c + 1):
        y_temp = np.argwhere(labelTrain == i)[:, 1]
        X_temp = trainFeature[:, y_temp]
        P = np.concatenate((P, np.array([len(y_temp)]) / N),
                           axis=0)  # P = np.concatenate((P, np.array([len(y_temp) / N])), axis=0)  #or:   P = np.append(P, np.array([np.sum(y_temp) / N]), axis=0)
        a = np.mean(np.array(X_temp), axis=1)
        m[:, i - 1] = np.mat(np.mean(X_temp, axis=1))  # m = np.append(m, np.array([np.mean(X_temp, axis=1)]).T)
        m_Xtemp = np.mat(np.tile(np.mean(X_temp, axis=1), (1, X_temp.shape[1])))  # 均值矩阵
        X_temp = X_temp - m_Xtemp
        Sw = Sw + X_temp * X_temp.T  # Sw = Sw + np.array(np.cov(X_temp, rowvar=1))  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    Sw = Sw / N
    # Sb
    m0 = np.mat(np.mean(trainFeature,
                        axis=1))  # m0 = np.mat(np.tile(np.mean(trainFeature, axis=1), (1, m.shape[1])))  # m0：平均人脸矩阵
    Sb = np.array([0])
    P = np.mat(P)
    for i in range(c):
        # Sb = Sb + P[0, i] * (m[:, i]-m0)*(m[:, i]-m0).T  # Sb + P(i) * ((m(:, i)-m0)*(m(:, i)-m0)');  %矩阵形式
        Sb = Sb + P[0, i] * m[:, i] * m[:, i].T
    # St
    St = Sw + Sb  # 矩阵形式
    return Sw, Sb


def forced_reversible(mat):  # 引入矩阵对角元素微调量，防止mat不可逆
    constant = np.max(mat) / mat.shape[1]  # np.abs(np.max(mat)/mat.shape[1])
    const = 0
    while True:
        try:
            mat.I
        except:
            const += constant
            mat = np.mat(mat + np.eye(mat.shape[1]) * const)  # 引入对角常量，使得矩阵Se可逆
        else:
            break
    return mat


def Su_Se(Sw, Sb, trainFeature, labelTrain):  # trainFeature（d x n矩阵）：列代表样本  labelTrain（1 x n矩阵）：存储样本类标 Su: identity covariance matrixes   Se: within the same identity covariance matrixes
    [L, N] = np.shape(trainFeature)  # 设X有L * N维
    c = labelTrain.max()
    Su = np.mat(Sb)  # 类间离散度矩阵Sb,初始化Su
    Se = np.mat(Sw)  # 类内离散度矩阵Sw,初始化Se`
    U = np.mat(np.zeros([L, c]))
    E = np.mat(np.zeros([L, N]))
    print('Joint Bayesian: iteration number ing...' )
    for k in range(20):  # EM算法迭代次数
        # print('Joint Bayesian: iteration number %d ing...' % (k + 1))
        cnt = 0
        F = Se.I  # np.linalg.pinv(Se) # F = np.linalg.inv(Se)  # F = Se.I
        SF = Su * F
        for i in range(1, c + 1):  # 一次迭代求解E U 步骤  U:身份  E：差异
            y_temp = np.argwhere(labelTrain == i)[:, 1]
            X_temp = np.mat(trainFeature[:, y_temp])
            subjectNum = len(y_temp)
            G = -((subjectNum + 1) * Su + Se).I * SF  # 防止(subjectNum*Su+Se)不可逆
            cnt = cnt + subjectNum

            x_sum = np.sum(X_temp, axis=1)  # 均值矩阵 x_mean = np.mat(np.sum(np.array(X_temp), axis=1)).T/subjectNum
            ui = Su * (F + (subjectNum + 1) * G) * x_sum
            E[:, cnt - subjectNum:cnt] = X_temp + Se * G * np.mat(
                np.tile(np.array(x_sum), (1, subjectNum)))  # 构造1 X subjectNum 个 copy
            U[:, i - 1] = ui

        Su = 1 / c * U * U.T  # Su = 1/N*U*U.T
        Se = 1 / N * E * E.T  # Se = 1 / N * E * E.T

    # Su = np.mat(Sb) #类间离散度矩阵Sb,初始化Su
    # Se = np.mat(Sw) #类内离散度矩阵Sw,初始化Se
    F = Se.I
    G = -(2 * Su + Se).I * Su * Se.I  # 防止(subjectNum*Su+Se)不可逆
    A = (Su + Se).I - ((Su + Se) - Su * (Su + Se).I * Su).I
    return G, A


def judgeFace(G, A, judgeFea, trainFeature):
    minIndex = 0
    minVals = np.inf
    for i in range(np.shape(trainFeature)[1]):  # 获取列数，shape(LBPoperator)[1]
        trainFea = trainFeature[:, i]
        similarity = judgeFea.T * A * judgeFea + trainFea.T * A * trainFea - 2 * judgeFea.T * G * trainFea
        diff = float(np.real(-similarity))
        if diff < minVals:
            minIndex = i
            minVals = diff
    return minIndex


def judgeFace_ROC(G, A, judgeFea, trainFeature, judgeFeaLabel, trainFeatureLabels):
    minIndex = 0
    minVals = np.inf

    Probe_dis_List = []
    Probe_pairLabel_List = []
    for i in range(np.shape(trainFeature)[1]):  # 获取列数，shape(LBPoperator)[1]
        trainFea = trainFeature[:, i]
        similarity = judgeFea.T * A * judgeFea + trainFea.T * A * trainFea - 2 * judgeFea.T * G * trainFea
        diff = float(np.real(-similarity))
        if diff < minVals:
            minIndex = i
            minVals = diff

        # ROC
        Probe_dis_List.append(-diff)
        if trainFeatureLabels[0, i] == judgeFeaLabel:
            Probe_pairLabel_List.append(1)
        else:
            Probe_pairLabel_List.append(0)
    return minIndex, Probe_dis_List, Probe_pairLabel_List

'PCA'
def zeroMean(dataMat):# 零均值化
    meanVal = np.mean(dataMat, axis=1)  # axis=0按列求均值，axis=1按行求均值,即求各个特征的均值
    newData = dataMat - np.tile(meanVal, (1, dataMat.shape[1])) # newData = dataMat - meanVal
    return newData, meanVal


def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca_dimension(dataMat, n,ColAsSample=True): # PCA 维度
    dataMat = np.mat(dataMat.numpy())
    if ColAsSample==False:
        dataMat = np.mat(dataMat).T
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=1)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = (newData.T * n_eigVect).T  # 低维特征空间的数据
    reconMat = (n_eigVect*lowDDataMat)+meanVal  # 重构数据
    if ColAsSample==False:
        lowDDataMat = np.mat(lowDDataMat).T
        lowDDataMat = np.array(lowDDataMat)
        reconMat = np.mat(reconMat).T
        reconMat = np.array(reconMat)
    return lowDDataMat, n_eigVect

def pca_percentage(dataMat,percentage=0.99,ColAsSample=True): # PCA 百分比  ColAsSample=True 列代表样本，ColAsSample=False 行代表样本
    dataMat = np.mat(dataMat.numpy())
    if ColAsSample==False:
        dataMat = np.mat(dataMat).T
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData, rowvar=1)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=(newData.T * n_eigVect).T               #低维特征空间的数据
    reconMat = (n_eigVect*lowDDataMat)+meanVal  #重构数据
    if ColAsSample==False:
        lowDDataMat = np.mat(lowDDataMat).T
        lowDDataMat = np.array(lowDDataMat)
        reconMat = np.mat(reconMat).T
        reconMat = np.array(reconMat)
    return lowDDataMat, n_eigVect

def pca_reduction(dataMat,valueOfPercentageOrDimension,ColAsSample=True, PercentageOrDimension = 'Percentage'):
    if PercentageOrDimension == 'Percentage':
        lowDDataMat, reconMat = pca_percentage(dataMat, percentage=valueOfPercentageOrDimension, ColAsSample=ColAsSample)
    elif PercentageOrDimension == 'Dimension':
        lowDDataMat, reconMat = pca_dimension(dataMat, n=valueOfPercentageOrDimension, ColAsSample=ColAsSample)
    return lowDDataMat, reconMat




#######################################################################
# Evaluate
def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()


    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index=np.reshape(query_index,[-1])

    CMC_tmp = compute_mAP(index, good_index)

    # VR score and label
    VR_index=np.array(range(0,len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label=list(VR_label)
    VR_score=list(score)

    return CMC_tmp,VR_score,VR_label

def evaluate_EuclideanOld(qf, ql, gf, gl):

    score = torch.FloatTensor(len(gl), 1).zero_()
    qf = qf.reshape([1, -1])

    for idx in range(len(gl)):
        gf_sample = gf[idx, :]

        gf_sample = gf_sample.reshape([1, -1])

        # gf_sample_T=gf_sample.reshape([-1, 1])

        s=torch.sqrt(torch.sum((qf-gf_sample)**2,1))
        # s = torch.sum((qf - gf_sample) ** 2, 1)

        score[idx, 0] = -s

    score = score.squeeze(1).cpu()
    score = score.numpy()



    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index=np.reshape(query_index,[-1])

    CMC_tmp = compute_mAP(index, good_index)

    # VR score and label
    VR_index=np.array(range(0,len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label=list(VR_label)
    VR_score=list(score)

    return CMC_tmp,VR_score,VR_label

def evaluate_Euclidean(qf, ql, gf, gl):

    score = torch.FloatTensor(len(gl), 1).zero_()
    qf = qf.reshape([1, -1])

    for idx in range(len(gl)):
        gf_sample = gf[idx, :]

        gf_sample = gf_sample.reshape([1, -1])

        # gf_sample_T=gf_sample.reshape([-1, 1])

        s=torch.sqrt(torch.sum((qf-gf_sample)**2,1))
        # gf_sample = gf_sample.reshape([1, -1])
        #
        # gap=(qf - gf_sample)
        # s=torch.mm(gap,gap.transpose(0, 1))

        # gf_sample_T=gf_sample.reshape([-1, 1])

        # s=torch.sqrt(torch.sum((qf-gf_sample)**2,1))
        # s = torch.sum((qf - gf_sample) ** 2, 1)

        score[idx, 0] = s

    score = score.squeeze(1).cpu()
    score = score.numpy()


    # predict index
    index = np.argsort(score)  # from small to large
    # index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index=np.reshape(query_index,[-1])

    CMC_tmp = compute_mAP(index, good_index)

    # VR score and label
    VR_index=np.array(range(0,len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label=list(VR_label)
    VR_score=list(-score)#Roc curve

    return CMC_tmp,VR_score,VR_label

def evaluate_JB(qf, ql, gf, gl, G, A):

    gf_T_Tmp = gf.transpose(0, 1)  # each col as a sample

    qf_sample = qf.reshape([-1, 1])

    qf_sample_T = qf_sample.reshape([1, -1])
    score_Tmp = torch.mm(torch.mm(qf_sample_T, A), qf_sample).expand(1, gf.shape[0]) + torch.diag(
        torch.mm(torch.mm(gf, A), gf_T_Tmp), diagonal=0) - torch.mm(
        torch.mm(qf_sample_T, G), gf_T_Tmp).mul(2.0)

    score_Tmp = score_Tmp.reshape([-1, 1])
    score_Tmp = score_Tmp.squeeze(1).cpu()
    score = score_Tmp.numpy()


    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index = np.reshape(query_index, [-1])

    CMC_tmp = compute_mAP(index, good_index)

    # VR score and label
    VR_index = np.array(range(0, len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label = list(VR_label)
    VR_score = list(score)

    return CMC_tmp, VR_score, VR_label

def evaluate_JB_slow(qf, ql, gf, gl, G, A):
    score = torch.FloatTensor(len(gl), 1).zero_()

    for idx in range(len(gl)):
        gf_sample = gf[idx, :]

        gf_sample = gf_sample.reshape([-1, 1])
        gf_sample_T = gf_sample.reshape([1, -1])

        qf_sample = qf.reshape([-1, 1])
        qf_sample_T = qf_sample.reshape([1, -1])

        s = torch.mm(torch.mm(gf_sample_T, A), gf_sample) + torch.mm(torch.mm(qf_sample_T, A), qf_sample) - torch.mm(
            torch.mm(qf_sample_T, G), gf_sample).mul(2.0)
        score[idx, 0] = s

    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index = np.reshape(query_index, [-1])

    CMC_tmp = compute_mAP(index, good_index)

    # VR score and label
    VR_index = np.array(range(0, len(gl)))
    VR_label = np.in1d(VR_index, good_index)
    VR_label = list(VR_label)
    VR_score = list(score)

    return CMC_tmp, VR_score, VR_label


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # find good_index index
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1

    return cmc

def ROC_curve_ori(VR_label_list, VR_score_list):

    fpr, tpr, threshold = roc_curve(VR_label_list, VR_score_list)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    for idx, val in enumerate(fpr):
        if val > 0.000001:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.00001:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            val_0_00001 = tpr[idx]
            break
    for idx, val in enumerate(fpr):
        if val > 0.0001:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            val_0_0001 = tpr[idx]
            break
    for idx, val in enumerate(fpr):
        if val > 0.001:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            val_0_001 = tpr[idx]
            break
    for idx, val in enumerate(fpr):
        if val > 0.01:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            val_0_01 = tpr[idx]
            break
    for idx, val in enumerate(fpr):
        if val > 0.1:
            print('fpr,tpr', fpr[idx], tpr[idx],threshold[idx])
            break

def ROC_curve(VR_label_list, VR_score_list):

    fpr, tpr, threshold = roc_curve(VR_label_list, VR_score_list)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    flag1,flag2,flag3,flag4,flag5,flag6=True,True,True,True,True,True

    for idx, val in enumerate(fpr):
        if val > 0.000001 and flag1==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag1=False

        if val > 0.00001 and flag2==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag2 = False

        if val > 0.0001 and flag3==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag3 = False

        if val > 0.001 and flag4==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag4 = False

        if val > 0.01 and flag5==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag5 = False

        if val > 0.1 and flag6==True:
            print('fpr,tpr', fpr[idx], tpr[idx])
            flag6 = False




if __name__ == '__main__':
    pass

