# Domain-private-Factor-Detachment-Network
We propose a novel Domain-private Factor Detachment (DFD) network to disentangle domain-dependent factors and achieve identity information distillation. Our approach consists of three key components, including Domain-identity Representation Learning (DiRL), Cross-domain Factor Detachment (CdFD) and Cross-domain Aggregation Learning (CAL). Firstly, the proposed DiRL aims to achieve domain-specific information distillation and learn identity-related representations. Specifically, three sub-networks, i.e., NIR sub-Network (NIR-Net), VIS sub-Network (VIS-Net) and IDentity-dependent sub-Network (ID-Net) are designed to learn NIR facial representations, VIS facial representations and identity-dependent representations, respectively, and they can promote each other to facilitate the learning of identity-discriminative representations. Secondly, considering that the entangled modal components in face representations negatively affect the subsequent matching process, to reduce modality-related components, we model the cross-modal face matching problem into three parts, comprising Identity Variation (IV), Inter-Spectrum Variation (ISV) and Identity-Domain Variation (IDV). The CdFD is presented to eliminate ISV components and IDV components by introducing inter-spectrum invariant constraint and identity-domain invariant constraint, so that cross-modal face recognition can be performed under pure identity information differences without modal interference. Finally, the CAL is developed to learn modality-invariant yet discriminative representations by exploring within-class aggregation, negative pair separability and cross-domain positive pair compactness. Experimental results on multiple challenging NIR-VIS face databases demonstrate the effectiveness of the DFD approach.

# DFD代码说明

环境：python 3.6; tensorflow 1.3.0
# # 1. 训练与测试一体
运行指令：python DFD_method.py
实验结果：
CASIA NIR-VIS 2.0（first fold）数据集，测试结果Cosine distance度量的Rank-1 准确率为99.4%左右。如下截图： 

![image](https://user-images.githubusercontent.com/29362830/212851718-7a860439-78dd-44ba-ada3-0876772db1ed.png)
![image](https://user-images.githubusercontent.com/29362830/212851795-f5703327-b4d1-4713-8e88-4ca2eea145f2.png)

# # 2.	文件夹内容说明
----DFD
	----Experiments_Results (文件夹说明：实验结果的log记录)
	----src (文件夹说明：主要程序目录)
----requirements.txt (文件说明：跑该代码时的python库环境，忽视不必要的库)
		----logs (文件夹说明：跑代码过程产生的log记录保存路径)
		----models (文件夹说明：跑代码过程保存的模型路径)
----Resnet50_CBAM_xxx_20201013 (文件夹说明：加载的预训练模型)
			----models.py (文件说明：网络结构定义)
		----DFD_method.py (文件夹说明：main函数，训练和测试一体的main函数)
		----facenet.py (文件夹说明：main函数中调用的部分函数，在此文件定义)
		----lfw.py (文件夹说明：lfw测试代码，在此文件定义)
----NewPaper_validate_on_CASIA_NIR_VIS_2_0_Rank_1_speedup.py (文件夹说明：跨模态数据库测试代码，在此文件定义)

# # 3.	数据集路径说明

# # # CASIA NIR-VIS 2.0数据集，包含10-fold实验
CASIA NIR-VIS 2.0的first fold训练集路径: 
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/vis_train_1.txt/
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/nir_train_1.txt/
CASIA NIR-VIS 2.0的first fold测试集路径: 
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/vis_gallery_1.txt/
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/nir_probe_1.txt/

CASIA NIR-VIS 2.0的ten fold训练集路径: 
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/vis_train_10.txt/
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/nir_train_10.txt/
CASIA NIR-VIS 2.0的ted fold测试集路径: 
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/vis_gallery_10.txt/
./../../Datasets/CASIA NIR-VIS 2.0/NIR-VIS-2.0/protocols/nir_probe_10.txt/

# # # Oulu-CASIA NIR-VIS数据集
Oulu-CASIA NIR-VIS训练集路径: 
./../../Datasets/Oulu_40Class/Oulu_20Class_train/
Oulu-CASIA NIR-VIS测试集路径: 
./../../Datasets/Oulu_40Class/Oulu_20Class_test/

# # # BUAA NIR-VIS数据集
BUAA NIR-VIS训练集路径: 
./../../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_train/
BUAA NIR-VIS测试集路径: 
./../../Datasets/BUAA_VISNIR_train_test/BUAA_VISNIR_test/

# # # LAMP-HQ NIR-VIS数据集，包含10-fold实验
LAMP-HQ NIR-VIS的first fold训练集路径: 
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/train/train_vis1.txt/
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/train/train_nir1.txt/
LAMP-HQ NIR-VIS的first fold测试集路径: 
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/test/gallery_vis1.txt/
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/test/probe_nir1.txt/

LAMP-HQ NIR-VIS的ten fold训练集路径: 
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/train/train_vis10.txt/
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/train/train_nir10.txt/
LAMP-HQ NIR-VIS的ten fold测试集路径: 
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/test/gallery_vis10.txt/
./../../Datasets/ LAMP_HQ_NIR_VIS /LAMP_HQ/Protocols/test/probe_nir10.txt/

# Reference
[1] Weipeng Hu, Haifeng Hu, Domain-private Factor Detachment Network for NIR-VIS Face Recognition, IEEE Trans. on Information Forensics and Security, vol. 17, pp. 1435-1449, 2022. DOI:  10.1109/TIFS.2022.3160612

# Some related works
[1] Weipeng Hu, Wenjun Yan, Haifeng Hu, Dual Face Alignment Learning Network for NIR-VIS Face Recognition, IEEE Trans. on Circuits and Systems for Video Technology, vol.32, no.4, pp.2411-2424, 2022. DOI:  10.1109/TCSVT.2021.3081514
[2] Weipeng Hu, Haifeng Hu, Dual Adversarial Disentanglement and Deep Representation Decorrelation for NIR-VIS Face Recognition, IEEE Trans. on Information Forensics and Security, vol.16, no.1, pp.70-85, 2020. DOI: 10.1109/TIFS.2020.3005314


# Note
Part of our code is based on Github's open source project (https://github.com/davidsandberg/facenet).



