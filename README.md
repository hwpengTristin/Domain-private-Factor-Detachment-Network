# Domain-private-Factor-Detachment-Network
We propose a novel Domain-private Factor Detachment (DFD) network to disentangle domain-dependent factors and achieve identity information distillation. Our approach consists of three key components, including Domain-identity Representation Learning (DiRL), Cross-domain Factor Detachment (CdFD) and Cross-domain Aggregation Learning (CAL).

# DFD代码说明

环境：python 3.6; tensorflow 1.3.0
# # 1.	测试与测试一体
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
