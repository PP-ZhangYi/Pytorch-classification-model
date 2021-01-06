# Pytorch-classification-model
Based on Pytroch framework, the classification model is created

1. 神經網絡模型
腳本可以選擇的網絡有：Mobilenet_v2, Googlenet, Inception_v3, resnet50, Densnet121。 當然也可以添加你自己的網絡。

 

2.模型訓練
a. 數據路徑下包含train和val兩個文件夾，文件夾下面存放所有類別的數據，一個類別一個文件夾。

b. resize是圖片壓縮尺寸，crop-size是圖片中心剪裁后輸入網絡的尺寸。

c. 如果要加載訓練過的模型，開啟pre，并設置model-path模型路徑。

