注： DATA.mat 存放的是 金融标的物每天的价格

在Matlab 中把文件夹加入工作路径
在命令行中输入  RunLstm(numdely,cell_num，cost_gate)即可

其中：
numdely  是选择预测点的数目
cell_num 是隐含层的结点数目 
cost_gate是误差的阈值（此处一般取0.25）

example: RunLstm(9,5,0.25)
