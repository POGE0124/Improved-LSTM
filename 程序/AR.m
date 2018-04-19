clear all
close all
clc
load('data2.mat');
b=a(1:100);
x=b';
y=iddata([x,0]'); %转化为系统辨识工具箱能识别的数据类型
AR1=ar(x,2);%估计参数，AR1同y一样是一个结构数组
for i=101:168
p2=predict(y(1:i),AR1,1);%1步预测
xp2=p2.OutputData;
x=[x xp2(end)];
y=iddata([x,0]'); %转化为系统辨识工具箱能识别的数据类型
end
x=x(1:168);
plot(x)
save ar x