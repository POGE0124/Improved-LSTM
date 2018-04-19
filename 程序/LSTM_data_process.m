function [train_data_norm,train_data]=LSTM_data_process(numdely)
load('data2.mat');
numdata = size(a,1);
numsample = numdata - numdely - 1;
train_data_norm = zeros(numdely+1, numsample);
for i = 1 :numsample
    train_data_norm(:,i) = a(i:i+numdely)';
end     
data_num=size(train_data_norm,2);  
train_data=train_data_norm;
%%归一化过程
for n=1:data_num
    train_data_norm(:,n)=train_data_norm(:,n)/sqrt(sum(train_data_norm(:,n).^2));  
end
