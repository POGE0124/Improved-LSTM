function RunLstm()
clc
close all
clear all
% % 绘制Error-Cost曲线图
% load('ErrorCost3.mat');
% for n=1:1:length(ErrorCost3)
%     semilogy(n,ErrorCost3(1,n),'*');
%     hold on;
%     axis([0 3000 0.01 100]);
%     title('Improved Error-Cost');  
%     xlabel('Training number');
%     ylabel('Training error ');
% end
numdely=9;
cell_num=5;
cost_gate=0.01;
%% 数据加载，并归一化处理
[train_data_norm,train_data]=LSTM_data_process(numdely);
data_length=size(train_data_norm,1)-1;
data_num=size(train_data_norm,2);
%% 网络参数初始化
input_num=data_length;
output_num=1;
ab=20;
n=input('请输入1：离线建模、2：在线预测:');
switch (n)
    case 1
% 网络中门的偏置
para.bias_input_gate=rand(1,cell_num)/ab;
para.bias_forget_gate=rand(1,cell_num)/ab;
para.bias_output_gate=rand(1,cell_num)/ab;
%网络权重初始化
para.weight_input_x=rand(input_num,cell_num)/ab;
para.weight_input_h=rand(output_num,cell_num)/ab;
para.weight_inputgate_x=rand(input_num,cell_num)/ab;
para.weight_inputgate_c=rand(cell_num,cell_num)/ab;
para.weight_forgetgate_x=rand(input_num,cell_num)/ab;
para.weight_forgetgate_c=rand(cell_num,cell_num)/ab;
para.weight_outputgate_x=rand(input_num,cell_num)/ab;
para.weight_outputgate_c=rand(cell_num,cell_num)/ab;
%hidden_output权重
para.weight_preh_h=rand(cell_num,output_num)/ab;
%网络状态初始化
% cost_gate=0.25;
para.h_state=rand(output_num,data_num)/ab;
para.cell_state=rand(cell_num,data_num)/ab;
tic
%% 网络训练学习
for iter=1:3000
    yita=0.01;            %每次迭代权重调整比例%1/(10+sqrt(iter))
    for m=1:data_num
        %前馈部分
        if(m==1)
            input_cell_state=tanh(train_data_norm(1:input_num,m)'*para.weight_input_x);
            input_gate_input=train_data_norm(1:input_num,m)'*para.weight_inputgate_x+para.bias_input_gate;
            output_gate_input=train_data_norm(1:input_num,m)'*para.weight_outputgate_x+para.bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            para.cell_state(:,m)=(input_gate.*input_cell_state)';
        else
            input_cell_state=tanh(train_data_norm(1:input_num,m)'*para.weight_input_x+para.h_state(:,m-1)'*para.weight_input_h);
            input_gate_input=train_data_norm(1:input_num,m)'*para.weight_inputgate_x+para.cell_state(:,m-1)'*para.weight_inputgate_c+para.bias_input_gate;
            forget_gate_input=train_data_norm(1:input_num,m)'*para.weight_forgetgate_x+para.cell_state(:,m-1)'*para.weight_forgetgate_c+para.bias_forget_gate;
            output_gate_input=train_data_norm(1:input_num,m)'*para.weight_outputgate_x+para.cell_state(:,m-1)'*para.weight_outputgate_c+para.bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            para.cell_state(:,m)=(input_gate.*input_cell_state+para.cell_state(:,m-1)'.*forget_gate)';   
        end
        para.pre_h_state=tanh(para.cell_state(:,m)').*output_gate;
        para.h_state(:,m)=(para.pre_h_state*para.weight_preh_h)'; 
    end
    % 误差的计算
    Error=para.h_state(:,:)-train_data_norm(end,:);
    para.Error_Cost(1,iter)=sum(Error.^2);
    if para.Error_Cost(1,iter) < cost_gate
            iter
        break;
     end
    if iter>1 && (para.Error_Cost(1,iter-1) < para.Error_Cost(1,iter))
            iter
            para.Error_Cost(1,iter)
        break;
    end
                [ para.weight_input_x,...
                para.weight_input_h,...
                para.weight_inputgate_x,...
                para.weight_inputgate_c,...
                para.weight_forgetgate_x,...
                para.weight_forgetgate_c,...
                para.weight_outputgate_x,...
                para.weight_outputgate_c,...
                para.weight_preh_h ]=LSTM_updata_weight(m,yita,Error,...
                                                   para.weight_input_x,...
                                                   para.weight_input_h,...
                                                   para.weight_inputgate_x,...
                                                   para.weight_inputgate_c,...
                                                   para.weight_forgetgate_x,...
                                                   para.weight_forgetgate_c,...
                                                   para.weight_outputgate_x,...
                                                   para.weight_outputgate_c,...
                                                   para.weight_preh_h,...
                                                   para.cell_state,para.h_state,...
                                                   input_gate,forget_gate,...
                                                   output_gate,input_cell_state,...
                                                   train_data_norm,para.pre_h_state,...
                                                   input_gate_input,...
                                                   output_gate_input,...
                                                   forget_gate_input);


end
toc

%% 保存参数
save para
% 绘制Error-Cost曲线图
for n=1:1:iter
    semilogy(n,para.Error_Cost(1,n),'*');
    hold on;
    title('Error-Cost');  
    xlabel('Training number');
    ylabel('Training error ');
end
% ErrorCost3=para.Error_Cost;
% save ErrorCost3 ErrorCost3
%% 数据检验
    case 2
load para
%数据加载
for i=1:size(train_data,2)
test_data=train_data(:,i);
test_norm=test_data;
test_norm=test_norm/sqrt(sum(test_norm.^2));
reverse_norm = sqrt(sum(test_data.^2));
%前馈
m=data_num;
input_cell_state=tanh(test_norm(1:input_num)'*para.weight_input_x+para.h_state(:,m-1)'*para.weight_input_h);
input_gate_input=test_norm(1:input_num)'*para.weight_inputgate_x+para.cell_state(:,m-1)'*para.weight_inputgate_c+para.bias_input_gate;
forget_gate_input=test_norm(1:input_num)'*para.weight_forgetgate_x+para.cell_state(:,m-1)'*para.weight_forgetgate_c+para.bias_forget_gate;
output_gate_input=test_norm(1:input_num)'*para.weight_outputgate_x+para.cell_state(:,m-1)'*para.weight_outputgate_c+para.bias_output_gate;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
para.cell_state_test=(input_gate.*input_cell_state+para.cell_state(:,m-1)'.*forget_gate)';
para.pre_h_state=tanh(para.cell_state_test').*output_gate;
para.h_state_test=(para.pre_h_state*para.weight_preh_h)'* reverse_norm;

fprintf('----The %dth actual result is %f----\n' ,i,train_data(end,i));
fprintf('----The %dth prediction result is %f----\n' ,i,para.h_state_test);
prediction(i)=para.h_state_test;
end
prdiction=[train_data(1:size(train_data,1)-1,1)',prediction];
actual=train_data(1,:);
%% AR模型参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data2.mat');
b=a(1:100);
x=b';
y=iddata([x,0]'); %转化为系统辨识工具箱能识别的数据类型
AR1=ar(x,2);%估计参数，AR1同y一样是一个结构数组
for i=100:168
p2=predict(y(1:i),AR1,9);%1步预测
xp2=p2.OutputData;
x=[x xp2(end)];
y=iddata([x,0]'); %转化为系统辨识工具箱能识别的数据类型
end
ar_result=x(11:168);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%prediction=prediction%+para.Error_Cost(1,end);%error=sum((actual-prediction)/length(prediction));%
plot(actual*100,'m','Linewidth',1.5)
hold on
plot(ar_result*100,'b:','Linewidth',1.5)
hold on
load ('bp158.mat')
plot(bp*100,'r--','Linewidth',1.5)
hold on
load('lstm_result.mat')
plot(lstm_result*100,'g-.','Linewidth',1.5)
hold on
plot(prediction*100,'k-*','Linewidth',0.3)

xlabel('Number of Cycle')
ylabel('SOH(%)')
title('Prediction of various algorithms')
h=legend('The Actual SOH','Prognostics Based on AR','Prognostics Based on NN','Prognostics Based on LSTM','Prognostics Based on Improved LSTM','location','SouthWest');
set(h,'Box','on');
set(h,'Fontsize',10);
%% 误差对比图
figure
plot((actual-ar_result).*(actual-ar_result),'b:','Linewidth',1.5)
aa=(actual-ar_result).*(actual-ar_result);
sum(aa)
hold on
plot((actual-bp).*(actual-bp),'r--','Linewidth',1.5)
sum((actual-bp).*(actual-bp))
hold on
plot((actual-lstm_result).*(actual-lstm_result),'g-.','Linewidth',1.5)
sum((actual-lstm_result).*(actual-lstm_result))
hold on
plot((actual-prediction).*(actual-prediction),'k-*','Linewidth',0.3)
sum((actual-prediction).*(actual-prediction))
% title('MSE of Various Methods');
xlabel('Number of Cycle');
ylabel('MSE');
title('MSE of various algorithms');
hold on
h=legend('MSE Based on AR','MSE Based on NN','MSE Based on LSTM','MSE Based on Improved LSTM','location','NorthEast');
set(h,'Box','on');
set(h,'Fontsize',10);
axis([0 160 -0.001 0.04]);



end
