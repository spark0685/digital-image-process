%% �������ͼ���񻯺��� mySharpen.m��Ȼ�󽫱��������еı���
%% filename���á�moon��������main_sharpen.m�����ɵõ�ͼ���񻯺�Ľ��
%% EEIS Department, USTC
clc; 
clear;
close all;

%% ��ȡͼƬ
filename = 'moon'; %����ͼ��1
im = imread([filename, '.jpg']);

%% ��ͼ�������
im_s = mySharpen(im);

%% ��������浽��ǰĿ¼�µ�result�ļ�����
imwrite(im_s, sprintf('result/_%s_s.jpg', filename)); 

%% ��ʾ���
figure(1); 
subplot(121); imshow(im); title('ԭͼ'); axis on
subplot(122); imshow(im_s); title('ͼ����'); axis on

