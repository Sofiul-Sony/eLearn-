# Generated with SMOP  0.41-beta
from libsmop import *
# 

    
@function
def test_learn(*args,**kwargs):
    varargin = test_learn.varargin
    nargin = test_learn.nargin

    # test learn++
    K=5
# test_learn.m:3
    addpath('../src/')
    
    load('ionosphere')
    u=unique(Y)
# test_learn.m:7
    
    labels=zeros(numel(Y),1)
# test_learn.m:8
    # convert the string labels to numeric labels
    for n in arange(1,numel(Y)).reshape(-1):
        for c in arange(1,numel(u)).reshape(-1):
            if u[c] == Y[n]:
                labels[n]=c
# test_learn.m:14
                break
    
    # shuffle the data
    i=randperm(numel(Y))
# test_learn.m:21
    data=X(i,arange())
# test_learn.m:22
    labels=labels(i)
# test_learn.m:23
    clear('Description','X','Y','c','i','n','u')
    cv=cvpartition(numel(labels),'k',K)
# test_learn.m:26
    z=zeros(numel(labels),1)
# test_learn.m:27
    for k in arange(1,K - 1).reshape(-1):
        z=z + (training(cv,k) > 0)
# test_learn.m:29
    
    ts_idx=find(z == K - 1)
# test_learn.m:31
    tr_idx=find(z != K - 1)
# test_learn.m:32
    data_tr=data(tr_idx,arange())
# test_learn.m:35
    data_te=data(ts_idx,arange())
# test_learn.m:36
    labels_tr=labels(tr_idx)
# test_learn.m:37
    labels_te=labels(ts_idx)
# test_learn.m:38
    cv=cvpartition(numel(labels_tr),'k',K)
# test_learn.m:40
    for k in arange(1,K).reshape(-1):
        data_tr_cell[k]=data_tr(training(cv,k) == 0,arange())
# test_learn.m:42
        labels_tr_cell[k]=labels_tr(training(cv,k) == 0)
# test_learn.m:43
    
    clear('K','cv','data','labels','z','tr_idx','ts_idx','k','data_tr','labels_tr')
    model.type = copy('CART')
# test_learn.m:47
    net.base_classifier = copy(model)
# test_learn.m:48
    net.iterations = copy(3)
# test_learn.m:49
    net.mclass = copy(numel(unique(labels_te)))
# test_learn.m:50
    net,errs=learn(net,data_tr_cell,labels_tr_cell,data_te,labels_te,nargout=2)
# test_learn.m:52
    plot(errs)