# Generated with SMOP  0.41-beta
from libsmop import *
# 

    
@function
def learn(net=None,data_train=None,labels_train=None,data_test=None,labels_test=None,*args,**kwargs):
    varargin = learn.varargin
    nargin = learn.nargin

    Tk=net.iterations
# learn.m:3
    
    K=length(data_train)
# learn.m:4
    
    net.classifiers = copy(cell(dot(Tk,K),1))
# learn.m:5
    
    net.beta = copy(zeros(dot(Tk,K),1))
# learn.m:6
    
    c_count=0
# learn.m:7
    
    errs=zeros(dot(Tk,K),1)
# learn.m:8
    
    for k in arange(1,K).reshape(-1):
        data_train_k=data_train[k]
# learn.m:13
        labels_train_k=labels_train[k]
# learn.m:14
        D=ones(numel(labels_train_k),1) / numel(labels_train_k)
# learn.m:15
        if k > 1:
            predictions=classify_ensemble(net,data_train_k,labels_train_k,c_count)
# learn.m:19
            epsilon_kt=sum(D(predictions != labels_train_k))
# learn.m:21
            beta_kt=epsilon_kt / (1 - epsilon_kt)
# learn.m:22
            D[predictions == labels_train_k]=dot(beta_kt,D(predictions == labels_train_k))
# learn.m:23
        for t in arange(1,Tk).reshape(-1):
            c_count=c_count + 1
# learn.m:27
            D=D / sum(D)
# learn.m:29
            index=randsample(arange(1,numel(D)),numel(D),true,D)
# learn.m:31
            net.classifiers[c_count]=classifier_train(net.base_classifier,data_train_k(index,arange()),labels_train_k(index))
# learn.m:33
            y=classifier_test(net.classifiers[c_count],data_train_k)
# learn.m:38
            epsilon_kt=sum(D(y != labels_train_k))
# learn.m:39
            net.beta[c_count]=epsilon_kt / (1 - epsilon_kt)
# learn.m:40
            predictions=classify_ensemble(net,data_train_k,labels_train_k,c_count)
# learn.m:43
            E_kt=sum(D(predictions != labels_train_k))
# learn.m:45
            if E_kt > 0.5:
                E_kt=0.5
# learn.m:47
            Bkt=E_kt / (1 - E_kt)
# learn.m:50
            D[predictions == labels_train_k]=dot(Bkt,D(predictions == labels_train_k))
# learn.m:51
            D=D / sum(D)
# learn.m:52
            predictions,posterior=classify_ensemble(net,data_test,labels_test,c_count,nargout=2)
# learn.m:54
            errs[c_count]=sum(predictions != labels_test) / numel(labels_test)
# learn.m:56
    
    
@function
def classify_ensemble(net=None,data=None,labels=None,lims=None,*args,**kwargs):
    varargin = classify_ensemble.varargin
    nargin = classify_ensemble.nargin

    n_experts=copy(lims)
# learn.m:65
    weights=log(1.0 / net.beta(arange(1,lims)))
# learn.m:66
    p=zeros(numel(labels),net.mclass)
# learn.m:67
    for k in arange(1,n_experts).reshape(-1):
        y=classifier_test(net.classifiers[k],data)
# learn.m:69
        for m in arange(1,numel(y)).reshape(-1):
            p[m,y(m)]=p(m,y(m)) + weights(k)
# learn.m:73
    
    __,predictions=max(p.T,nargout=2)
# learn.m:76
    predictions=predictions.T
# learn.m:77
    posterior=p / repmat(sum(p,2),1,net.mclass)
# learn.m:78
