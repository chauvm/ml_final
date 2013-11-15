'''
Steepest descent
n = dimension of x
m = number of intermediate features
'''
import time
import numpy as np
import matplotlib.pyplot as plt
t = time.time()
#init data, change numbers here!!!
n = 2
m = 4
# beta = np.array([-1.53, 1, 1, 1, 1])
# W = np.array([-0.37, -1.6, 0.41, -0.41, -1, 1, -1, 1, 1, 1, -1, -1]).reshape(n+1,m)
beta = np.array([-1.51, 1, 1, 1, 1])
W = np.array([-0.4, -1.61, 0.4, -0.4, -1, 1, -1, 1, 1, 1, -1, -1]).reshape(n+1,m)
init = np.append(W.flatten(),beta)
delta = 0.01
s = 0.5 #for armijo
b = 0.5 #for armijo
sigma = 0.000001 #for armijo
eta_default = 0.000009
stoc_lim = 10000 # for stochastic gradient
size = 25 # for stochastic gradient
#read training data
fname_train = "C:\Users\Anh Huynh\Documents\Aptana Studio 3 Workspace\Stat535_Homework\hw2-nn-train.dat"
read_buffer = np.genfromtxt(fname_train,dtype=str,delimiter="\n")
raw_train_data=[[float(u) for u in line.split(" ")] for line in read_buffer]
train_data = [[[line[0],line[1]],line[2]] for line in raw_train_data]
x_train, y_train = zip(*train_data)
# fname_test = "C:\Users\Anh Huynh\Documents\Aptana Studio 3 Workspace\Stat535_Homework\hw2-nn-test.dat"
# read_buffer = genfromtxt(fname_test,dtype=str,delimiter="\n")
# raw_test_data=[[float(u) for u in line.split(" ")] for line in read_buffer]
# test_data = [[[line[0],line[1]],line[2]] for line in raw_test_data]
# test_x, test_y = zip(*test_data)

##########################Engine Room############################

#multi-layer neural network classifier
def sigmoid(u):
    return 1/(1+np.exp(-u))
def k_neural(layers, x):
    '''
    layers = [W_1, W_2,..., beta].
    Mathematically: data*W_1*W_2*...*W_l*beta^T
    return f = last_layer*beta or sigmoid(last_layer*beta)
    '''
    z = [np.insert(x,0,1)] #insert 1 into the first entry
    for W in layers[:-1]:
        z.append(z[-1].dot(W))
        z_new = [sigmoid(u) for u in z[-1]] #apply sigmoid to each
        z[-1] = z_new
    z_aug = np.insert(z[-1], 0,1) # insert 1 into the head of the list
    if (layers[-1].dot(z_aug.T)>0):
        return 1
    else:
        return -1
def nn(params,x):
    layers = [params[:(n+1)*m].reshape(n+1,m),params[(n+1)*m:]]
    return k_neural(layers,x)

#methods to numerically compute gradients and hessians, 
#http://heuristic-methods.googlecode.com/hg-history/30c5186ae4ca529ca27dc626acc10beb02fa1509/hm/gradient.py
def gradient(func, params, delta):
    """
    Calculate the gradient of func evaluated at params
    """      
    dims = len(params)
    grad = np.zeros(dims)  
    tmp = np.zeros(dims)        
    # Compute the gradient
    for i in xrange(dims):     
        tmp[i] = delta
        grad[i] = (func((params + tmp)) - func((params - tmp)))/delta
        tmp[i] = 0
    return grad
 
def loss(f, params, x, y):
    return sum([np.log(1 + np.exp(-y[i]*f(params, x[i]))) for i in range(len(x))])/(len(x)+0.0) #f is the classifier

def stocloss(f, params,x,y,size):
    I = [np.random.randint(0,len(x)) for i in range(size)]
    return sum([np.log(1+np.exp(-y[i]*f(params,x[i]))) for i in I])/(len(I)+0.0)

#auxiliary methods for glearn
def check_convergence(L_k, L_k_1):
    if (1-L_k/L_k_1)<0.0001:
        return True
    else:
        return False

def armijo(f,params,s, beta, sigma,d):
    eta = s
    while (True):
        if f(params)-f(params-eta*d)>sigma*eta*(-gradient(f,params,delta).dot(d)):
            break
        else:
            eta = eta*beta
    return eta

def glearn(L, delta, init, line_search_option, stochastic_option):
    '''
    use steepest descent method to optimize a given function
    '''
    params = [init]
    k = 0
    Loss = [L(init)]
    while (True):
        grad = gradient(L, params[-1],delta)
        if line_search_option==True:
            eta = armijo(L,params[-1],s, b, sigma,grad)
        elif stochastic_option:
            eta = eta_default/(np.sqrt(k)/500.0+100.0)
        else:
            eta = eta_default
        new_params =  params[-1] - eta*grad
        params = np.concatenate((params,[new_params]))  #if bug check this point: used to be params = [new_params]
        Loss = np.concatenate((Loss,[L(params[-1])]))
        k += 1
        if stochastic_option:
            if k>stoc_lim:
                break
        elif check_convergence(Loss[-1],Loss[-2]):
            break
    return [params,Loss,k] #return the entire list of params gotten through the learning process

##############################Race Tracks##############################
def L(params):
    return loss(nn, params, x_train, y_train)
def L_stoc(params):
    return stocloss(nn, params, x_train, y_train, size)
# print "doing GFIX"
# GFIX, L_GFIX, k_GFIX = glearn(L, delta, init, False, False)
# print L_GFIX
# print "doing GLS"
# GLS, L_GLS, k_GLS = glearn(L, delta, init, True, False)
# print L_GLS

# print time.time()-t
# print "plotting"
# plt.figure(1)
# plt.plot([i+1 for i in range(k_GFIX+1)], [params[(n+1)*m] for params in GFIX], 'r-', [i+1 for i in range(k_GLS+1)], [params[(n+1)*m] for params in GLS], 'b-')
# plt.xlabel("beta_0 vs k, red: GFIX; blue: GLS")
# plt.show()

GS, L_GS, k_GS = glearn(L_stoc,delta, init, False, True)
print L_GS
print time.time()-t
plt.plot([i+1 for i in range(k_GS+1)],L_GS, 'r-')
plt.show()

# def dec_reg_plot(step,xlim, ylim,classifier):
#     plt.xlim(xlim[0],xlim[1])
#     plt.ylim(ylim[0],ylim[1])
#     x = xlim[0]
#     y = ylim[0]
#     while (True):
#         if classifier(np.array([x,y]))>0:
#             plt.plot(x,y,'ys')
#         y+= step
#         if y > ylim[1]:
#             x+=step
#             y = ylim[0]
#         if x > xlim[1]:
#             break
#     plt.xlabel('red means y=1')
# #     plt.show()
#     return 0

# def classifier(data):
#     return nn(params,data)
# dec_reg_plot(0.005,[0,1],[0,1],classifier)

# for i in range(len(train_x)):
#     if nn(params,train_x[i])>0:
#         plt.plot(train_x[i][0],train_x[i][1],'ro')
#     else:
#         plt.plot(train_x[i][0],train_x[i][1],'bo')
# plt.show()
