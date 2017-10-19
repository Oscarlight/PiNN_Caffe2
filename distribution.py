import numpy as np
import csv
import os
import math as math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as colors
from pylab import rcParams
path = './maxloss7e5/distribution/'

FIGURE_SIZE = (10,6)
FONT_SIZE = 16
LINE_WIDTH = 2
MAJOR_LABEL_SIZE = 14
MINOR_LABEL_SIZE = 0

def readalphafile(filename):
    alphalist = []
    epochlist = []
    train_losslist = []
    eval_losslist = []
    train_l1_metriclist = []
    eval_l1_metriclist = []
    train_scaled_l1_metriclist = []
    eval_scaled_l1_metriclist = []
    with open(filename) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile,delimiter=',')
        rows = list(csvReader)

        for row in rows:
            alphalist.append(row['alpha'])
            epochlist.append(row['epoch'])
            train_losslist.append(row['train_loss'])
            eval_losslist.append(row['eval_loss'])
            train_l1_metriclist.append(row['train_l1_metric'])
            eval_l1_metriclist.append(row['eval_l1_metric'])
            train_scaled_l1_metriclist.append(row['train_scaled_l1_metric'])
            eval_scaled_l1_metriclist.append(row['eval_scaled_l1_metric'])

        alphalist = np.array(map(float, alphalist), dtype = np.float64)   
        epochlist = np.array(map(float, epochlist), dtype = np.float64)
        train_losslist = np.array(map(float, train_losslist), dtype = np.float64)
        eval_losslist = np.array(map(float, eval_losslist), dtype = np.float64)
        train_l1_metriclist = np.array(map(float, train_l1_metriclist), dtype = np.float64)
        eval_l1_metriclist = np.array(map(float, eval_l1_metriclist), dtype = np.float64)
        train_scaled_l1_metriclist = np.array(map(float, train_scaled_l1_metriclist), dtype = np.float64)
        eval_scaled_l1_metriclist = np.array(map(float, eval_scaled_l1_metriclist), dtype = np.float64)

        for i in range(0,len(eval_losslist)):
            if(eval_losslist[i]>10):
                eval_losslist[i] = np.nan
            else:
                pass

    return alphalist,epochlist,train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist

def read_distribution_file(filename):
    train_losslist = []
    eval_losslist = []
    train_l1_metriclist = []
    eval_l1_metriclist = []
    train_scaled_l1_metriclist = []
    eval_scaled_l1_metriclist = []
    with open(filename) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile,delimiter=',')
        rows = list(csvReader)

        for row in rows:
            train_losslist.append(row['train_loss'])
            eval_losslist.append(row['eval_loss'])
            train_l1_metriclist.append(row['train_l1_metric'])
            eval_l1_metriclist.append(row['eval_l1_metric'])
            train_scaled_l1_metriclist.append(row['train_scaled_l1_metric'])
            eval_scaled_l1_metriclist.append(row['eval_scaled_l1_metric'])

        train_losslist = np.array(map(float, train_losslist), dtype = np.float64)
        eval_losslist = np.array(map(float, eval_losslist), dtype = np.float64)
        train_l1_metriclist = np.array(map(float, train_l1_metriclist), dtype = np.float64)
        eval_l1_metriclist = np.array(map(float, eval_l1_metriclist), dtype = np.float64)
        train_scaled_l1_metriclist = np.array(map(float, train_scaled_l1_metriclist), dtype = np.float64)
        eval_scaled_l1_metriclist = np.array(map(float, eval_scaled_l1_metriclist), dtype = np.float64)

        train_loss_mean = sum(train_losslist) / len(train_losslist)
        train_loss_variance = np.var(train_losslist)
        eval_loss_mean = sum(eval_losslist) / len(eval_losslist)
        eval_loss_variance = np.var(eval_losslist)
        train_l1_metric_mean = sum(train_l1_metriclist) / len(train_l1_metriclist)
        train_l1_metric_variance = np.var(train_l1_metriclist)
        eval_l1_metric_mean = sum(eval_l1_metriclist) / len(eval_l1_metriclist)
        eval_l1_metric_variance = np.var(eval_l1_metriclist)
        train_scaled_l1_metric_mean = sum(train_scaled_l1_metriclist) / len(train_scaled_l1_metriclist)
        train_scaled_l1_metric_variance = np.var(train_scaled_l1_metriclist)
        eval_scaled_l1_metric_mean = sum(eval_scaled_l1_metriclist) / len(eval_scaled_l1_metriclist)
        eval_scaled_l1_metric_variance = np.var(eval_scaled_l1_metriclist)

    return train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist

def give_mean_variance(filename):
    train_losslist = []
    eval_losslist = []
    train_l1_metriclist = []
    eval_l1_metriclist = []
    train_scaled_l1_metriclist = []
    eval_scaled_l1_metriclist = []
    with open(filename) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile,delimiter=',')
        rows = list(csvReader)

        for row in rows:
            train_losslist.append(row['train_loss'])
            eval_losslist.append(row['eval_loss'])
            train_l1_metriclist.append(row['train_l1_metric'])
            eval_l1_metriclist.append(row['eval_l1_metric'])
            train_scaled_l1_metriclist.append(row['train_scaled_l1_metric'])
            eval_scaled_l1_metriclist.append(row['eval_scaled_l1_metric'])

        train_losslist = np.array(map(float, train_losslist), dtype = np.float64)
        eval_losslist = np.array(map(float, eval_losslist), dtype = np.float64)
        train_l1_metriclist = np.array(map(float, train_l1_metriclist), dtype = np.float64)
        eval_l1_metriclist = np.array(map(float, eval_l1_metriclist), dtype = np.float64)
        train_scaled_l1_metriclist = np.array(map(float, train_scaled_l1_metriclist), dtype = np.float64)
        eval_scaled_l1_metriclist = np.array(map(float, eval_scaled_l1_metriclist), dtype = np.float64)

        
        train_loss_mean = np.nanmean(train_losslist)
        train_loss_variance = math.sqrt(np.var(train_losslist))
        eval_loss_mean = np.nanmean(eval_losslist)
        eval_loss_variance = math.sqrt(np.var(eval_losslist))
        train_l1_metric_mean = np.nanmean(train_l1_metriclist)
        train_l1_metric_variance = math.sqrt(np.var(train_l1_metriclist))
        eval_l1_metric_mean = np.nanmean(eval_l1_metriclist)
        eval_l1_metric_variance = math.sqrt(np.var(eval_l1_metriclist))
        train_scaled_l1_metric_mean = np.nanmean(train_scaled_l1_metriclist)
        train_scaled_l1_metric_variance = math.sqrt(np.var(train_scaled_l1_metriclist))
        eval_scaled_l1_metric_mean = np.nanmean(eval_scaled_l1_metriclist)
        eval_scaled_l1_metric_variance = math.sqrt(np.var(eval_scaled_l1_metriclist))

    return train_loss_mean,train_loss_variance,eval_loss_mean,eval_loss_variance,train_l1_metric_mean,train_l1_metric_variance,eval_l1_metric_mean,eval_l1_metric_variance,train_scaled_l1_metric_mean,train_scaled_l1_metric_variance,eval_scaled_l1_metric_mean,eval_scaled_l1_metric_variance

def plotdistribution(alphalist,mean_list,var_list,label):
    order = np.argsort(alphalist)
    xs = np.array(alphalist)[order]
    ys = np.array(mean_list)[order]

    if(label == 'train error'):
        plt.plot(
        xs, 
        ys, 'r', 
        label='train error'
        )

    elif(label == 'train_scaled_l1_metric'):
        plt.plot(
        xs, 
        ys, 'b', 
        label='train_scaled_l1_metric'
        )

    elif(label == 'train_l1_metric'):
        plt.plot(
        xs, 
        ys, 'g', 
        label='train_l1_metric'
        )

    elif(label == 'eval error'):
        plt.plot(
        xs, 
        ys, 'r--', 
        label='eval error'
        )        

    elif(label == 'eval_scaled_l1_metric'):
        plt.plot(
        xs, 
        ys, 'b--', 
        label='eval_scaled_l1_metric'
        )

    elif(label == 'eval_l1_metric'):
        plt.plot(
        xs, 
        ys, 'g--', 
        label='eval_l1_metric'
        )

    else:
        plt.plot(
        alphalist, 
        mean_list
        )      

   

    meanplussigma = [sum(x) for x in zip(mean_list, var_list)]
    meanminussigma = [a_i - b_i for a_i, b_i in zip(mean_list, var_list)]
    #plt.fill_between(alphalist,meanplussigma,meanminussigma,color='grey', alpha='0.5')
    #plt.legend()
    #plt.savefig(label+"diff_alpha.png")
    #plt.show()



def runonce():
    for i in range(1,13):
        path1 = './maxloss7e5/saved_data_'+str(i)+'/diff-alpha.csv'
        alphalist,epochlist,train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = readalphafile(path1)
        for k in range(0,len(alphalist)):
            alpha = alphalist[k]
            f = open(path+str(alpha)+"losses.csv", "a")
            if(os.stat(path+str(alpha)+"losses.csv").st_size == 0):
                f.write("{},{},{},{},{},{}\n".format("train_loss","eval_loss","train_l1_metric","eval_l1_metric","train_scaled_l1_metric","eval_scaled_l1_metric" ))
                f.write("{},{},{},{},{},{}\n".format(train_losslist[k],eval_losslist[k],train_l1_metriclist[k],eval_l1_metriclist[k],train_scaled_l1_metriclist[k],eval_scaled_l1_metriclist[k]))
            
            else:
                f.write("{},{},{},{},{},{}\n".format(train_losslist[k],eval_losslist[k],train_l1_metriclist[k],eval_l1_metriclist[k],
                        train_scaled_l1_metriclist[k],eval_scaled_l1_metriclist[k]))    
            f.close()
        
    

#runonce()

train_loss_mean_list = []
train_loss_var_list =[]
eval_loss_mean_list = []
eval_loss_var_list = []
train_l1_metric_mean_list = []
train_l1_metric_var_list = []
eval_l1_metric_mean_list = []
eval_l1_metric_var_list = []
train_scaled_l1_metric_mean_list = []
train_scaled_l1_metric_var_list = []
eval_scaled_l1_metric_mean_list = []
eval_scaled_l1_metric_var_list = []

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        print(filename)
        train_loss_mean,train_loss_variance,eval_loss_mean,eval_loss_variance,train_l1_metric_mean,train_l1_metric_variance,eval_l1_metric_mean,eval_l1_metric_variance,train_scaled_l1_metric_mean,train_scaled_l1_metric_variance,eval_scaled_l1_metric_mean,eval_scaled_l1_metric_variance = give_mean_variance (os.path.join(path, filename))
        print(eval_l1_metric_mean)
        train_loss_mean_list.append(train_loss_mean)
        train_loss_var_list.append(train_loss_variance)
        eval_loss_mean_list.append(eval_loss_mean)
        eval_loss_var_list.append(eval_loss_variance)
        train_l1_metric_mean_list.append(train_l1_metric_mean)
        train_l1_metric_var_list.append(train_l1_metric_variance)
        eval_l1_metric_mean_list.append(eval_l1_metric_mean)
        eval_l1_metric_var_list.append(eval_l1_metric_variance)
        train_scaled_l1_metric_mean_list.append(train_scaled_l1_metric_mean)
        train_scaled_l1_metric_var_list.append(train_scaled_l1_metric_variance)
        eval_scaled_l1_metric_mean_list.append(eval_scaled_l1_metric_mean)
        eval_scaled_l1_metric_var_list.append(eval_scaled_l1_metric_variance)
        continue
        
    else:
        continue


alphalist = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.15,0.1,0.25,0.2]
f = open("average-diff-alpha.csv", "w")
f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("alpha","train_loss_mean","train_loss_var","eval_loss_mean","eval_loss_var","train_l1_metric_mean","train_l1_metric_var",
                                           "eval_l1_metric_mean","eval_l1_metric_var","train_scaled_l1_metric_mean","train_scaled_l1_metric_mean","eval_scaled_l1_metric_mean","eval_scaled_l1_metric_var" ))
for x in zip(alphalist,train_loss_mean_list,train_loss_var_list,eval_loss_mean_list,eval_loss_var_list,train_l1_metric_mean_list,train_l1_metric_var_list,eval_l1_metric_mean_list,eval_l1_metric_var_list ,train_scaled_l1_metric_mean_list,train_scaled_l1_metric_var_list,eval_scaled_l1_metric_mean_list,eval_scaled_l1_metric_var_list):
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],x[9],x[10],x[11],x[12]))
f.close()

#plotdistribution(alphalist,train_loss_mean_list,train_loss_var_list,'train error')
#plotdistribution(alphalist,eval_loss_mean_list,eval_loss_var_list,'eval error')
#plotdistribution(alphalist,train_l1_metric_mean_list,train_l1_metric_var_list,'train_l1_metric')
plotdistribution(alphalist,eval_l1_metric_mean_list,eval_l1_metric_var_list,'eval_l1_metric')
#plotdistribution(alphalist,train_scaled_l1_metric_mean_list,train_scaled_l1_metric_var_list,'train_scaled_l1_metric')
#plotdistribution(alphalist,eval_scaled_l1_metric_mean_list,eval_scaled_l1_metric_var_list,'eval_scaled_l1_metric')

def plot_scatter(label):
    if(label == 'train error'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                train_losslist,s=50, c='r',alpha = 0.5 
                )

    if(label == 'train_scaled_l1_metric'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                train_scaled_l1_metriclist,s=50, c='b',alpha = 0.5 
                )

    if(label == 'train_l1_metric'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                train_l1_metriclist,s=50, c='g',alpha= 0.5 
                )

    if(label == 'eval error'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                print(alpha)
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                eval_losslist,s=50, c='g',alpha = 0.5 
                )
                print(eval_losslist)

    if(label == 'eval_scaled_l1_metric'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                eval_scaled_l1_metriclist,s=50, c='b',alpha=0.5 
                )

    if(label == 'eval_l1_metric'):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                alpha = (float(filename[0:filename.find('losses.csv', 0)]))
                print(alpha)
                train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,train_scaled_l1_metriclist,eval_scaled_l1_metriclist = read_distribution_file(os.path.join(path, filename))
                alphalist = alpha*np.ones(12)
                plt.scatter(
                alphalist, 
                eval_l1_metriclist,s=50, c='g',alpha=0.5 
                )
                print(eval_l1_metriclist)



plot_scatter('eval_l1_metric')
#plt.axis([0, 0.3, 0, 2])
ax = plt.gca()
#ax.set_xlim([xmin,xmax])
#ax.set_ylim([0,2])
#ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('alpha', fontsize=FONT_SIZE)
plt.ylabel('loss', fontsize=FONT_SIZE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['axes.linewidth'] = LINE_WIDTH
plt.tick_params(axis='both', which='major', labelsize=MAJOR_LABEL_SIZE)
plt.tick_params(axis='both', which='minor', labelsize=MINOR_LABEL_SIZE)
plt.show()






