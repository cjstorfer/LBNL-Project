from __future__ import print_function#, division
import sys
sys.path.append('..')
import time
from astropy.table import Table
import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from deeplens.resnet_classifier import deeplens_classifier
home = expanduser("~")



user_imgs = raw_input("please input the number of origional training images: ")
orig_epoch = raw_input("please input the number of origional epochs: ")

model = deeplens_classifier()


imgs = int(user_imgs)



# download_path=home+'/Desktop/' 
download_path='//Volumes/CJSTORFER/' 

export_path=home+'/Desktop/'   # To be adjusted on your machine



d = Table.read(export_path+'catalogs_'+str(imgs)+'_RGB.hdf5', path='/ground')  # Path to be adjusted on your machine




x = np.asarray(d['image']).reshape((-1,3,101,101))
# print x.shape

y = np.asarray(d['is_lens']).reshape((-1,1))
# print y.shape

xval = np.asarray(d['image'][int(imgs*.75):]).reshape((-1,3,101,101))
yval = np.asarray(d['is_lens'][int(imgs*.75):]).reshape((-1,1))
# print xval.shape
# print yval.shape




vmin=-1e-9
vmax=1e-9
scale=100

mask = np.where(x == 100)
mask_val = np.where(xval == 100)

x[mask] = 0
xval[mask_val] = 0


x = np.clip(x, vmin, vmax)/vmax * scale
xval = np.clip(xval, vmin, vmax)/vmax * scale 

x[mask] = 0
xval[mask_val] = 0

model.load('/Users/Chris/CMUDeepLens/Trained_Sets/'+str(user_imgs)+'imgs_'+str(orig_epoch)+'epochs_RGB/deeplens_params_'+str(orig_epoch)+'.npy', x, y)

completed_epochs = model.n_epochs

print ('Model has been trained for'+str(completed_epochs)+' epochs...')

user_epochs = raw_input("Please input # of aditional epochs to train: ")

n_epochs = int(user_epochs)


model.n_epochs = n_epochs


print ('Continue fitting...')
model.fit(x,y,xval,yval)

model.n_epochs = completed_epochs+n_epochs
print ('Total # of epochs trained: ' + str(completed_epochs + n_epochs))

model.save('deeplens_params_final.npy')



# model.eval_purity_completeness(xval,yval)



# tpr,fpr,th = model.eval_ROC(xval,yval)




# plt.title('ROC on Training set')
# plt.plot(fpr,tpr)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.xlim(0,1); 
# plt.ylim(0,1.)
# plt.grid(True)

# plt.show()