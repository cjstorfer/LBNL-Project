import sys
sys.path.append('..')
import time
from astropy.table import Table
import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")

user_imgs = raw_input("Please input # of images: ")
user_epochs = raw_input("Please input # of epochs: ")

imgs = int(user_imgs)
n_epochs = int(user_epochs)

# download_path=home+'/Desktop/' 
download_path='//Volumes/CJSTORFER/' 




export_path=home+'/Desktop/'   # To be adjusted on your machine


cat = Table.read(download_path+'GroundBasedTraining/classifications.csv')
cat= cat[0:imgs]
len(cat)


ims = np.zeros((imgs, 3, 101, 101))
counter = 0


for i, id in enumerate(cat['ID']):
    if (i%10 == 0):
        counter +=1
        print str(counter)+"0 done."
    for j, b in enumerate(['R', 'I', 'G']):
        ims[i, j] = fits.getdata(download_path+'GroundBasedTraining/Public/Band'+str(j+1)+'/imageSDSS_'+b+'-'+str(id)+'.fits')


cat['image'] = ims


cat.write(export_path+'catalogs_'+str(imgs)+'_RGB.hdf5', path='/ground', append=True)

print "Done !"


from astropy.table import Table


d = Table.read(export_path+'catalogs_'+str(imgs)+'_RGB.hdf5', path='/ground')  # Path to be adjusted on your machine
print d['image'].shape

x = np.asarray(d['image']).reshape((-1,3,101,101))
print x.shape

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


print 'Lens(1): '+ str(np.sum(y==1))
print 'Non-lens(0): '+ str(np.sum(y==0))

# im = x[0].T
# plt.subplot(221)
# plt.imshow(im[:,:,0])
# plt.subplot(222)
# plt.imshow(im[:,:,1])
# plt.subplot(223) 
# plt.imshow(im[:,:,2])
# plt.subplot(224)
# plt.imshow(im[:,:,3])
# plt.show()

begin_training = raw_input("Press enter to begin traning: ")

import datetime
print(datetime.datetime.now())

from deeplens.resnet_classifier import deeplens_classifier
print 'Training begining...'
print str(n_epochs)+' epochs...'
print str(imgs) + ' images...'

model = deeplens_classifier(learning_rate=0.001, learning_rate_steps=3, learning_rate_drop=0.1, batch_size=128, n_epochs=n_epochs)           



model.fit(x,y,xval,yval) 



model.save('deeplens_params_final.npy')



model.eval_purity_completeness(xval,yval)



tpr,fpr,th = model.eval_ROC(xval,yval)




plt.title('ROC on Training set')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(0,1); 
plt.ylim(0,1.)
plt.grid(True)



p = model.predict_proba(xval)
plt.show()

# # ## Classify Testing set
# # 
# # In this section, we test the model on one of the test datasets provided as part of the challenge

# # In[76]:


# from deeplens.utils.blfchallenge import classify_ground_challenge 

# # Utility function to classify the challenge data with a given model
# cat = classify_ground_challenge(model, '/data2/BolognaSLChallenge/Dataset3') # Applies the same clipping 
#                                                                              # and normalisation as during training


# # In[23]:


# # Export the classified catalog, ready to submit to the challenge
# cat.write('deeplens_ground_classif.txt',format='ascii.no_header')


# # In[24]:


# from astropy.table import join

# # Load results catalog
# cat_truth = Table.read('ground_catalog.4.csv',  format='csv', comment="#")

# # Merging with results of the classification
# cat = join(cat_truth,cat,'ID')

# # Renaming columns for convenience
# cat['prediction'] = cat['is_lens']
# cat['is_lens'] = ( cat['no_source'] == 0).astype('int')


# # In[32]:


# from sklearn.metrics import roc_curve,roc_auc_score

# # Compute the ROC curve
# fpr_test,tpr_test,thc = roc_curve(cat['is_lens'], cat['prediction'])


# # In[38]:


# plot(fpr_test,tpr_test,label='CMU DeepLens')
# xlim(0,1)
# ylim(0,1)
# legend(loc=4)
# xlabel('FPR')
# ylabel('TPR')
# title('ROC evaluated on Testing set')
# grid('on')


# # In[36]:


# # Get AUROC metric on the whole testing set
# roc_auc_score(cat['is_lens'], cat['prediction'])


# # As a point of reference, our winning submission got an AUROC of 0.9814321
