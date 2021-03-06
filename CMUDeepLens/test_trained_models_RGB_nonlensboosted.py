import sys
sys.path.append('..')
import time
from astropy.table import Table
import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from deeplens.resnet_classifier import deeplens_classifier

from deeplens.utils.blfchallenge import classify_ground_challenge 

from sklearn import metrics

home = expanduser("~")


imgs = 500

# Path to the downloaded files
# download_path=home+'/Desktop/' # To be adjusted on your machine
download_path='//Volumes/CJSTORFER/' # To be adjusted on your machine



# Path to export the data
export_path=home+'/Desktop/'   # To be adjusted on your machine


#Loads x, y dataset that belong to the previously trained CNN

d = Table.read(export_path+'catalogs_'+str(imgs)+'_RGB.hdf5', path='/ground')  



x = np.asarray(d['image'])[0:120,:,:,:].reshape((-1,3,101,101))

y = np.asarray(d['is_lens'])[0:120,].reshape((-1,1))

x_additional = np.asarray(d['image'])[120:500,:,:,:].reshape((-1,3,101,101))
y_additional = np.asarray(d['is_lens'])[120:500,].reshape((-1,1))


x_nonlens = np.zeros((0,3,101,101))
for a, b in zip(x_additional, y_additional):
    if b==0:
        x_nonlens = np.concatenate((x_nonlens,a.reshape(1,3,101,101)),axis = 0)

x_nonlens = x_nonlens[0:60]

y_nonlens = np.zeros((60,))

x = np.concatenate((x, x_nonlens))

y = np.concatenate((y.reshape(120,), y_nonlens)).reshape(180,1)

# Clipping and scaling parameters applied to the data as preprocessing
vmin=-1e-9
vmax=1e-9
scale=100

mask = np.where(x == 100)

x[mask] = 0

# Simple clipping and rescaling the images
x = np.clip(x, vmin, vmax)/vmax * scale

x[mask] = 0



model = deeplens_classifier()
model.load('/Users/Chris/CMUDeepLens/Trained_Sets/180imgs_120epochs_RGB_nonlensboosted/deeplens_params_final.npy', x, y)





#--------------------------------------------Loading new testing dataset--------------------------------------------#


test_imgs = 1000



# Loads the catalog
# cat_test = Table.read(download_path+'GroundBasedTraining/classifications.csv')
# cat_test= cat_test[imgs:imgs+test_imgs]

# ims_test = np.zeros((test_imgs, 3, 101, 101))
# counter = 0
# # print ims.shape
# # Loads the images
# for i, id in enumerate(cat_test['ID']):
#     if (i%10 == 0) and (i != 0):
#         counter +=1
#         print str(counter)+"0 done."
#     for j, b in enumerate(['R', 'I', 'G']):
#         ims_test[i, j] = fits.getdata(download_path+'GroundBasedTraining/Public/Band'+str(j+1)+'/imageSDSS_'+b+'-'+str(id)+'.fits')


# # Concatenate images to catalog
# cat_test['image'] = ims_test    

# # Export catalog as HDF5
# cat_test.write(export_path+'catalogs_test_'+str(test_imgs)+'_RGB.hdf5', path='/ground', append=True)

# print "Done !"


# # Loads the table created in the previous section
d = Table.read(export_path+'catalogs_test_'+str(test_imgs)+'_RGB.hdf5', path='/ground')  


x_test = np.array(d['image']).reshape((-1,3,101,101))


y_test = np.array(d['is_lens']).reshape((-1,1))


# Clipping and scaling parameters applied to the data as preprocessing
vmin=-1e-9
vmax=1e-9
scale=100

mask = np.where(x_test == 100)

x_test[mask] = 0

# Simple clipping and rescaling the images
x_test = np.clip(x_test, vmin, vmax)/vmax * scale

x_test[mask] = 0



#--------------------------------------------Preping the test set--------------------------------------------#
n_sets = 10
set_size = int(x_test.shape[0]/n_sets)

x_test_set = np.empty((n_sets, set_size, 3, 101, 101))
y_test_set = np.empty((n_sets, set_size))

tpr = np.empty((n_sets, 1000))
fpr = np.empty((n_sets, 1000))
th = np.empty((n_sets, 1000))
roc_auc = np.empty((n_sets, ))
plt.figure(figsize=(25,25))
for i in range(n_sets): 
	print 'test '+ str(i+1)
	x_test_set[i] = x_test[i*set_size:(i+1)*set_size]
	y_test_set[i] = y_test[i*set_size:(i+1)*set_size].reshape(-1,)
	# print y_test_set[i].shape
	# print y_test_set[i].reshape(-1,1).shape
	# print x_test_set[i].shape
	tpr[i],fpr[i],th[i] = model.eval_ROC(x_test_set[i], y_test_set[i].reshape(-1,1))
	roc_auc[i] = metrics.auc(fpr[i], tpr[i])
	plt.suptitle('ROC curves and AUC for test sets of 100 imgs\nPredictions made by CNN trained on 1000 imgs for 70 epochs', fontsize=25)
	plt.subplot(5,2,i+1)
	plt.title('ROC on test set: ' + str(i), fontsize=20)
	plt.plot(fpr[i],tpr[i])
	plt.xlabel('FPR', fontsize=15)
	plt.ylabel('TPR', fontsize=15)
	plt.xlim(0,1); 
	plt.ylim(0,1.)
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.xlim([-0.02, 1.0])
	plt.ylim([0.0, 1.02])
	plt.text(1, 0, 'AUC: ' + str("%.4f" % roc_auc[i]),
	        verticalalignment='bottom', horizontalalignment='right', fontsize=20)
	plt.grid(True)
plt.savefig("test_trials_ROC.png")
plt.show()


plt.figure()
x = np.average(fpr, axis = 0)
y = np.average(tpr, axis = 0)
stdev_y = np.std(tpr,axis = 0)
avg_auc = np.average(roc_auc, axis = 0)
stdev_auc = np.std(roc_auc,axis = 0)
plt.plot(x,y)
plt.fill_between(x, y+stdev_y, y-stdev_y, facecolor = 'red')
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.text(1, 0, 'Average AUC: ' + str("%.4f" % avg_auc),
		verticalalignment='bottom', horizontalalignment='right', fontsize=20)
plt.text(1, 0.1, 'SD of AUC: ' + str("%.4f" % stdev_auc),
		verticalalignment='bottom', horizontalalignment='right', fontsize=20)
plt.grid(True)
plt.plot([0,1],[0,1], 'g--')
plt.title('STDEV on test sets', fontsize=20)
plt.savefig("test_ROC_stats.png")
plt.show()
	# p = model.predict_proba(xval)

















