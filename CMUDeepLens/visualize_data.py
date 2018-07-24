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


imgs = 400

# Path to the downloaded files
# download_path=home+'/Desktop/' # To be adjusted on your machine
download_path='//Volumes/CJSTORFER/' # To be adjusted on your machine



# Path to export the data
export_path=home+'/Desktop/'   # To be adjusted on your machine


d = Table.read(export_path+'catalogs_'+str(imgs)+'.hdf5', path='/ground')  


x = np.array(d['image']).reshape((-1,4,101,101))


y = np.array(d['is_lens']).reshape((-1,1))


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
model.load('/Users/Chris/CMUDeepLens/Trained_Sets/400imgs_120epochs/deeplens_params_final.npy', x, y)


#--------------------------------------------Loading new testing dataset--------------------------------------------#


test_imgs = 1000



# # Loads the catalog
# cat_test = Table.read(download_path+'GroundBasedTraining/classifications.csv')
# cat_test= cat_test[imgs:imgs+test_imgs]

# ims_test = np.zeros((test_imgs, 4, 101, 101))
# counter = 0
# # print ims.shape
# # Loads the images
# for i, id in enumerate(cat_test['ID']):
#     if (i%10 == 0) and (i != 0):
#         counter +=1
#         print str(counter)+"0 done."
#     for j, b in enumerate(['R', 'I', 'G', 'U']):
#         ims_test[i, j] = fits.getdata(download_path+'GroundBasedTraining/Public/Band'+str(j+1)+'/imageSDSS_'+b+'-'+str(id)+'.fits')


# # Concatenate images to catalog
# cat_test['image'] = ims_test    

# # Export catalog as HDF5
# cat_test.write(export_path+'catalogs_test_'+str(test_imgs)+'.hdf5', path='/ground', append=True)

# print "Done !"


# # Loads the table created in the previous section
d = Table.read(export_path+'catalogs_test_'+str(test_imgs)+'.hdf5', path='/ground')  


x_test = np.array(d['image']).reshape((-1,4,101,101))


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

print x_test.shape

#--------------------------------------------Preping the test set--------------------------------------------#
idx_0 = np.where(y_test==0)
idx_1 = np.where(y_test == 1)


nlens = np.delete(x_test,idx_0, axis = 0)
lens = np.delete(x_test,idx_1,axis = 0 )
print nlens.shape
print lens.shape

prob_0 = model.predict_proba(nlens)
print 'prob 0 done'


prob_1 = model.predict_proba(lens)
print 'prob 1 done'



n_bins = 20

plt.hist(prob_0, bins = n_bins, color = 'red')
plt.hist(prob_1, bins = n_bins, color = 'blue')
plt.title('Histogram of predicted probablity for each image')
plt.xlabel('probablity')
plt.ylabel('N')
plt.savefig('probability distrobution.png')
plt.show()

