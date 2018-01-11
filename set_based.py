from dcnn_model.feature_extractor import ExtractFeature
import numpy as np
from keras.preprocessing import image
from keras.engine import  Model
from keras.layers import Input
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import hdist 

image_input = Input(shape=(224, 224, 3))


# FC7 Features
vgg_model = ExtractFeature() # pooling: None, avg or max
out = vgg_model.get_layer('fc7').output
#vgg_model_fc7 = Model(image_input, out)
vgg_model_fc7 = Model(vgg_model.input, out)
#vgg_model_fc7.summary()

dictOfDesc = {}

print 'extracting features...'

for folder_name in os.listdir('young'):
        cwd = os.path.join('young',folder_name)
        desc=[]
        for file_name in os.listdir(cwd):
                img=image.load_img(os.path.join(cwd,file_name),target_size=(224,224))
                if img is None:
                        continue
                #print file_name,'loaded'
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                # TF order aka 'channel-last'
                x = x[:, :, :, ::-1]

                vgg_model_fc7_preds = vgg_model_fc7.predict(x)
                descriptor = vgg_model_fc7_preds[0]
                #dist = np.linalg.norm(a-descriptor)
                
                desc.append(descriptor)
        dictOfDesc[folder_name] = desc

print 'testing...'

for folder_name in os.listdir('old'):
        cwd = os.path.join('old',folder_name)
        old_desc = []
        for file_name in os.listdir(cwd):
                # Change the image path with yours.
                img = image.load_img(os.path.join(cwd,file_name), target_size=(224, 224))
                print file_name,'loaded'
        
                #plt.figure(1)
                #plt.imshow(img)
                #plt.show(block=False)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                # TF order aka 'channel-last'
                x = x[:, :, :, ::-1]


                vgg_model_fc7_preds = vgg_model_fc7.predict(x)

                a = vgg_model_fc7_preds[0]
                old_desc.append(a)
                print a
        
        

        min_dist = sys.maxint
        result_name = 'None'
        for name,young_desc in dictOfDesc.iteritems():
                
                young_arr = np.array(young_desc)
                old_arr = np.array(old_desc)
                
                dist = hdist.ModHausdorffDist(young_arr,old_arr)
                
                if dist < min_dist:
                        min_dist = dist
                        result_name = name
        print 'folder '+folder_name+': matches with '+result_name
        strr = raw_input("press enter to continue")
        #plt.close('all')




