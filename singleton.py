from dcnn_model.feature_extractor import ExtractFeature
import numpy as np
from keras.engine import  Model
from keras.layers import Input
from keras.preprocessing import image
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dist_metrics import manhattan_distance,euclidean_distance,cosine_similarity,tanimoto_coefficient

image_input = Input(shape=(224, 224, 3))


# FC7 Features
vgg_model = ExtractFeature()
out = vgg_model.get_layer('fc7').output
#vgg_model_fc7 = Model(image_input, out)
vgg_model_fc7 = Model(vgg_model.input, out)
#vgg_model_fc7.summary()

images=[]

print 'extracting features...'

for filename in os.listdir('child'):
        img=image.load_img(os.path.join('child',filename),target_size=(224,224))
     
        #print filename,'loaded'
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # Tensorflow data format = 'channels_last'
        x = x[:, :, :, ::-1]
        
        vgg_model_fc7_preds = vgg_model_fc7.predict(x)
        descriptor = vgg_model_fc7_preds[0]
        #dist = np.linalg.norm(a-descriptor)
        if img is not None:
                images.append([descriptor,filename])
        #print len(images)

print 'testing...'

for filename in os.listdir('adult'):
        # Change the image path with yours.
        img = image.load_img(os.path.join('adult',filename), target_size=(224, 224))
        #print filename,'loaded'
        plt.figure(1)
        plt.imshow(img)
        #plt.show(block=False)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # TF order aka 'channel-last'
        x = x[:, :, :, ::-1]


        vgg_model_fc7_preds = vgg_model_fc7.predict(x)

        a = vgg_model_fc7_preds[0]
        print filename,a


        img_list = []
        for i in images:
                dist = cosine_similarity(a,i[0])
                img_list.append([dist,i[1]])
                
        img_list.sort(key = lambda x:x[0]) 
        img=[]
        for i in range(1):
                print img_list[i][1],images[i][0]
                print 'first three distances : ',img_list[0][0],img_list[1][0],img_list[2][0]
                imgt=mpimg.imread(os.path.join('child',img_list[i][1]))
                plt.figure(i+2)
                plt.imshow(imgt)
                plt.show(block=False)
        name = raw_input("press any key to continue")
        plt.close('all')




