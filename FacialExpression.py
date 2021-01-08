import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

filname = 'fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
df=pd.read_csv('fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    
    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y



X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)


# keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 48, 48, 1)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Activation , Dropout ,Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *



def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())                                   ####Stable learning
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    
    return model
model=my_model()
model.summary()



from tensorflow.keras import backend as K

path_model='model_filter.h5' # save model at this location after each epoch
K.clear_session() # destroys the current graph and builds a new one
model=my_model() # create the model

K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model

h = model.fit(x=X_train,
            y=y_train,
            batch_size=64,
            epochs=20,
            verbose=1,
            validation_data=(X_test,y_test),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
y_pos = np.arange(len(objects))
print(y_pos)


def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
plt.show()

#model.save("Saved_Model") # SAVE THE MODEL

#model = tf.keras.models.load_model('Saved_Model')  #LOAD THE SAVED MODEL


y_pred=model.predict(X_test)
y_test.shape


from skimage import io
img = image.load_img('sad.jpg', color_mode = "grayscale", target_size=(48, 48))
show_img = image.load_img('sad.jpg',  target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
#print(custom[0])
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(show_img)
plt.show()

m = 0.000000000000000000001
a = custom[0]
for i in range(0,len(a)):
    if a[i] > m:
        m = a[i]
        ind = i
        
print('Expression Prediction:',objects[ind])
        


from skimage import io
img = image.load_img('sad1.jpg', color_mode = "grayscale", target_size=(48, 48))
show_img = image.load_img('sad1.jpg', grayscale=False, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(show_img)
plt.show()

m = 0.000000000000000000001
a = custom[0]
for i in range(0,len(a)):
    if a[i] > m:
        m = a[i]
        ind = i
        
print('Expression Prediction:',objects[ind])


                    ####    CODE FOR ACCESSING CAMERA FOR REAL TIME RECOGNITION       ####

#from cv2 import *
#
#def getExpr(file):
#    from skimage import io
#    img = tf.keras.preprocessing.image.load_img(file, color_mode = "grayscale", target_size=(48, 48))
#    show_img = tf.keras.preprocessing.image.load_img(file, grayscale=False, target_size=(200, 200))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis = 0)
#
#    x /= 255
#
#    custom = model.predict(x)
#    emotion_analysis(custom[0])
#
#    x = np.array(x, 'float32')
#    x = x.reshape([48, 48]);
#
#    plt.gray()
#    plt.imshow(show_img)
#    plt.show()
#
#    m = 0.000000000000000000001
#    a = custom[0]
#    for i in range(0,len(a)):
#        if a[i] > m:
#            m = a[i]
#            ind = i
#
#    print('Expression Prediction:',objects[ind])
#
#cam = cv2.VideoCapture(0)
#
#cv2.namedWindow("test")
#
#img_counter = 0
#
#while True:
#    ret, frame = cam.read()
#    if not ret:
#        print("failed to grab frame")
#        break
#    cv2.imshow("test", frame)
#
#    k = cv2.waitKey(1)
#    if k%256 == 27:
#        # ESC pressed
#        print("Escape hit, closing...")
#        break
#    elif k%256 == 32:
#        # SPACE pressed
#        img_name = "opencv_frame_1{}.png".format(img_counter)
#        cv2.imwrite(img_name, frame)
#        print("{} written!".format(img_name))
#        getExpr(img_name)
#        img_counter += 1
#
#cam.release()
#
#cv2.destroyAllWindows()
#
#
#from skimage import io
#img = tf.keras.preprocessing.image.load_img("opencv_frame_0.png", color_mode = "grayscale", target_size=(48, 48))
#show_img = tf.keras.preprocessing.image.load_img("opencv_frame_0.png", grayscale=False, target_size=(200, 200))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis = 0)
#
#x /= 255
#
#custom = model.predict(x)
#emotion_analysis(custom[0])
#
#x = np.array(x, 'float32')
#x = x.reshape([48, 48]);
#
#plt.gray()
#plt.imshow(show_img)
#plt.show()
#
#m = 0.000000000000000000001
#a = custom[0]
#for i in range(0,len(a)):
#    if a[i] > m:
#        m = a[i]
#        ind = i
#
#print('Expression Prediction:',objects[ind])
#
#
#cam = cv2.VideoCapture(0)
#
#cv2.namedWindow("test")
#
#img_counter = 0
#
#while True:
#    ret, frame = cam.read()
#    if not ret:
#        print("failed to grab frame")
#        break
#    cv2.imshow("test", frame)
#
#    k = cv2.waitKey(1)
#    if k%256 == 27:
#        # ESC pressed
#        print("Escape hit, closing...")
#        break
#    elif k%256 == 32:
#        # SPACE pressed
#        img_name = "opencv_frame_1{}.png".format(img_counter)
#        cv2.imwrite(img_name, frame)
#        print("{} written!".format(img_name))
#        getExpr(img_name)
#        img_counter += 1
#
#cam.release()
#
#cv2.destroyAllWindows()
#
#
#cam = cv2.VideoCapture(0)
#
#cv2.namedWindow("test")
#
#img_counter = 0
#
#while True:
#    ret, frame = cam.read()
#    if not ret:
#        print("failed to grab frame")
#        break
#    cv2.imshow("test", frame)
#
#    img_name = "opencv_frame_1{}.png".format(img_counter)
#    cv2.imwrite(img_name, frame)
#    print("{} written!".format(img_name))
#    getExpr(img_name)
#    img_counter += 1
#
#cam.release()
#
#cv2.destroyAllWindows()










