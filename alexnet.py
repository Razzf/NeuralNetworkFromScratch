
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D, BatchNormalization
 
import numpy as np
np.random.seed(1000)

model = Sequential()

model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())


model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())


model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())


model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(38))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
validation_split=0.2, shuffle=True)

def inception_module(x, fs_1x1, fs_3x3, fs_5x5, fs_pool_proj, fs_3x3_rd, fs_5x5_rd, name=None):
    
    conv_1x1 = Conv2D(fs_1x1, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    
    conv_3x3 = Conv2D(fs_3x3_rd, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(fs_3x3, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(fs_5x5_rd, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(fs_5x5, (5, 5), strides=(1, 1), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(fs_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output


input_layer = Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x, fs_1x1=64, fs_3x3_rd=96, fs_3x3=128, fs_5x5_rd=16, fs_5x5=32, fs_pool_proj=32, name='inception_3a')

x = inception_module(x,fs_1x1=128, fs_3x3_rd=128, fs_3x3=192, fs_5x5_rd=32, fs_5x5=96, fs_pool_proj=64, name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x, fs_1x1=192, fs_3x3_rd=96, fs_3x3=208, fs_5x5_rd=16, fs_5x5=48, fs_pool_proj=64, name='inception_4a')

x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x, fs_1x1=160, fs_3x3_rd=112, fs_3x3=224, fs_5x5_rd=24, fs_5x5=64, fs_pool_proj=64, name='inception_4b')

x = inception_module(x, fs_1x1=128, fs_3x3_rd=128, fs_3x3=256, fs_5x5_rd=24, fs_5x5=64, fs_pool_proj=64, name='inception_4c')

x = inception_module(x, fs_1x1=112, fs_3x3_rd=144, fs_3x3=288, fs_5x5_rd=32, fs_5x5=64, fs_pool_proj=64, name='inception_4d')

x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x, fs_1x1=256, fs_3x3_rd=160, fs_3x3=320, fs_5x5_rd=32, fs_5x5=128, fs_pool_proj=128, name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x, fs_1x1=256, fs_3x3_rd=160, fs_3x3=320, fs_5x5_rd=32, fs_5x5=128, fs_pool_proj=128, name='inception_5a')

x = inception_module(x, fs_1x1=384, fs_3x3_rd=192, fs_3x3=384, fs_5x5_rd=48, fs_5x5=128, fs_pool_proj=128, name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(10, activation='softmax', name='output')(x)