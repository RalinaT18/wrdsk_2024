# wrdsk_2024
import numpy as np
from keras.models import Model
from keras.layers import Dense
import io
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from keras.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D

#Метрика (не рабочая)
class MulticlassAUC(tf.keras.metrics.AUC):

    def __init__(self, pos_label, from_logits=False, sparse=True, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):

        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)

#Загрузка архитектуры
EFNB3 = EfficientNetB3(include_top=False, input_shape=(224,224,3))
EFNB3.trainable=False

#Модель
for l in EFNB3.layers:
  l.trainable=False

x=GlobalAveragePooling2D()(EFNB3.output)
x=Dense(units=4096, activation='relu')(x)
x=Dense(units=2048, activation='relu')(x)
x=Dense(units=256, activation='relu')(x)
x=Dense(units=14, activation='softmax')(x)
EFNB3=Model(inputs=EFNB3.input, outputs=x)

#Сборка модели
EFNB3.compile(
              optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy',
                                                                        MulticlassAUC(pos_label=0)])

#Загрузка весов
EFNB3.load_weights("EFNB3_tomato_leaf.h5")

#Загрузка фото
img1 = cv2.imread("/content/drive/MyDrive/Проект/Да-да, начинаем диплом и практику/Tomato growth stage/2. Growth/Growth (47).jpg")
img1 = cv2.resize(img1,(224, 224))

img1 = np.array(img1)
img1.shape
img1=img1.reshape(1,224,224,3)

predictions = EFNB3.predict(img1)
print(predictions)
indices = predictions.argmax()

if indices==0: print('Bacterial_spot')
elif indices==1: print('Early_blight')
elif indices==2: print('Late_blight')
elif indices==3: print('Leaf_Mold')
elif indices==4: print('Septoria_leaf_spot')
elif indices==5: print('Spider_mites Two-spotted_spider_mite')
elif indices==6: print('Target_Spot')
elif indices==7: print('Tomato_Yellow_Leaf_Curl_Virus')
elif indices==8: print('Tomato_mosaic_virus')
elif indices==9: print('healthy')
else: print('powdery_mildewy')
