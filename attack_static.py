#%%
MODEL_NAME = 'ATTACK-STATIC'


#%%
import gzip
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from common.captcha import CaptchaGenerator
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

inputs = Input(shape=(45,155,1))

x = Conv2D(6, kernel_size=(3, 3), activation='relu')(inputs)
x = Conv2D(3, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)

x = Dense(196, activation='relu')(x)
x = Dense(128, activation='relu')(x)

yy = [Dense(10, activation='softmax', name=f'y{i}')(x) for i in range(1, 6)]

model = Model(inputs=inputs, outputs=[*yy], name=MODEL_NAME)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['accuracy'])

captcha = CaptchaGenerator(0.7, 'static') # static CAPTCHA
captcha_train = captcha.train_generator(4096)
captcha_valid = captcha.validate_generator(256)

history = pd.DataFrame()

for i in range(20):
    print(f'* Round {i+1}...', datetime.now().isoformat())
    train_data = next(captcha_train)
    valid_data = next(captcha_valid) 
    hist = model.fit(x=train_data[0], 
                    y=train_data[1],
                    validation_data=(valid_data[0], valid_data[1]),
                    epochs=1, batch_size=64, verbose=0)
    history = history.append(pd.DataFrame(hist.history), ignore_index=True)
    model.save(f'CheckPoint/{model.name}-{i:03}')

with gzip.open(f'{model.name}_history.pickle', 'wb') as f:
    pickle.dump(history, f)
print('all done', datetime.now().isoformat())


#%%
import gzip
import pickle
import matplotlib.pyplot as plt

with gzip.open(f'{MODEL_NAME}_history.pickle','rb') as f:
    history = pickle.load(f)

plt.rc('font', size=12)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
history.loc[:, 'y1_acc':'y5_acc'].prod(axis=1).multiply(100).plot(label='Training Accuracy', alpha=0.8, lw=0.8, marker='o', fillstyle='none')
history.loc[:, 'val_y1_acc':'val_y5_acc'].prod(axis=1).multiply(100).plot(label='Validation Accuracy', alpha=0.8, lw=0.8, marker='x')
plt.xticks(range(0, len(history)+1, len(history)//4))
plt.legend()
plt.show()


#%%
from common.captcha import CaptchaGenerator
from tensorflow.compat.v1.keras.models import load_model

model = load_model(f'CheckPoint/{MODEL_NAME}-019')
captcha = CaptchaGenerator(1, 'static')
captcha_test = captcha.test_generator()

test_data = next(captcha_test)

model.evaluate(x=test_data[0],
               y=test_data[1])