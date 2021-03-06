#%%
MODEL_NAME = 'ATTACK-LeNet5'


#%%
import gzip
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from common.captcha import CaptchaGenerator
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, BatchNormalization

inputs = Input(shape=(45, 155, 1))

x = Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)
x = BatchNormalization()(x)
x = AveragePooling2D(pool_size=2)(x)
x = Conv2D(24, kernel_size=(3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = AveragePooling2D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(84, activation='relu')(x)
x = BatchNormalization()(x)

yy = [Dense(10, activation='softmax', name=f'y{i}')(x) for i in range(1, 6)]

model = Model(inputs=inputs, outputs=[*yy], name=MODEL_NAME)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

captcha = CaptchaGenerator(0.7)
captcha_train = captcha.train_generator(4096)
captcha_valid = captcha.validate_generator(256)

history = pd.DataFrame()

for i in range(100):
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

print('MAX training   accuracy: {:.5f}%'.format(history.loc[:, 'y1_acc':'y5_acc'].prod(axis=1).multiply(100).max()))
print('MAX validation accuracy: {:.5f}%'.format(history.loc[:, 'val_y1_acc':'val_y5_acc'].prod(axis=1).multiply(100).max()))


#%%
from common.captcha import CaptchaGenerator
from tensorflow.compat.v1.keras.models import load_model
import numpy as np

model = load_model(f'CheckPoint/{MODEL_NAME}-021')
captcha = CaptchaGenerator(1, 'test')
captcha_test = captcha.test_generator()

test_data = next(captcha_test)

acc = model.evaluate(x=test_data[0], y=test_data[1])
print('Test accuracy: {:.5f}%'.format(np.prod(acc[-5:])))


#%%
train = train_data[0]
labels = [train_data[1][f'y{i}'].argmax(axis=1) for i in range(1, 6)]
labels = list(zip(*labels))
expected = model.predict(train)
print(expected[0])
print(labels[0])
#for i, label in enumerate(labels):
#    print(i, label)
#    plt.imshow(train[i].reshape([45, 155]))
#    plt.show()

expected_labels = [expected[i].argmax(axis=1) for i in range(5)]
expected_labels = list(zip(*expected_labels))
for i in range(255):
    print(f'expected - {expected_labels[i]}, real - {labels[i]}')
    plt.imshow(train[i].reshape([45, 155]))
    plt.show()

