import os
import gzip
import pickle
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.keras.utils import to_categorical

CAPTCHA_SIZE = (155, 45)
CAPTCHA_LEN = 5

class CaptchaGenerator():
    IMAGE_BASE = f'../../CAPTCHA/DATA'
    _captchas = None
    _train_len = 0
    _directory = None
    
    def __init__(self, train_ratio, directory=None):
        cnt = 0
        captchas = set([])
        self._directory = directory
        
        if directory is not None:
            img_path = self.IMAGE_BASE + '/' + directory
            dl = set(np.array(os.listdir(img_path)))
            captchas = captchas | dl
        else:
            for i in range(10):
                img_path = self.IMAGE_BASE + f'/{i}'
                dl = set(np.array(os.listdir(img_path)))
                captchas = captchas | dl
                
        self._captchas = np.array(list(captchas))
        self._train_len = int(len(captchas)*train_ratio)

        np.random.shuffle(self._captchas)
        
        print("total data :", len(self._captchas))
        print("train count:", self._train_len)

    def captcha_generator(self, is_train, batch_size):
        images = np.empty([batch_size, CAPTCHA_SIZE[1], CAPTCHA_SIZE[0]], dtype=np.float)
        labels = np.empty([batch_size, CAPTCHA_LEN], dtype=np.float)

        sel = self._captchas[:self._train_len] if is_train else self._captchas[self._train_len:]
        for j in range(batch_size):
            fname = np.random.choice(sel, 1)[0]
            path = f'{self.IMAGE_BASE}'
            path += f'/{self._directory}/{fname}' if self._directory is not None else f'/{fname[0]}/{fname}'
            
            im = Image.open(path)
            im = im.convert('P', palette=Image.ADAPTIVE)
            images[j] = np.array(im)
            im.close()
            labels[j, :] = np.array(list(fname)[:5], dtype=np.float)
            
        return (images.reshape([images.shape[0], images.shape[1], images.shape[2], 1]),
                to_categorical(labels, num_classes=10))
            
    def train_generator(self, batch_size=64):
        while True:
            gen = self.captcha_generator(True, batch_size)
            yield (gen[0], {'y1':gen[1][:, 0], 
                            'y2':gen[1][:, 1], 
                            'y3':gen[1][:, 2], 
                            'y4':gen[1][:, 3], 
                            'y5':gen[1][:, 4]})
    
    def validate_generator(self, batch_size=64):
        while True:
            gen = self.captcha_generator(False, batch_size)
            yield (gen[0], {'y1':gen[1][:, 0], 
                            'y2':gen[1][:, 1], 
                            'y3':gen[1][:, 2], 
                            'y4':gen[1][:, 3], 
                            'y5':gen[1][:, 4]})
            
    def test_generator(self):
        while True:
            gen = self.captcha_generator(True, self._train_len)
            yield (gen[0], {'y1':gen[1][:, 0], 
                            'y2':gen[1][:, 1], 
                            'y3':gen[1][:, 2], 
                            'y4':gen[1][:, 3], 
                            'y5':gen[1][:, 4]})


def fit_model(model, train_ratio, directory=None):
    captcha = CaptchaGenerator(0.7, 'static') # static CAPTCHA
    captcha_train = captcha.train_generator(1024)
    captcha_valid = captcha.validate_generator(128)

    history = pd.DataFrame()

    for i in range(30):
        print(f'* Round {i+1}...', datetime.now().isoformat())
        train_data = next(captcha_train)
        valid_data = next(captcha_valid) 
        hist = model.fit(x=train_data[0], 
                        y=train_data[1],
                        validation_data=(valid_data[0], valid_data[1]),
                        epochs=1, batch_size=64, verbose=0)
        history = history.append(pd.DataFrame(hist.history), ignore_index=True)
        
    with gzip.open(f'{model.name}_history.pickle', 'wb') as f:
        pickle.dump(history, f)
    print('all done', datetime.now().isoformat())