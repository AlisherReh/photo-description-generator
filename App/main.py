from kivy.uix.label import Label
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
from pickle import load
import time

from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from kivy.core.window import Window
from kivy.uix.image import Image
from kivymd.uix.list import MDList, OneLineAvatarListItem, ImageLeftWidget

Window.size = (350, 700)

Builder.load_file("homescreen.kv")

import pyttsx3
from translate import Translator

class Homescreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mycamera = self.ids.camera
        self.myimage = Image()
        self.resultbox = self.ids.resultbox
        self.mybox = self.ids.mybox

        # Инициализация pyttsx3
        self.engine = pyttsx3.init()

        # Инициализация переводчика
        self.translator = Translator(to_lang="Russian")

        # Язык по умолчанию
        self.language = 'en'

        # Загрузка токенизатора и модели
        self.tokenizer = load(open('tokenizer.pkl', 'rb'))
        self.max_length = 34
        self.model = load_model('model_6.h5')

    def extract_features(self, filename):
        # Загрузка модели VGG16
        model = VGG16()
        # Перестроение модели для удаления последнего слоя
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # Загрузка и предварительная обработка изображения
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # Извлечение признаков
        feature = model.predict(image, verbose=0)
        return feature

    def captureyouface(self):
        # Захват текущего времени для уникального имени файла
        timenow = time.strftime("%Y%m%d_%H%M%S")
        # Захват изображения с камеры
        self.mycamera.export_to_png(f"myimage_{timenow}.png")
        self.myimage.source = f"myimage_{timenow}.png"

        # Загрузка и подготовка фотографии для генерации описания
        photo = self.extract_features(f"myimage_{timenow}.png")
        
        # Генерация описания
        description = self.generate_desc(photo)

        # Перевод описания, если выбран русский язык
        if self.language == "ru":
            description_text = self.translate_description(description)
        else:
            description_text = description
        
        # Озвучивание описания
        self.speak_description(description_text)
        
        # Добавление элемента списка с изображением и его описанием
        self.resultbox.add_widget(
            OneLineAvatarListItem(
                ImageLeftWidget(
                    source=f"myimage_{timenow}.png",
                    size_hint_x=0.3,
                    size_hint_y=1,
                    size=(300, 300)
                ),
                text=description_text  # Добавление описания в качестве текста
            )
        )

    def generate_desc(self, photo):
        # Начало процесса генерации последовательности
        in_text = 'startseq'
        # Итерация по длине последовательности
        for i in range(self.max_length):
            # Кодирование входной последовательности
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            # Дополнение входной последовательности
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            # Прогнозирование следующего слова
            yhat = self.model.predict([photo, sequence], verbose=0)
            # Преобразование вероятности в целое число
            yhat = argmax(yhat)
            # Преобразование целого числа в слово
            word = self.word_for_id(yhat)
            # Остановка, если преобразование слова не удалось
            if word is None:
                break
            # Добавление слова к входной последовательности
            in_text += ' ' + word
            # Остановка, если предсказано 'endseq'
            if word == 'endseq':
                break
        return in_text

    def word_for_id(self, integer):
        # Преобразование целого числа в слово
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def translate_description(self, description):
        try:
            # Перевод описания
            translation = self.translator.translate(description)
            return translation
        except Exception as e:
            print("Ошибка при переводе описания:", e)
            return description  # Возвращение оригинального описания, если перевод не удался

    def speak_description(self, description):
        # Озвучивание описания с помощью pyttsx3
        self.engine.say(description)
        self.engine.runAndWait()

    def switch_language(self):
        # Переключение между английским и русским языками
        if self.language == 'en':
            self.language = 'ru'
            self.ids.language_label.text = "Language: Russian"
        else:
            self.language = 'en'
            self.ids.language_label.text = "Language: English"

class MyApp(MDApp):
    def build(self):
        return Homescreen()

if __name__ == "__main__":
    MyApp().run()
