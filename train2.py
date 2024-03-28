from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

# загрузка содержимого файла в память
def load_doc(filename):
    # открытие файла только для чтения
    file = open(filename, 'r')
    # чтение всего текста
    text = file.read()
    # закрытие файла
    file.close()
    return text

# загрузка предопределенного списка идентификаторов фотографий
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # обработка построчно
    for line in doc.split('\n'):
        # пропуск пустых строк
        if len(line) < 1:
            continue
        # получение идентификатора изображения
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# загрузка чистых описаний в память
def load_clean_descriptions(filename, dataset):
    # загрузка документа
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # разделение строки по пробелам
        tokens = line.split()
        # разделение идентификатора и описания
        image_id, image_desc = tokens[0], tokens[1:]
        # пропуск изображений, не присутствующих в наборе
        if image_id in dataset:
            # создание списка
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # добавление описания с токенами
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # сохранение описания
            descriptions[image_id].append(desc)
    return descriptions

# загрузка признаков фотографий
def load_photo_features(filename, dataset):
    # загрузка всех признаков
    all_features = load(open(filename, 'rb'))
    # фильтрация признаков
    features = {k: all_features[k] for k in dataset}
    return features

# преобразование словаря чистых описаний в список описаний
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# создание токенизатора на основе описаний
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# определение максимальной длины описания
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# создание последовательностей изображений, входных последовательностей и выходных слов для изображения
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    # обработка каждого описания для изображения
    for desc in desc_list:
        # кодирование последовательности
        seq = tokenizer.texts_to_sequences([desc])[0]
        # разделение одной последовательности на несколько пар X, y
        for i in range(1, len(seq)):
            # разделение на входную и выходную пару
            in_seq, out_seq = seq[:i], seq[i]
            # заполнение входной последовательности
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # кодирование выходной последовательности
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # сохранение
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

# определение модели генерации описаний
def define_model(vocab_size, max_length):
    # модель извлечения признаков
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # модель последовательности
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # модель декодирования
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # объединение моделей [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # компиляция модели
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # суммарное описание модели
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# генератор данных, используемый в вызове model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    # бесконечный цикл по изображениям
    while 1:
        for key, desc_list in descriptions.items():
            # извлечение признаков изображения
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]

# загрузка обучающего набора данных (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Набор данных: %d' % len(train))
# описания
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Описания: train=%d' % len(train_descriptions))
# признаки фотографий
train_features = load_photo_features('features.pkl', train)
print('Фотографии: train=%d' % len(train_features))
# создание токенизатора
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Размер словаря: %d' % vocab_size)
# определение максимальной длины описания
max_length = max_length(train_descriptions)
print('Длина описания: %d' % max_length)

# определение модели
model = define_model(vocab_size, max_length)
# обучение модели, запуск эпох вручную и сохранение после каждой эпохи
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    # создание генератора данных
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    # обучение на одну эпоху
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # сохранение модели
    model.save('model_' + str(i) + '.h5')
