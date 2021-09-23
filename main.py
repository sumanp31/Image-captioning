import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_photos(filename):
    #load the data 
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos

def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = " ".join(image_caption)
            descriptions[image].append(desc)
    return descriptions

def load_features(photos):
    #loading all features
    features_mat = load(open("features.p","rb"))
    features = {k:features_mat[k] for k in photos}
    return features

filename = "Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)



def dict_to_list(descriptions):
    # converting dictionary to clean list of descriptions
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

from keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):

    #creating tokenizer class to  vectorise text corpus where each integer will represent token in dictionary
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(train_descriptions)

            
def create_sequences(tokenizer, max_length, desc_list, feature, X1, X2, y):
    # walk through each description for the image
    for desc in desc_list[:3]:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return X1, X2, y
#You can check the shape of the input and output for your model

def data_generator(descriptions, features, tokenizer, max_length):
    X1, X2, y = list(), list(), list()
    i = 1
    for key, description_list in descriptions.items():
        #retrieve photo features
        feature = features[key][0]
        input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature, X1, X2, y)
        # print (input_image)
    return [[np.array(input_image), np.array(input_sequence)], np.array(output_word)]


from keras.utils.vis_utils import plot_model

def define_model(vocab_size, max_length):
    # define the captioning model
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    # fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(inputs1)
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    # se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se1)
    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
model = define_model(vocab_size, max_length)
steps = len(train_descriptions)
# making a directory models to save our models

[x1,x2], y = data_generator(train_descriptions, train_features, tokenizer, max_length)
print ("X and y created")
print (x1.shape, x2.shape, y.shape)
model.fit([x1, x2], y, epochs=20,steps_per_epoch= steps, verbose=1)
model.save("models/model" + ".h5")