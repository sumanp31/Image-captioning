import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input

def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def all_img_captions(filename):
    # create dictionary with all imgs with their captions
    file = load_doc(filename)
    captions = file.split('\n')
    des_dict ={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in des_dict:
            des_dict[img[:-2]] = [caption]
        else:
            des_dict[img[:-2]].append(caption)
    return des_dict


def cleaning_text(captions):
    #Data cleaning- lower casing, removing puntuations and words containing numbers

    table = str.maketrans('','',string.punctuation) #removes punctuations
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):
            img_caption.replace("-"," ")
            desc = img_caption.split()
            desc = [word.lower() for word in desc] #converts to lowercase
            desc = [word.translate(table) for word in desc] #remove punctuation from each token
            desc = [word for word in desc if(len(word)>1)]  #remove single letters
            desc = [word for word in desc if(word.isalpha())]  ##remove tokens with numbers in them
            img_caption = ' '.join(desc) #convert back to string
            captions[img][i]= img_caption
    return captions

def text_vocabulary(des_dict):
    # build vocabulary of all unique words
    vocab = set()
    for key in des_dict.keys():
        [vocab.update(d.split()) for d in des_dict[key]]
    return vocab

def save_des_dict(des_dict, filename):
    #All description in one file 
    lines = list()
    for key, desc_list in des_dict.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

dataset_images = "Flicker8k_Dataset"
filename = "Flickr8k.token.txt"
#loading the file that contains all data
#mapping them into descriptions dictionary img to 5 captions
des_dict = all_img_captions(filename)
print("Length of descriptions =" ,len(des_dict))

clean_des_dict = cleaning_text(des_dict)

vocabulary = text_vocabulary(clean_des_dict)
print("Length of vocabulary = ", len(vocabulary))
#saving each description to file 
save_des_dict(clean_des_dict, "descriptions.txt")


def extract_features(directory):
    model = Xception( include_top=False, pooling='avg' )
    features = {}
    i = 1
    for img in os.listdir(directory):
        print (i)
        i += 1
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        features[img] = feature
    return features

features = extract_features(dataset_images)
dump(features, open("features.p","wb"))