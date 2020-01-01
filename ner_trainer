#THIS MODEL IS UPDATED TO ADAPT IN JSON FILES
from __future__ import unicode_literals, print_function
import plac
from pathlib import Path
import spacy
import random
import json
import io

LABEL = 'Company'
TRAIN_DATA = []
with io.open('company_test_data.json', 'r', encoding='utf-8-sig') as json_file:
    data = json.load(json_file)
    for p in data['rasa_nlu_data']['common_examples']:
        text = p['text']
        entity_arr = []
        for entity in p['entities']:
            entity_arr.append((entity['start'], entity['end'], entity['entity']))
        TRAIN_DATA.append((p['text'], {'entities': entity_arr}))
print(TRAIN_DATA)

def train_spacy(model, data, iterations):
    TRAIN_DATA = data
    
    if model is not None:
        nlp = spacy.load(model) #load the existing spacy model
        print("Loaded Model '%s'" % model)
    else:
        nlp = spacy.blank('en') #creates a blank language class
        print("Created a blank 'en' spacy model.")
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')   #creating the pipeline 
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe('ner')
        
    ner.add_label(LABEL)   #add new entity label to entitiy recognizer
            
    if model is None:
        optimizer = nlp.begin_training()
    else:
        #begin_training initializes the models, so it ZEROES OUT existing entity types
        optimizer = nlp.entity.create_optimizer()
    
    #get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): #disables the other pipes and only trains the NER
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text], #batch of texts
                    [annotations], #batch of annotations
                    drop = 0.35, #dropout - make it harder to memorize the data
                    sgd = optimizer, #callable to update the weights
                    losses = losses)
            print('Losses',losses)
    return nlp
    
    product_nlp = train_spacy('en', TRAIN_DATA, 20)
    
#saving the model
output_dir = Path("Desktop")
product_nlp.to_disk(output_dir)
print("Saved model to", output_dir)

#testing the saved model
print("Loading from", output_dir)
nlp_trained = spacy.load(output_dir)
test_txt = input("Enter some training text: ")
doc = nlp_trained(test_txt)
for ent in doc.ents:
    print(ent.text, ent.label_)

