import sys
sys.path.append('../')
import os 
import spacy
import es_core_news_lg
import argparse
import codecs
import time
from request_data import samples_loader_from_minio
from utils.dataset_utils import simplify_entity
from segtok.segmenter import split_single
from flair.data import Sentence
from utils.general_utils import boolean_string



def tokenize_with_spacy(text, entities, tokenizer, lower_tokens=False, no_accent_marks=False, referral_name=None):
    """ 
    The given text is tokenized prioritizing not to lose entities that ends in the middle of a word.
    This because on many occasions words are stick together in a free text.
    """ 
    idx = 0
    no_tagged_tokens_positions = []
    tagged_tokens_positions = [(entity['start_idx'], entity['end_idx']) for entity in entities]
    entity_tokens = spacy_tokens(text, tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)  
    for tagged_token in tagged_tokens_positions:
        no_tagged_tokens_positions.append((idx, tagged_token[0])) # We add text before tagged token
        idx = tagged_token[1] 
    no_tagged_tokens_positions.append((idx, len(text)))           # We add text from last token tagged end possition to end of text.
    no_entity_tokens = spacy_tokens(text, no_tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)
    tokens = sorted(entity_tokens+no_entity_tokens, key=lambda entity:entity["start_idx"])
    return [tokens]




def spacy_tokens(text, pos_list, tokenizer, lower_tokens, no_accent_marks): 
    """ 
    Given a list of pairs of start-end positions in the text, 
    the text within these positions is tokenized and returned in tokens array.
    """
    tokens = []
     # TODO: Add new tokenizers (e.g, Nltk) to compare performance.
    for poss in pos_list:
        text_tokenized = tokenizer(text[poss[0]:poss[1]])
        for span in text_tokenized.sents:
            sentence = [text_tokenized[i] for i in range(span.start, span.end)]
            for token in sentence:
                token_dict = {}
                token_dict['start_idx'] = token.idx + poss[0]
                token_dict['end_idx'] = token.idx + poss[0] + len(token)
                token_dict['text'] = text[token_dict['start_idx']:token_dict['end_idx']]
                if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                    continue
                if len(token_dict['text'].split(' ')) != 1:
                    token_dict['text'] = token_dict['text'].replace(' ', '-')
                # TODO: Before adding token to token list, process irregular tokens with custom parsing.
                if lower_tokens: token_dict['text'] = token_dict['text'].lower()
                if no_accent_marks: token_dict = remove_accent_mark(token_dict)
                tokens.append(token_dict)
    
    return tokens

def remove_accent_mark(token_dict):
    try:
        token_dict['text'] = token_dict['text'].replace('á','a')
        token_dict['text'] = token_dict['text'].replace('é','e')
        token_dict['text'] = token_dict['text'].replace('í','i')
        token_dict['text'] = token_dict['text'].replace('ó','o')
        token_dict['text'] = token_dict['text'].replace('ú','u')
        return token_dict
    except:
        return token_dict

def get_nested_entities(annotation, referral):  
    """ 
    Given a text and its annotation file, it returns all inner and outer entities annotated.
    """
    entities = []
    for line in annotation.splitlines():
        entity_info = {}
        entity = line.split()
        if entity[0].startswith('T') and not ';' in entity[3]:
            entity_info['label'] = simplify_entity(entity[1]) 
            entity_info['start_idx'] = int(entity[2])
            entity_info['end_idx'] = int(entity[3])
            entity_info['text'] = referral[1][int(entity[2]): int(entity[3])] # Llego con la nueva entidad posiblemente a agregar.
            add = True # Booleano para saber si la incorporo o no 

            for entity_added in entities: # Por cada una de las entidades ya agregadas
                if entity_info['label']==entity_added['label'] and entity_info['start_idx']>=entity_added['start_idx'] and entity_info['end_idx']<=entity_added['end_idx']: # En caso que sea del mismo tipo y este anidada dentro de la otra, no se agrega.
                    add = False 
                    break

                elif entity_info['label']==entity_added['label'] and ((entity_info['start_idx']<entity_added['start_idx'] and entity_info['end_idx']>=entity_added['end_idx']) or (entity_info['start_idx']<=entity_added['start_idx'] and entity_info['end_idx']>entity_added['end_idx'])): 
                    add = False
                    entities.remove(entity_added)
                    entities.append(entity_info)
                    break
            if add and not colapse_with_others(entities, entity_info): 
                entities.append(entity_info)
    
    
    return entities



def get_flat_entities(annotations, referral):
    eliminate = {}
    for ann in annotations:
        for ann2 in annotations:
            if ann is ann2:
                continue
            if ann2['start_idx'] >= ann['end_idx'] or ann2['end_idx'] <= ann['start_idx']:
                continue 
            if eliminate.get((ann['label'], ann['start_idx'], ann['end_idx'])) or eliminate.get((ann['label'], ann2['start_idx'], ann2['end_idx'])):
                continue
            elim, keep = eliminate_and_keep(ann, ann2)
            eliminate[(elim['label'], elim['start_idx'], elim['end_idx'])] = True
    flat_entities = [anno for anno in annotations if not (anno['label'], anno['start_idx'], anno['end_idx']) in eliminate]
    return flat_entities

def colapse_with_others(entities, entity_info):
    for entity in entities:
        if (entity_info['start_idx']<entity['start_idx'] and entity_info['end_idx']>entity['start_idx'] and entity_info['end_idx']<entity['end_idx'])\
            or (entity_info['start_idx'] > entity['start_idx'] and entity_info['start_idx'] < entity['end_idx'] and entity_info['end_idx'] > entity['end_idx']):
            return True
    return False

def eliminate_and_keep(ann, ann2):
    if (ann['start_idx'], ann['end_idx']) == (ann2['start_idx'], ann2['end_idx']):
        return ann2, ann
    elif ann['end_idx']-ann['start_idx'] == ann2['end_idx']-ann2['start_idx']:
        return ann2, ann
    else:
        if ann['end_idx']-ann['start_idx'] < ann2['end_idx']-ann2['start_idx']:
            return ann, ann2
        else:
            return ann2, ann 



def convert_to_conll(referrals, annotations, tokenizer_type, lower_tokens, no_accent_marks, include_path, output_path):
    """ 
    Function used to create conll file format from ann-txt annotations.
    """
    output_file = codecs.open(output_path, 'w', 'UTF-8')
    if tokenizer_type =='spacy': tokenizer = spacy.load('es_core_news_lg', disable = ['ner', 'tagger'])
    for referral, annotation in zip(referrals, annotations):
        if include_path: output_file.write(referral[0])
        nested_entities = get_nested_entities(annotation[1], referral)
        nested_entities = sorted(nested_entities, key = lambda entity: entity["start_idx"])
        flat_entities = get_flat_entities(nested_entities, referral)
        flat_entities = sorted(flat_entities, key = lambda entity: entity["start_idx"])
        
        if tokenizer_type == 'spacy': sentences = tokenize_with_spacy(referral[1], flat_entities, tokenizer, lower_tokens, no_accent_marks, referral_name= referral[0])

        for sentence in sentences:
            inside_entity = {'Disease': 0, 'Abbreviation': 0, 'Finding': 0, 'Procedure': 0, 'Body_Part': 0, 'Family_Member': 0, 'Medication': 0}
            for i, token in enumerate(sentence):
                token['label'] = 'O'
                token_labels = []
                for entity in nested_entities:
                    
                    if token['start_idx'] < entity['start_idx']:
                        continue
                    
                    elif token['end_idx'] == entity['end_idx'] and token['start_idx'] == entity['start_idx'] or \
                        (token['start_idx']==entity['start_idx'] and token['text']==' '.join(entity['text'].split())) \
                        or (token['end_idx']==entity['end_idx'] and token['text']==' '.join(entity['text'].split())):

                        inside_entity[entity['label']] = 0
                        token_labels.append('B-' + entity['label'])

                    elif token['end_idx'] < entity['end_idx'] and not inside_entity[entity['label']]:
                        inside_entity[entity['label']] = 1
                        token_labels.append('B-' + entity['label'])

                    elif token['end_idx'] < entity['end_idx'] and inside_entity[entity['label']]:
                        token_labels.append('I-' + entity['label'])

                    elif token['end_idx'] == entity['end_idx'] and not inside_entity[entity['label']]:
                        inside_entity[entity['label']]=0
                        token_labels.append('B-' + entity['label'])
                        
                    elif token['end_idx'] == entity['end_idx'] and inside_entity[entity['label']]:
                        inside_entity[entity['label']]=0
                        token_labels.append('I-' + entity['label'])

                    elif entity['start_idx']>token['end_idx']:
                        break
                    else: 
                        continue

                if len(token_labels)!=0:
                    token_labels = sorted(token_labels, key=lambda entity:entity, reverse=True)
                    output_file.write(f"{token['text']} {' '.join(token_labels)}\n")

                elif token['text']=='.' and i!=len(sentence)-1:
                    output_file.write(f"{token['text']} {token['label']}\n\n")

                else:
                    output_file.write(f"{token['text']} {token['label']}\n")
            output_file.write('\n')
    output_file.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type = str, required = True)
    parser.add_argument('--access_key', type = str, required = True)
    parser.add_argument('--secret_key', type = str, required = True)
    parser.add_argument('--n_annotations', type = int, default = 4000, required = False)
    parser.add_argument('--tokenizer', type = str, default = 'spacy', required = False)
    parser.add_argument('--lower_tokens', default=False, type=boolean_string)
    parser.add_argument('--no_accent_marks', default=False, type=boolean_string)
    parser.add_argument('--include_path', default=False, type=boolean_string)
    parser.add_argument('--output_filename', type = str, default = 'entities', required = False)
    
    args = parser.parse_args()  
    server = args.server
    access_key = args.access_key
    secret_key = args.secret_key
    n_annotations = args.n_annotations
    tokenizer = args.tokenizer
    lower_tokens = args.lower_tokens
    no_accent_marks = args.no_accent_marks
    include_path = args.include_path
    output_filename = args.output_filename

    actual_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(actual_path, f'conll/{output_filename}.conll')
    referrals, annotations = samples_loader_from_minio(server, access_key, secret_key, n_annotations)
    convert_to_conll(referrals, annotations, tokenizer, lower_tokens, no_accent_marks, include_path, output_path)