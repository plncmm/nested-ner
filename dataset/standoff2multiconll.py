import os 
import spacy
import es_core_news_lg
import argparse
import codecs
import time
import sys
sys.path.append('../')
from request_data import samples_loader_from_minio
from utils.utils import simplify_entity


windows_linux_ambiguity = ['040967d8fcd7b1216f5508e6247feb56.txt', '0b2fcd9d867b12bbdd1d545dbc7734a4.txt', '1193f738abc13af0b42682fcc8cc2856.txt', '12edbd22022e53aa49902142ab229ea0.txt', '20a1512af065810c085bfd739fb0b95c.txt', '27abd89e62838b138ef221105ed5ae79.txt', '29e458dfc7f9bde36e18d8b399d826c2.txt', '2eaa0bf10fe3df6b4a37998932bb3177.txt', '34aee0dc0704dab872f66f57f9853967.txt', '3f91371c8db625aebfcf7c9b107c9792.txt', '431fa0771dd908e4f2239ffe29432098.txt', '6304a13d6c690be03068101e6fba1d21.txt', '6449da6fe857120f4514c76b038d60df.txt', '689df12ff070addc5d5dd5f9f19de695.txt', '75a4f05d848b024118f965cfc2525fc7.txt', '790aa95c00964bb379f917946c05dc53.txt', '8307dcc315b2470f039df9c3e1f14824.txt', '8b68a631a40aea69dafe503111e819a3.txt', '8d96000f3c4e1d60a88ca412351528c4.txt', '959175742bb533cb09259a3e53302dae.txt', '960c136f298b1ed3f81aa4be6e472124.txt', '9d4785fb19b4d0270b6263fd2d4c8d3b.txt', 'a6732479b0dfd705db9354ca6b5d7b81.txt', 'a69c789e2ec9b0ae646112ab10bc09c4.txt', 'b2c8afc7cde5114381f480cde7bb51db.txt', 'c0d4a5ead6e1a904787ec8b9c77744eb.txt', 'c39b23e3c090ae43db5f396ad2cb5325.txt', 'c7293a5abd2026a69b8b5d30d8022be3.txt', 'ca2b3df2ee46c5154c6852996edfe681.txt', 'ce5f19da4af25901cde62c382c2a0223.txt', 'cef0017d4ce75c3877c448837b7ec6db.txt', 'd134661f20d9756551604d939204441c.txt', 'd2e5d05a5d29c2d7dbca07afdbb399b9.txt', 'd54787d3547597e0ca1da62913df88eb.txt', 'd5bfc338911ea3461f5db8566273935c.txt', 'd71280a811359eb2fae14f5aadfed3b6.txt', 'd776c02b2abdee36ae76816dfbec9cd6.txt', 'ead85756aa7f51b704c7c059150636c7.txt', 'f45adcff9848e90d597396d977dff310.txt', 'f878ef1e5a37fc61716455aa099b5fc2.txt', 'fcce9e8f75eff37a396da7ea81bcfb8c.txt']

def tokenize(text, entities, tokenizer, lower_tokens=False, no_accent_marks=False, referral_name=None):
    """ 
    The given text is tokenized prioritizing not to lose entities that ends in the middle of a word.
    This because on many occasions words are stick together in a free text.
    """
    
    idx = 0
    no_tagged_tokens_positions = []
    tagged_tokens_positions = [(entity['start_idx'], entity['end_idx']) for entity in entities]
   
    entity_tokens = tokenize_pos_list(text, tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)  
 
    for tagged_token in tagged_tokens_positions:
        no_tagged_tokens_positions.append((idx, tagged_token[0])) # We add text before tagged token
        idx = tagged_token[1] 
    
    no_tagged_tokens_positions.append((idx, len(text)))           # We add text from last token tagged end possition to end of text.
    no_entity_tokens = tokenize_pos_list(text, no_tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)
    tokens = sorted(entity_tokens+no_entity_tokens, key=lambda entity:entity["start_idx"])
    
    return [tokens]

def tokenize_pos_list(text, pos_list, tokenizer, lower_tokens, no_accent_marks): 
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

def get_nested_entities(annotation, referral, entity_types = None):  
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



def get_flat_entities(annotations, referral, entity_types = None):
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
        # Same length, pick by starting position
        return ann2, ann
    else:
        if ann['end_idx']-ann['start_idx'] < ann2['end_idx']-ann2['start_idx']:
            return ann, ann2
        else:
            return ann2, ann 

def convert_to_conll(referrals, annotations, output_path, tokenizer = None, entity_types = None, multiconll = False, lower_tokens=False, no_accent_marks=False, verbose=False):
    """ 
    Function used to create conll file format from ann-txt annotations.
    """
    tokenizer = spacy.load('es_core_news_lg', disable = ['ner', 'tagger'])
    output_file = codecs.open(output_path, 'w', 'UTF-8')
    
    for referral, annotation in zip(referrals, annotations):
        if referral[0] in windows_linux_ambiguity:
            continue
        nested_entities = get_nested_entities(annotation[1], referral, entity_types)
        nested_entities = sorted(nested_entities, key=lambda entity:entity["start_idx"])
        flat_entities = get_flat_entities(nested_entities, referral, entity_types)
        flat_entities = sorted(flat_entities, key=lambda entity:entity["start_idx"])
        
        sentences = tokenize(referral[1], flat_entities, tokenizer, lower_tokens, no_accent_marks, referral_name= referral[0])
        
        entities = nested_entities if multiconll else flat_entities
        for sentence in sentences:
            inside_entity = {'Disease': 0, 'Abbreviation': 0, 'Finding': 0, 'Procedure': 0, 'Body_Part': 0, 'Family_Member': 0, 'Medication': 0}
            for i, token in enumerate(sentence):
                token['label'] = 'O'
                token_labels = []
                for entity in entities:
                    
                    if token['start_idx'] < entity['start_idx']:
                        continue
                    
                    # Esta condición es nueva
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
    parser.add_argument('--server', type=str, required=False)
    parser.add_argument('--access_key', type=str, required=False)
    parser.add_argument('--secret_key', type=str, required=False)
    parser.add_argument('--region', type=str, required=False)
    parser.add_argument('--output_filename', type=str, required=False)
    parser.add_argument('--multi_conll', type=bool, default=False)
    parser.add_argument('--lower_tokens', type=bool, default=False)
    parser.add_argument('--no_accent_marks', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument(
        '-t', 
        '--types', 
        default=None,
        metavar='TYPE', 
        nargs='*', 
        help='Filter entities to given types')
    
    args = parser.parse_args()  
    server = args.server
    access_key = args.access_key
    secret_key = args.secret_key
    region = args.region
    output_filename = args.output_filename
    multiconll = args.multi_conll
    lower_tokens = args.lower_tokens
    no_accent_marks = args.no_accent_marks
    verbose = args.verbose
    entity_types = args.types 
    actual_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(actual_path, f'conll/{output_filename}.conll')
    referrals, annotations = samples_loader_from_minio(server,access_key,secret_key, region, return_filename = True, n_annotations=3000)
    convert_to_conll(referrals, annotations, output_path, multiconll=multiconll)