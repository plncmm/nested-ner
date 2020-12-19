import codecs
import numpy as np
from tabulate import tabulate

def fix(pred, entities):
    new_pred = []
    for token_labels in pred:
        new_token_labels = []
        if 'O' in token_labels and len(token_labels)>1:
            ar = []
            for tag in token_labels:
                if tag!='O':
                    ar.append(tag)

            for entity in entities:
                if f'B-{entity}' in ar and f'I-{entity}' in ar:
                    ar.remove(f'I-{entity}')
            new_pred.append(ar)
        
        else:
            ar = []
            for tag in token_labels:
                ar.append(tag)

            for entity in entities:
                if f'B-{entity}' in ar and f'I-{entity}' in ar:
                    ar.remove(f'I-{entity}')
            new_pred.append(ar)
    return new_pred


def keep_bio_format(pred):
    cnt = 0
    new_pred = []
    new_pred.append(pred[0])
    for elem in pred[1:]:
        previous_token = pred[cnt]
        cnt+=1
        new_val = []
        for tag in elem:
            new_val.append(tag)

        for tag in new_val:
            if tag.startswith('I-'):
                if f'B-{tag[2:]}' not in previous_token and f'I-{tag[2:]}' not in previous_token:
                    new_val.remove(tag)
                    new_val.append(f'B-{tag[2:]}') 
                    break 
         
        new_pred.append(new_val)
        
        
    return new_pred








def entities_start_end(entities_dict, entities, debug=False):
    new_dict = {}
    for entity in entities:
        new_dict[entity]=[]
    
    for k,v in entities_dict.items():
        v = sorted(v, key=lambda item: item[1])
        idx = 0
        while(idx<len(v)):
            largo = 0
            if v[idx][0].startswith('B-'):
                if idx==len(v)-1:
                    new_dict[k].append((f'B-{k}', v[idx][1], v[idx][1]))
                    break
                else:
                    start_pos = v[idx][1]
                    actual_idx = idx
                    for tag in v[actual_idx+1:]:
                        if tag[0].startswith('B-'):
                            new_dict[k].append((f'B-{k}', start_pos, start_pos+largo))
                            idx+=1
                            break
                        elif tag[0].startswith('I-') and tag[1]!=start_pos+largo+1:
                            new_dict[k].append((f'B-{k}', start_pos, start_pos+largo))
                            idx+=1
                            break
                        
                        else:
                            idx+=1
                            largo+=1
                            if idx == len(v)-1:
                                new_dict[k].append((f'B-{k}', start_pos, start_pos+largo))
                                break
                            
            else:
                idx+=1
    return new_dict



    
def get_all_entities(labels, entities, debug=False):
    entities_dict = {}
    for entity in entities:
        entities_dict[entity]=[]
    
    for k, _ in entities_dict.items():
        for i, label in enumerate(labels):
            for tag in label: 
                if tag[2:]==k:
                    entities_dict[k].append([tag, i])
    
    original_entities = entities_dict
    entities_dict = entities_start_end(entities_dict, entities, debug)
    
    entities_mean = {}
    for k,v in entities_dict.items():
        entities_mean[k] = len(original_entities[k])/len(entities_dict[k]) if len(entities_dict[k])>0 else 0
    return entities_dict, entities_mean





def exact_entity_f1_score(y_true, y_pred, entities):
    entities_support = {}
    pred_support = {}
    prec_dict = {}
    recall_dict = {}
    f1_dict = {}
    for entity in entities:
        entities_support[entity] = 0
        pred_support[entity] = 0
        prec_dict[entity] = 0
        recall_dict[entity] = 0
        f1_dict[entity] = 0

    real_entities, entities_mean = get_all_entities(y_true, entities)
    pred_entities, pred_mean = get_all_entities(y_pred, entities, debug=True)

    


    for k, _ in recall_dict.items():
        cnt = 0

        for elem in real_entities[k]:
            for elem2 in pred_entities[k]:
                if elem[1]==elem2[1] and elem[2]==elem[2]:
                    cnt+=1
                    break 
        entities_support[k]+=len(real_entities[k])
        recall_dict[k]=cnt/len(real_entities[k]) if len(real_entities[k])!=0 else 1
     
    
    for k, _ in prec_dict.items():
        cnt = 0
        for elem in pred_entities[k]:
            for elem2 in real_entities[k]:
                if elem[1]==elem2[1] and elem[2]==elem[2]:
                    cnt+=1
                    break

        pred_support[k]+=len(pred_entities[k])
        prec_dict[k]=cnt/len(pred_entities[k]) if len(pred_entities[k])!=0 else 1
    
    
    for k,v in prec_dict.items():
        f1_dict[k]=(round(2*prec_dict[k]*recall_dict[k]/(prec_dict[k]+recall_dict[k]),2), entities_support[k], entities_mean[k]) if (prec_dict[k]+recall_dict[k])!=0 else (0,entities_support[k])

    weighted_f1 = 0
    macro_f1 = 0
    total_entities = 0
    for k,v in f1_dict.items():
        if v[1]!=0:
            total_entities+=1
        weighted_f1+=v[0]*v[1]
        macro_f1+=v[0]

    total_support = 0
    for k,v in entities_support.items():
        total_support+=v

    print('\n')
    print(f'Weighted entity-level f1_score: {round(weighted_f1/total_support,2)}')
    print(f'Macro entity-level f1_score: {round(macro_f1/total_entities,2)}')
    return f1_dict










def entity_f1_score(y_true, y_pred, entities):
    entities_support = {}
    pred_support = {}
    prec_dict = {}
    recall_dict = {}
    f1_dict = {}
    for entity in entities:
        entities_support[entity] = 0
        pred_support[entity] = 0
        prec_dict[entity] = 0
        recall_dict[entity] = 0
        f1_dict[entity] = 0

    real_entities, entities_mean = get_all_entities(y_true, entities)
    pred_entities, pred_mean = get_all_entities(y_pred, entities, debug=True)

    


    for k, _ in recall_dict.items():
        cnt = 0

        for elem in real_entities[k]:
            if elem in pred_entities[k]:
                cnt+=1
                
                
        entities_support[k]+=len(real_entities[k])
        recall_dict[k]=cnt/len(real_entities[k]) if len(real_entities[k])!=0 else 1
     
    
    for k, _ in prec_dict.items():
        cnt = 0
        for elem in pred_entities[k]:
            if elem in real_entities[k]:
                cnt+=1
        pred_support[k]+=len(pred_entities[k])
        prec_dict[k]=cnt/len(pred_entities[k]) if len(pred_entities[k])!=0 else 1
    
    
    for k,v in prec_dict.items():
        f1_dict[k]=(round(2*prec_dict[k]*recall_dict[k]/(prec_dict[k]+recall_dict[k]),2), entities_support[k], entities_mean[k]) if (prec_dict[k]+recall_dict[k])!=0 else (0,entities_support[k])

    weighted_f1 = 0
    macro_f1 = 0
    total_entities = 0
    for k,v in f1_dict.items():
        if v[1]!=0:
            total_entities+=1
        weighted_f1+=v[0]*v[1]
        macro_f1+=v[0]

    total_support = 0
    for k,v in entities_support.items():
        total_support+=v

    print('\n')
    print(f'Weighted entity-level f1_score: {round(weighted_f1/total_support,2)}')
    #print(f'Macro entity-level f1_score: {round(macro_f1/total_entities,2)}')
    return f1_dict, round(weighted_f1/total_support,4)



def f1_score_token(y_true, y_pred, entities):
    real_support = {'O': 0}
    pred_support = {'O': 0}
    prec_dict = {'O': 0}
    recall_dict = {'O': 0}
    f1_dict = {'O': 0}

    for entity in entities:
        real_support[f'B-{entity}'] = 0
        pred_support[f'B-{entity}'] = 0
        prec_dict[f'B-{entity}'] = 0
        recall_dict[f'B-{entity}'] = 0
        f1_dict[f'B-{entity}'] = 0
        real_support[f'I-{entity}'] = 0
        pred_support[f'I-{entity}'] = 0
        prec_dict[f'I-{entity}'] = 0
        recall_dict[f'I-{entity}'] = 0
        f1_dict[f'I-{entity}'] = 0
    
    for real, pred in zip(y_true, y_pred):
        for entity in pred:
            pred_support[entity]+=1
            if entity in real:
                prec_dict[entity]+=1
        
        for entity in real:
            real_support[entity]+=1
            if entity in pred:
                recall_dict[entity]+=1
    
    for k,v in prec_dict.items():
        prec = v/pred_support[k] if pred_support[k]!=0 else 1
        recall = recall_dict[k]/real_support[k] if real_support[k]!=0 else 1
        f1_dict[k]= (round(2*(prec)*(recall)/((prec)+(recall)), 2), real_support[k]) if ((prec)+(recall))!=0 else (0,real_support[k]) 
    
    total_support = 0
    for k,v in real_support.items():
        total_support+=v

    weighted_f1 = 0
    macro_f1 = 0
    total_entities = 0
    for k,v in f1_dict.items():
        if v[1]!=0:
            total_entities+=1
        weighted_f1+=v[0]*v[1]
        macro_f1+=v[0]

    print('\n')
    print(f'Weighted token-level f1_score: {round(weighted_f1/total_support,2)}')
    print(f'Macro token-level f1_score: {round(macro_f1/total_entities,2)}')
    
    return f1_dict










def tabulate_dict(metric_dict):
    f1_dict = {k: v for k, v in sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)}
    f1_tokens = []
    for k,v in f1_dict.items():
        if v[0]!=0:
            f1_tokens.append([k, v[0], v[1]]) 
    return tabulate(f1_tokens, headers=['Entity type', 'F1-Score', 'Support'])

def tabulate_f1_score(metric_dict, include_mean=False):
    f1_dict = {k: v for k, v in sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)}
    f1_tokens = []
    for k,v in f1_dict.items():
        if v[0]!=0:
            f1_tokens.append([k, v[0], v[1]]) if not include_mean else f1_tokens.append([k, v[0], v[1], v[2]])
    return tabulate(f1_tokens, headers=['Entity type', 'F1-Score', 'Support', 'Tokens mean'])

def token_accuracy_strict(y_true, y_pred, entities): 
    entity_support = {'O': 0}
    acc_dict = {'O': 0}
    for entity in entities:
        entity_support[f'B-{entity}'] = 0
        acc_dict[f'B-{entity}'] = 0
        entity_support[f'I-{entity}'] = 0
        acc_dict[f'I-{entity}'] = 0

    for real, pred in zip(y_true, y_pred):
        for entity in real:
            entity_support[entity]+=1
            if entity in pred:
                acc_dict[entity]+=1
    
    for k, v in acc_dict.items():
        acc_dict[k]=(round(v/entity_support[k],2), entity_support[k]) if entity_support[k]!=0 else (0, 0)
    
  
    return acc_dict




def accuracy_score(y_true, y_pred, ignore=None, multiple=False):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if multiple:
        total_samples = 0
        correct_samples = 0
        for y_t, y_p in zip(y_true, y_pred):
            if len(y_t)>1:
                total_samples+=1
                if y_t == y_p:
                    correct_samples+=1
        return correct_samples/total_samples

    if ignore is not None:
        total_samples = 0
        correct_samples = 0
        for y_t, y_p in zip(y_true, y_pred):
            if ignore in y_t or ignore in y_p:
                continue 
            else:
                total_samples+=1
                if y_t == y_p:
                    correct_samples+=1
        return correct_samples/total_samples

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)
    score = nb_correct / nb_true
    return score



def get_labels_from_conll(conll):
    labels = []
    for line in conll.splitlines():
        line_info = line.split()
        if len(line_info)==1 or line_info[1]=='PAD': 
            labels.append(['O'])
        else:
            labels.append(line_info[1:])
    return labels

if __name__ == "__main__":
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    nn_output = codecs.open('testing.conll', 'r', 'utf-8').read()
    real_output = codecs.open('real.conll', 'r', 'utf-8').read()

    pred = get_labels_from_conll(nn_output)
    true = get_labels_from_conll(real_output)
    pred_fixed = keep_bio_format(fix(pred, entities))
    

    # Token level evaluation
    print(f'El accuracy a nivel de token es: {round(accuracy_score(true, pred, ignore = None),2)}')
    print(f'El accuracy a nivel de token es: {round(accuracy_score(true, pred_fixed, ignore = None),2)}')


    print(f"El accuracy a nivel de token sin considerar el token O es: {round(accuracy_score(true, pred, ignore = 'O'),2)}")
    print(f"El accuracy a nivel de token sin considerar el token O es: {round(accuracy_score(true, pred_fixed, ignore = 'O'),2)}")
    
    print(f"El accuracy en tokens múltiples es: {round(accuracy_score(true, pred, ignore = 'O', multiple=True),2)}")
    print(f"El accuracy en tokens múltiples es: {round(accuracy_score(true, pred_fixed, ignore = 'O', multiple=True),2)}")
    
    print(tabulate_dict(token_accuracy_strict(true, pred, entities)))
    print(tabulate_dict(token_accuracy_strict(true, pred_fixed, entities)))
    
    print(tabulate_dict(f1_score_token(true, pred,  entities)))
    print(tabulate_dict(f1_score_token(true, pred_fixed,  entities)))

    # Entity level evaluation
    print(tabulate_f1_score(entity_f1_score(true, fix(pred, entities), entities)[0], include_mean=True))
   # print(tabulate_f1_score(entity_f1_score(true, pred_fixed, entities)[0], include_mean=True))


    #print(tabulate_f1_score(exact_entity_f1_score(true, pred, entities), include_mean=True))
    #print(tabulate_f1_score(exact_entity_f1_score(true, pred_fixed, entities), include_mean=True))


    


    
    

  