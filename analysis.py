import codecs 

if __name__ == "__main__":
    with open('dataset/conll/entities.conll', 'r') as f:
        text = f.read()
        sentences = text.split('\n\n')[:-1]
        for sent in sentences:
            
            for line in sent.splitlines():
                
        