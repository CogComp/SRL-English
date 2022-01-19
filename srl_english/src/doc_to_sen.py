
import re
import nltk
from nltk.tokenize import sent_tokenize

import argparse

# nltk.download('punkt')

def doc_json(args):
    content = ""
    sen = []

    if args.mode=="text":
        context = args.text
    else:
        content = open(args.file, "r").read()
    
    sen = sent_tokenize(content)
    
    # if args.showout:
    #     for s in sen:
    #         print(s)
    #         print("_"*len(s), "\n")
    # print(sen)
    f = open(args.outFile, 'w')
    for s in sen:
        if(len(re.sub(r'[^a-zA-Z ]+', '', s))) > 2 and len(s.split(" ")) > 0:
            s = re.sub(r'[\"\“\”\"\'\']', "", s)
            f.write(s + "\n")
    f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert text to json')

    parser.add_argument('--mode', default='text', help='mode 1. file 2. text, default = text')
    parser.add_argument('--file', default='', help='filepath')
    parser.add_argument('--text', default='', help='content')
    parser.add_argument('--outFile', default='', help='output file name')
    parser.add_argument('--showout', default='', help='output file name')

    args = parser.parse_args()

    doc_json(args)

    

