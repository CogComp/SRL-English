import argparse
import json

parser = argparse.ArgumentParser(description="convert nom id output to nom srl input")
parser.add_argument("filename", type=str, help="the path to the nom id output")
parser.add_argument("outfile", type=str, help="the path to the nom srl input")


args = parser.parse_args()


def shift_indices_for_empty_strings(words, indices):
    shiftleft = 0
    new_indices = []
    new_words = []
    for idx, word in enumerate(words):
        if word=="" or word.isspace():
            shiftleft += 1
        else:
            if idx in indices:
                new_indices.append(idx-shiftleft)
            new_words.append(word)
    return new_words, new_indices

data = {}
with open(args.filename, 'r') as id_file, open(args.outfile, 'w') as outfile:
    for entry in id_file:
        data = json.loads(entry)
        indices = [idx for idx in range(len(data["nominals"])) if data["nominals"][idx]==1]
        new_words, new_indices = shift_indices_for_empty_strings(data["words"], indices)
        outfile.write(json.dumps({"sentence": " ".join(new_words), "indices": new_indices})+"\n")
