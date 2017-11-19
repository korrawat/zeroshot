available_synset_ids = set()
with open('available_synsets.txt') as f:
    for line in f:
        available_synset_ids.add(line[:-1])

available_2hop_ids = []
with open('hop2.txt') as f:
    for line in f:
        wnid = line[:-1]
        if wnid in available_synset_ids:
            available_2hop_ids.append(wnid)

with open('available_hop2.txt', 'w') as f:
    for wnid in available_2hop_ids:
        f.write(wnid + '\n')