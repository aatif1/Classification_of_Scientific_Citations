import os
from xml.etree import ElementTree
import glob
import re
import csv

paragraph = []
current_paragraph = []
label = []
for file_name in sorted(glob.glob(os.path.join('corpus', '*.xml'))):
    #full_file_name = os.path.abspath(file_name)
    dom = ElementTree.parse(file_name)
    sentence = dom.findall('s')
    for c in sentence:
        current_paragraph.append(c.text)
        if len(c.getchildren()) > 0 and 'label' in c.getchildren()[0].attrib:
            label.append(c.getchildren()[0].attrib['label'])
            paragraph.append(current_paragraph)
            current_paragraph = []


ims = []
with open('IMS_labels.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row['label']
        ims.append(text)
#making csv
with open("corpus.csv", "w") as csv_file:
    with open("corpus3.csv", "w") as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        # writer.writerow(["paragraph", "label"])
        writer.writerow(["data", 'label'])
        i = 0
        max = 1560
        while i < max:
            #writer.writerow((paragraph[i], label[i]))
            writer.writerow((paragraph[i], ims[i]))
            i = i + 1
