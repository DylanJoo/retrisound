import torch
import json
import re
import os
import string
import time

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = sent.split('</q>')[0]
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = sent.split('</p>')[0]
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, ' ', sent)
    return sent



