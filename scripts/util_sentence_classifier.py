import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os,sys,time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertTokenizer, BertModel, BertForSequenceClassification


def get_sentences_events(sentences,events):
    
    sentence_classification = []
    label_classification = []
    no_events_classification = []

    for i in range(len(sentences)):
        sum = 0
        for j in range(len(sentences[i])):
            sum += 1 if events[i][j] == 'EVENT' else 0
        sentence_classification.append(" ".join(sentences[i]))
        label_classification.append(1 if sum>0 else 0)
        no_events_classification.append(sum)

    df = pd.DataFrame({"sentence":sentence_classification,"label":label_classification,"no_of_events":no_events_classification})
    
    return df

def tokenize_sentences_make_dataloader(df,batch_size,test=False):

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    df_sentences = tokenizer(list(df.sentence),padding = True,truncation=True,max_length = 512,return_tensors='pt')
    df_labels = torch.tensor(df.label)

    df_data = TensorDataset(df_sentences.input_ids, df_sentences.attention_mask, df_labels)
    df_sampler = RandomSampler(df_data) if test==False else SequentialSampler(df_data)
    df_dataloader = DataLoader(df_data, sampler = df_sampler, batch_size = batch_size)

    return df_dataloader


def accuracy(preds,labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat),len(labels_flat)

def test(model,dev_dataloader,device):
    model.eval()
    model.to(device)
    correct = 0
    count = 0
    for batch in dev_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0].detach().cpu().numpy()
        labels = b_labels.to('cpu').numpy()
        tmp_eval_accuracy,temp_length = accuracy(logits, labels)

        correct += tmp_eval_accuracy
        count += temp_length

    print("Accuracy : ",correct/count)
    
    return (correct/count)


def sentence_event_classifier(model,sentence_in):

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    sentences = tokenizer(sentence_in,padding = True,truncation=True,max_length = 512,return_tensors='pt')
    output = model(sentences.input_ids,attention_mask=sentence.attention_mask)
    logit = output[0].detach().cpu().numpy()
    pred = np.argmax(logit,axis=1).flatten()[0]

    return pred