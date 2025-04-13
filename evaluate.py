import evaluate
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd
import os
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')
import argparse
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        # Chuyển đổi tất cả các ký tự thành chữ thường
        sentence = sentence.lower()
        # Xóa các ký tự đặc biệt và chuyển các số và chữ số thành token <n>
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Loại bỏ ký tự đặc biệt
        sentence = re.sub(r'\b\d+\b', '<n>', sentence)  # Chuyển các số thành token <n>
        # Loại bỏ từ, cụm từ, câu bị lặp lại liên tục nhiều lần
        sentence_words = sentence.split()
        cleaned_words = []
        previous_word = None
        for word in sentence_words:
            if word != previous_word:
                cleaned_words.append(word)
            previous_word = word
        cleaned_sentence = ' '.join(cleaned_words)
        # Thêm câu đã xử lý vào list kết quả
        processed_sentences.append(cleaned_sentence)
    return processed_sentences
# Hàm để xử lý DataFrame
def preprocess_df(df):
    df['processed_Caption'] = preprocess_sentences(df['Caption'])
    return df
def BERTscore (bertscore,valid_captions,cands) :
    large_bert2 = []
    length = len(cands["Caption"])
    for i in tqdm(range(length)):
        large_bert = bertscore.compute(predictions=[cands["Caption"][i]],
                                lang="en",
                                model_type="microsoft/deberta-xlarge-mnli",
                                references=[valid_captions["Caption"][i]])
        large_bert2.append(large_bert["f1"][0])
    return large_bert2
def evaluation(root,score):
    """
    root: path data evaluation 
    score
    """

    valid_captions = pd.read_csv(root + "valid_captions.csv")
    valid_captions = preprocess_df(valid_captions)

    # Large
    large_greedy = pd.read_csv(root + "large_greedy.csv")
    large_beam3 = pd.read_csv(root + "large_beam3.csv")
    large_beam4 = pd.read_csv(root + "large_beam4.csv")
    large_beam5 = pd.read_csv(root + "large_beam5.csv")
    large_beam10 = pd.read_csv(root + "large_beam10.csv")

    large_greedy = preprocess_df(large_greedy)
    large_beam3 = preprocess_df(large_beam3)
    large_beam4 = preprocess_df(large_beam4)
    large_beam5 = preprocess_df(large_beam5)
    large_beam10 = preprocess_df(large_beam10)

    # Base 
    base_greedy = pd.read_csv(root + "base_greedy.csv")
    base_beam3 = pd.read_csv(root + "base_beam3.csv")
    base_beam4 = pd.read_csv(root + "base_beam4.csv")
    base_beam5 = pd.read_csv(root + "base_beam5.csv")
    base_beam10 = pd.read_csv(root + "base_beam10.csv")


    base_greedy = preprocess_df(base_greedy)
    base_beam3 = preprocess_df(base_beam3)
    base_beam4 = preprocess_df(base_beam4)
    base_beam5 = preprocess_df(base_beam5)
    base_beam10 = preprocess_df(base_beam10)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", module_type="metric",checkpoint  ="BLEURT-20")
    if(score=='rouge'):
      # @title Base
        base_greedy_rouge = rouge.compute(predictions=base_greedy["Caption"], references=valid_captions["Caption"])["rouge1"]
        base_beam3_rouge = rouge.compute(predictions=base_beam3["Caption"], references=valid_captions["Caption"])["rouge1"]
        base_beam4_rouge = rouge.compute(predictions=base_beam4["Caption"], references=valid_captions["Caption"])["rouge1"]
        base_beam5_rouge = rouge.compute(predictions=base_beam5["Caption"], references=valid_captions["Caption"])["rouge1"]
        base_beam10_rouge = rouge.compute(predictions=base_beam10["Caption"], references=valid_captions["Caption"])["rouge1"]
        print( "Base Greedy:" , round(base_greedy_rouge,6))
        print( "Base Beam 3:" , round(base_beam3_rouge,6))
        print( "Base Beam 4:" , round(base_beam4_rouge,6))
        print( "Base Beam 5:" , round(base_beam5_rouge,6))
        print( "Base Beam 10:", round(base_beam10_rouge,6))

        # @title Large
        large_greedy_rouge = rouge.compute(predictions=large_greedy["Caption"], references=valid_captions["Caption"])["rouge1"]
        large_beam3_rouge = rouge.compute(predictions=large_beam3["Caption"], references=valid_captions["Caption"])["rouge1"]
        large_beam4_rouge = rouge.compute(predictions=large_beam4["Caption"], references=valid_captions["Caption"])["rouge1"]
        large_beam5_rouge = rouge.compute(predictions=large_beam5["Caption"], references=valid_captions["Caption"])["rouge1"]
        large_beam10_rouge = rouge.compute(predictions=large_beam10["Caption"], references=valid_captions["Caption"])["rouge1"]
        print( "Large Greedy:" , round(large_greedy_rouge,6))
        print( "Large Beam 3:" , round(large_beam3_rouge,6))
        print( "Large Beam 4:" , round(large_beam4_rouge,6))
        print( "Large Beam 5:" , round(large_beam5_rouge,6))
        print( "Large Beam 10:", round(large_beam10_rouge,6))
    elif(score=='bleu'):
       # @title Base
        base_greedy_bleu = bleu.compute(predictions=base_greedy["Caption"], references=valid_captions["Caption"])["precisions"][0]
        base_beam3_bleu  = bleu.compute(predictions=base_beam3["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        base_beam4_bleu  = bleu.compute(predictions=base_beam4["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        base_beam5_bleu  = bleu.compute(predictions=base_beam5["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        base_beam10_bleu = bleu.compute(predictions=base_beam10["Caption"], references=valid_captions["Caption"])["precisions"][0]
        print( "Base Greedy:" , round(base_greedy_bleu,6))
        print( "Base Beam 3:" , round(base_beam3_bleu,6))
        print( "Base Beam 4:" , round(base_beam4_bleu,6))
        print( "Base Beam 5:" , round(base_beam5_bleu,6))
        print( "Base Beam 10:", round(base_beam10_bleu,6))
        # @title Large
        large_greedy_bleu = bleu.compute(predictions=large_greedy["Caption"], references=valid_captions["Caption"])["precisions"][0]
        large_beam3_bleu  = bleu.compute(predictions=large_beam3["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        large_beam4_bleu  = bleu.compute(predictions=large_beam4["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        large_beam5_bleu  = bleu.compute(predictions=large_beam5["Caption"] , references=valid_captions["Caption"])["precisions"][0]
        large_beam10_bleu = bleu.compute(predictions=large_beam10["Caption"], references=valid_captions["Caption"])["precisions"][0]
        print( "Large Greedy:" , round(large_greedy_bleu,6))
        print( "Large Beam 3:" , round(large_beam3_bleu,6))
        print( "Large Beam 4:" , round(large_beam4_bleu,6))
        print( "Large Beam 5:" , round(large_beam5_bleu,6))
        print( "Large Beam 10:", round(large_beam10_bleu,6))
    elif(score=='meteor'):
        # @title Base
        base_greedy_meteor = meteor.compute(predictions=base_greedy["Caption"], references=valid_captions["Caption"])["meteor"]
        base_beam3_meteor = meteor.compute(predictions=base_beam3["Caption"], references=valid_captions["Caption"])["meteor"]
        base_beam4_meteor = meteor.compute(predictions=base_beam4["Caption"], references=valid_captions["Caption"])["meteor"]
        base_beam5_meteor = meteor.compute(predictions=base_beam5["Caption"], references=valid_captions["Caption"])["meteor"]
        base_beam10_meteor = meteor.compute(predictions=base_beam10["Caption"], references=valid_captions["Caption"])["meteor"]
        print( "Base Greedy:" , round(base_greedy_meteor,6))
        print( "Base Beam 3:" , round(base_beam3_meteor,6))
        print( "Base Beam 4:" , round(base_beam4_meteor,6))
        print( "Base Beam 5:" , round(base_beam5_meteor,6))
        print( "Base Beam 10:", round(base_beam10_meteor,6))
        # @title Large
        large_greedy_meteor = meteor.compute(predictions=large_greedy["Caption"], references=valid_captions["Caption"])["meteor"]
        large_beam3_meteor = meteor.compute(predictions=large_beam3["Caption"], references=valid_captions["Caption"])["meteor"]
        large_beam4_meteor = meteor.compute(predictions=large_beam4["Caption"], references=valid_captions["Caption"])["meteor"]
        large_beam5_meteor = meteor.compute(predictions=large_beam5["Caption"], references=valid_captions["Caption"])["meteor"]
        large_beam10_meteor = meteor.compute(predictions=large_beam10["Caption"], references=valid_captions["Caption"])["meteor"]
        print( "Large Greedy:" , round(large_greedy_meteor,6))
        print( "Large Beam 3:" , round(large_beam3_meteor,6))
        print( "Large Beam 4:" , round(large_beam4_meteor,6))
        print( "Large Beam 5:" , round(large_beam5_meteor,6))
        print( "Large Beam 10:", round(large_beam10_meteor,6))
    elif (score=='bertscore'):
        # @title Large
        base_greedy_bert = BERTscore(bertscore,valid_captions,base_greedy)
        base_beam3_bert  = BERTscore(bertscore,valid_captions,base_beam3)
        base_beam4_bert  = BERTscore(bertscore,valid_captions,base_beam4)
        base_beam5_bert  = BERTscore(bertscore,valid_captions,base_beam5)
        base_beam10_bert = BERTscore(bertscore,valid_captions,base_beam10)
        print( "Base Greedy:" , round(np.average(base_greedy_bert),6))
        print( "Base Beam 3:" , round(np.average(base_beam3_bert),6))
        print( "Base Beam 4:" , round(np.average(base_beam4_bert),6))
        print( "Base Beam 5:" , round(np.average(base_beam5_bert),6))
        print( "Base Beam 10:", round(np.average(base_beam10_bert),6))

        # @title Large
        large_greedy_bert = BERTscore(bertscore,valid_captions,large_greedy)
        large_beam3_bert  = BERTscore(bertscore,valid_captions,large_beam3)
        large_beam4_bert  = BERTscore(bertscore,valid_captions,large_beam4)
        large_beam5_bert  = BERTscore(bertscore,valid_captions,large_beam5)
        large_beam10_bert = BERTscore(bertscore,valid_captions,large_beam10)
        print( "Large Greedy:" , round(np.average(large_greedy_bert),6))
        print( "Large Beam 3:" , round(np.average(large_beam3_bert),6))
        print( "Large Beam 4:" , round(np.average(large_beam4_bert),6))
        print( "Large Beam 5:" , round(np.average(large_beam5_bert),6))
        print( "Large Beam 10:", round(np.average(large_beam10_bert),6))



def main():
     # Initializes a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands.
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command.
    parser_train = subparsers.add_parser('eval')

    # Adds an argument for the directory containing public data.
    parser_train.add_argument('--root', type=str,default='./')
    
    parser_train.add_argument('--score', type=str,default='rouge')
