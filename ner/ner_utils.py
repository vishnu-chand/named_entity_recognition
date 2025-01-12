import re
import torch
import random
import os.path
import numpy as np
import unicodedata
import pandas as pd
from copy import deepcopy
from os.path import basename
import plotly.graph_objects as go
from types import SimpleNamespace

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda xFormatXpw13: f'{xFormatXpw13:8.4f}')
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------- general utils ---------------------------------------------------

def set_seed(seed=10487):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


set_seed()


def dot_dict(data, reverse=False):
    if not reverse:
        if hasattr(data, 'items'):
            namespace = SimpleNamespace()
            for key, value in data.items():
                setattr(namespace, key, dot_dict(value) if hasattr(value, 'items') else value)
            return namespace
        return data
    else:
        if isinstance(data, SimpleNamespace):
            result = dict()
            for key, value in vars(data).items():
                result[key] = dot_dict(value, reverse=True) if isinstance(value, SimpleNamespace) else value
            return result
        return data


# --------------------------------------------------- eval/plot/viz utils ---------------------------------------------------

def align_data(*lists):
    max_len = max(len(lst) for lst in lists)
    aligned_lists = [lst + [''] * (max_len - len(lst)) for lst in lists]
    column_widths = [max(len(str(row[i])) for row in aligned_lists if i < len(row)) for i in range(max_len)]

    for i, lst in enumerate(aligned_lists):
        aligned_lists[i] = [(elem + ' ' * column_widths[j])[:column_widths[j]] for j, elem in enumerate(lst)]
    return aligned_lists


def get_token_level_metrics(decoder_outputs, target_tensor):
    predictions = torch.argmax(decoder_outputs, dim=-1)
    predictions_flat = predictions.reshape(-1)
    target_flat = target_tensor.reshape(-1)

    tp = torch.sum((predictions_flat == target_flat))
    fp = torch.sum((predictions_flat != target_flat))  # TODO fix this
    fn = torch.sum((predictions_flat != target_flat))  # TODO fix this
    accuracy = tp / len(target_flat)

    precision, recall, f1 = [torch.tensor(0) for _ in range(3)]
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1_score": f1, 'accuracy': accuracy}


def class_wise_metrics(y_true, y_pred, classes):
    TP = {c: 0 for c in classes}
    FP = {c: 0 for c in classes}
    FN = {c: 0 for c in classes}

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            TP[true] += 1
        else:
            FP[pred] += 1
            FN[true] += 1

    recall = {}
    f1_score = {}
    precision = {}

    for cls in classes:
        tp = TP[cls]
        fp = FP[cls]
        fn = FN[cls]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        precision[cls] = p
        recall[cls] = r
        f1_score[cls] = f1

    accuracy = sum(TP[cls] for cls in classes) / len(y_true)

    precision_macro_avg = sum(precision.values()) / len(precision)
    recall_macro_avg = sum(recall.values()) / len(recall)
    f1_macro_avg = sum(f1_score.values()) / len(f1_score)

    detailed_metrics = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
    metrics = {'class_accuracy'     : accuracy,  #
               'precision_macro_avg': precision_macro_avg, 'recall_macro_avg': recall_macro_avg, 'f1_macro_avg': f1_macro_avg}
    return metrics, detailed_metrics


def plot_metrics(out_path, metrics):
    for name, metric in metrics.items():
        fig = go.Figure()

        df = pd.DataFrame(metric)
        for col in [d for d in df.columns if d != 'epoch']:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=col))

        fig.update_layout(title="Evaluation metrics over Epochs", xaxis_title="Epoch", yaxis_title="metrics", xaxis=dict(tickmode="linear"),  # Ensure all epochs are shown
                          yaxis=dict(tickformat=".2f"),  # Format y-axis with 2 decimal places
                          legend=dict(title="Legend"), template="plotly_white", )

        df.to_csv(f"{out_path}/{name}.csv")
        fig.write_html(f"{out_path}/{name}.html")


# --------------------------------------------------- data processing utils ---------------------------------------------------
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.n_words = len(self.word2index)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')] + [self.word2index['<EOS>']]

    def sentenceFromIndexes(self, decoded_ids, add_eos=True):
        decoded_words = []
        for idx in decoded_ids:
            decoded_word = self.index2word[idx.item()]
            if decoded_word == '<EOS>':
                if add_eos:
                    decoded_words.append(decoded_word)
                break
            else:
                decoded_words.append(decoded_word)
        return ' '.join(decoded_words)


def passage_to_word_boundary(passage, book_path):
    res = []
    start_ix = 0
    for sentence_ix, sentence in enumerate(passage.split('\n')):
        for word in sentence.split():
            res.append(dict(x=word, start_ix=start_ix, end_ix=start_ix + len(word), f_name=f"{basename(book_path)}_{sentence_ix}", y='other'))
            start_ix += len(word) + 1
    return res


def get_word_level_annotation(data, d):
    # TODO one word will have one sub type, change it to have multiple sub types
    for q in data:
        if q['start_ix'] <= d['start_ix'] and q['end_ix'] >= d['end_ix']:
            d['y'] = q['y']
            return d
    return d


def load_annotations(book_path):
    data = list()
    with open(book_path, 'r') as book:
        for line_no, line in enumerate(book):
            line = line.strip()
            if line:
                words = line.strip().split('\t')
                if words[0].startswith('T'):
                    assert len(words) == 3
                    if ';' not in words[1]:  # TODO fix it
                        y, start_ix, end_ix = words[1].split(' ')
                        x = ' '.join(words[2:])
                        data.append(dict(x=x, start_ix=int(start_ix), end_ix=int(end_ix), y=y))
    with open(f"{os.path.splitext(book_path)[0]}.txt", 'r') as book:
        passage = book.read()
    # print(' '.join(passage.split()))
    res = list()
    for d in passage_to_word_boundary(passage, book_path):
        annotation = get_word_level_annotation(data, d)
        res.append(annotation)
    data = res
    # df = pd.DataFrame(res)['x y sentence_ix'.split()]
    # print(df)
    # print(' '.join(df['x'].tolist()))
    # print("114 load_annotations main : ", );quit()
    return data


def english_cleaners(text, enable_phonemize, enable_expand_abbreviations):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    from cleaners import convert_to_ascii, lowercase, expand_abbreviations, collapse_whitespace
    text = convert_to_ascii(text)
    text = lowercase(text)
    if enable_expand_abbreviations:
        text = expand_abbreviations(text)
    # if enable_phonemize:
    #     text = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    text = collapse_whitespace(text)
    return text


def pre_prcs_data(cfg, datas):
    for f_name in datas:
        data = datas[f_name]
        if cfg.data_cfg.simple_cleaners:
            res = list()
            for d in data:
                d['x'] = english_cleaners(d['x'], cfg.data_cfg.enable_phonemize, cfg.data_cfg.enable_expand_abbreviations)
                d['y'] = english_cleaners(d['y'], cfg.data_cfg.enable_phonemize, cfg.data_cfg.enable_expand_abbreviations)
                if '-' in d['x']:
                    for w in d['x'].split('-'):
                        t = deepcopy(d)
                        t['x'] = w
                        res.append(t)
                else:
                    res.append(d)
            data = res

        if cfg.data_cfg.advance_cleaners:
            for d in data:
                ox = d['x']
                d['x'] = re.sub(r'[^a-z0-9\s]', '', d['x']).strip()  # remove special char
                d['x'] = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', d['x']).strip()  # remove rd, nd formats
                d['x'] = '<NUM>' if d['x'].isdigit() else d['x']  # remove number
                d['x'] = '<FIG>' if re.search(r'(fig|fig |figs|figs |figure|figures)\d+', d['x']) else d['x']  # remove fig

                # d['x'] = '<MIN>' if re.search(r'\d+(minute|min)', d['x']) else d['x']  # remove time
                # d['x'] = '<SEC>' if re.search(r'\d+(s)', d['x']) else d['x']  # remove time
                # d['x'] = '<DEG>' if re.search(r'\d+(deg)', d['x']) else d['x']  # remove degree
                # d['x'] = '<ML>' if re.search(r'\d+(ml)', d['x']) else d['x']  # remove ml

                if '' or '':
                    w = d['x'].replace('<NUM>', 'number').replace('<FIG>', 'figure').replace('<MIN>', 'minute').replace('<SEC>', 'second').replace('<DEG>', 'degree').replace('<ML>', 'ml')
                    if w and not w.isalpha():
                        print(f"""138 data_cfg_data sample_6 w: {d['x'], d['y']}""")
                if '' or '':
                    if ox != d['x']:
                        print(ox, d['x'])
        data = data + [dict(x='<PAD>', y='<PAD>', f_name=f_name)] * cfg.data_cfg.n_pad
        datas[f_name] = data
    return datas


def perform_eda(datas):
    import plotly.express as px
    if '' or '':
        df = pd.DataFrame([d for data in datas.values() for d in data])
        df = df[~df.y.str.contains('other')]  # drop other
        df = df[~df.y.str.contains('<PAD>')]  # drop <PAD>

        fig = px.histogram(df.y, x="y", color="y")
        fig.show()
    elif ' ' or '':
        df = pd.DataFrame([len(s) for s in datas.values()], columns=['length'])
        sentence_length = df['length'].value_counts().reset_index()
        fig = px.histogram(df, x="length", marginal='box')
        fig.show()
    elif '' or '':
        df = pd.DataFrame([d for data in datas.values() for d in data])
        print(df[df.x.apply(lambda x: len(x)) > 10])  # too long words
    elif ' ' or '':
        df = pd.DataFrame([d for data in datas.values() for d in data])
        df_word_counts = df['x'].value_counts().reset_index()
        print(df_word_counts)
    quit()


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def filterPair(p, max_length, use_eng_prefixes):
    if use_eng_prefixes:
        eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re ")
        return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length and p[1].startswith(eng_prefixes)
    else:
        return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


'''




T: text-bound annotation
R: relation
E: event
A: attribute
M: modification (alias for attribute, for backward compatibility)
N: normalization [new in v1.3]
#: note
*: equivalence


                # elif words[0].startswith('R'):
                #     y, arg1, arg2 = words[1].split(' ')
                #     assert len(words) == 2
                #     assert arg1.startswith('Arg1')
                #     assert arg2.startswith('Arg2')
                #     arg1 = arg1.split(':')[1]
                #     arg2 = arg2.split(':')[1]
                #     data.append(dict(type='relation', id=words[0], y=y, arg1=arg1, arg2=arg2))
                # elif words[0].startswith('E'):
                #     assert len(words) == 2
                #     y, x = words[1].split(':')
                #     data.append(dict(type='event', id=words[0], y=y, x=x))
                # elif words[0].startswith('A'):  # TODO
                #     pass
                # elif words[0].startswith('M'):  # TODO
                #     pass
                # elif words[0].startswith('#'):  # TODO
                #     pass
                # elif words[0].startswith('*'):  # TODO
                #     pass
                # else:
                #     print(words)






                elif words[0].startswith('A'):
                    y = words[1].split(' ')[0]
                    x = words[2]
                    data.append(dict(type='polarity', id=words[0], y=y, x=x))
                elif words[0] == '*':
                    y = 'OVERLAP'
                    x = words[1:]
                    data.append(dict(type='overlap', id=words[0], y=y, x=x))
                elif words[0].startswith('#'):
                    y = 'AnnotatorNotes'
                    target = words[1].split(' ')[1]
                    x = ' '.join(words[2:])
                    data.append(dict(type='note', id=words[0], y=y, target=target, x=x))
                else:
                    print(line)


'''
