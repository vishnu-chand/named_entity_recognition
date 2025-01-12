'''
named entity recognition task

reff/citation:
https://brat.nlplab.org/standoff.html
https://github.com/jaywalnut310/vits.git
https://aclanthology.org/2021.bionlp-1.16.pdf
https://www.geeksforgeeks.org/named-entity-recognition/
https://github.com/BeiqiZh/Named-Entity-Recognition/tree/main/code
https://www.kaggle.com/code/ritvik1909/named-entity-recognition-rnn
https://github.com/maxslimb/Named-Entity-Recognition/blob/main/main.py
https://zhoubeiqi.medium.com/named-entity-recognition-ner-using-keras-lstm-spacy-da3ea63d24c5
https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py


    conda deactivate; conda env remove -n llm; conda create -n llm python=3.10.14 -y; conda activate llm
    pip install PyYAML==6.0.2 torch==1.13.1 torchvision==0.14.1 numpy==1.26.4 einops==0.8.0 pandas==2.2.3 plotly==5.24.1 gradio==5.12.0
'''
import os
import time
import yaml
import argparse
from glob import glob
import torch.nn as nn
from pprint import pprint
from einops import rearrange
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from ner_utils import *


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),  #
                         nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True))


class CnnEncoder(nn.Module):
    def __init__(self, input_size, enc_layers):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.embedding = nn.Embedding(input_size, enc_layers[0][0])

        blocks = list()
        for in_ch, out_ch in enc_layers:
            blocks.append(conv_block(in_ch, out_ch))
        self.enc1, self.enc2, self.enc3, self.enc4 = blocks

    def forward(self, x):
        x = self.embedding(x)
        x = rearrange(x, 'b w e -> b e w')
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        return enc4, [enc1, enc2, enc3, enc4]


class CnnDecoder(nn.Module):
    def __init__(self, output_size, bottleneck, up_layers, dec_layers):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = conv_block(*bottleneck)

        blocks = list()
        for in_ch, out_ch in up_layers:
            blocks.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2))
        self.upconv4, self.upconv3, self.upconv2, self.upconv1 = blocks

        blocks = list()
        for in_ch, out_ch in dec_layers:
            blocks.append(conv_block(in_ch, out_ch))
        self.dec4, self.dec3, self.dec2, self.dec1 = blocks
        self.final = nn.Conv1d(dec_layers[-1][-1], output_size, kernel_size=1)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        enc1, enc2, enc3, enc4 = encoder_hidden
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.final(dec1)
        out = rearrange(out, 'b e w -> b w e')
        out = F.log_softmax(out, dim=-1)
        return out, None, None


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        '''
        r = sig(nn_r(x) + nn_r(h))
        u = sig(nn_z(x) + nn_z(h))
        n = tan(nn_n(x) + r * nn_n(h))
        h_next = (1 - u) * n + u * h
        '''
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class LinearDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        y_hat = encoder_outputs
        y_hat = self.fc(y_hat)
        y_hat = F.log_softmax(y_hat, dim=-1)
        return y_hat, None, None


class DecoderRNN(nn.Module):
    def __init__(self, SOS_token, max_length, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.SOS_token = SOS_token
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, SOS_token, max_length, output_size, hidden_size, dropout_p):
        super(AttnDecoderRNN, self).__init__()
        self.SOS_token = SOS_token
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        bs = encoder_outputs.size(0)
        decoder_input = torch.empty(bs, 1, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


class NerDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang, max_length):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        inp, tgt = self.pairs[index]
        inp_ids = self.input_lang.indexesFromSentence(inp)
        tgt_ids = self.output_lang.indexesFromSentence(tgt)

        inp_padded = np.zeros(self.max_length, dtype=np.int32)
        tgt_padded = np.zeros(self.max_length, dtype=np.int32)
        inp_padded[:len(inp_ids)] = inp_ids
        tgt_padded[:len(tgt_ids)] = tgt_ids
        input_tensor = torch.LongTensor(inp_padded).to(device)
        target_tensor = torch.LongTensor(tgt_padded).to(device)
        return input_tensor, target_tensor


def prepareData(cfg):
    if cfg.data_cfg.data_set == 'eng_fra':
        lines = open(f'{cfg.data_dir}/eng-fra.txt').read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        pairs = [list(p[::-1]) for p in pairs]
        print("Read %s sentence pairs" % len(pairs))
        pairs = [pair for pair in pairs if filterPair(pair, 10, True)]
        print("Trimmed to %s sentence pairs" % len(pairs))
    elif cfg.data_cfg.data_set == 'mac_2020':
        datas = list()
        book_paths = glob(f'/tmp/data/*.ann')
        if not book_paths:
            os.system('''
            rm -rf /tmp/data; mkdir /tmp/data;
            wget -O /tmp/data/download.zip "https://figshare.com/ndownloader/articles/9764942/versions/2";
            unzip -o /tmp/data/download.zip -d /tmp/data;
            unzip -o /tmp/data/MACCROBAT2020.zip -d /tmp/data;
            ''')
            book_paths = glob(f'/tmp/data/*.ann')
            assert book_paths, f'failed to download data'
        for book_path in sorted(book_paths):
            for data in load_annotations(book_path):
                data = {k: data[k] for k in 'x y f_name'.split()}
                datas.append(data)
        sentence_datas = defaultdict(list)
        for data in datas:
            sentence_datas[data['f_name']].append(data)
        datas = pre_prcs_data(cfg, sentence_datas)

        pairs = list()
        for f_name in datas:
            data = datas[f_name]
            x = ' '.join([d['x'] for d in data])
            y = ' '.join([d['y'] for d in data])
            pairs.append([x, y])
        print("Read %s sentence pairs" % len(pairs))
        pairs = [pair for pair in pairs if filterPair(pair, cfg.max_length, False)]
        print("Trimmed to %s sentence pairs" % len(pairs))
    else:
        raise NotImplementedError

    input_lang, output_lang = Lang('input_lang'), Lang('output_lang')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def get_dataloader(cfg, input_lang, output_lang, pairs):
    n_datas = len(pairs)
    bs, max_length = cfg.train_cfg.bs, cfg.max_length
    train, test, val = pairs[:int(0.6 * n_datas)], pairs[int(0.6 * n_datas):int(0.9 * n_datas)], pairs[int(0.9 * n_datas):]
    print(n_datas, len(train), len(test), len(val))

    val_dataloader = NerDataset(val, input_lang, output_lang, max_length)
    val_dataloader = DataLoader(val_dataloader, batch_size=bs, shuffle=True, drop_last=False, num_workers=os.cpu_count())
    test_dataloader = NerDataset(test, input_lang, output_lang, max_length)
    test_dataloader = DataLoader(test_dataloader, batch_size=bs, shuffle=False, drop_last=False, num_workers=os.cpu_count())
    train_dataloader = NerDataset(train, input_lang, output_lang, max_length)
    train_dataloader = DataLoader(train_dataloader, sampler=RandomSampler(train_dataloader), batch_size=bs, drop_last=True, num_workers=os.cpu_count())

    # train_dataloader = NerDataset(pairs, input_lang, output_lang, max_length)
    # train_dataloader = DataLoader(train_dataloader, sampler=RandomSampler(train_dataloader), batch_size=bs)
    return train_dataloader, test_dataloader, val_dataloader


def get_model(cfg, input_lang, output_lang):
    # encoders
    if cfg.train_cfg.encoder == 'EncoderRNN':
        encoder = EncoderRNN(input_lang.n_words, **vars(cfg.model_cfg.EncoderRNN)).to(device)
    elif cfg.train_cfg.encoder == 'CnnEncoder':
        encoder = CnnEncoder(input_lang.n_words, **vars(cfg.model_cfg.CnnEncoder)).to(device)
    elif cfg.train_cfg.encoder == 'TinyCnnEncoder':
        encoder = CnnEncoder(input_lang.n_words, **vars(cfg.model_cfg.TinyCnnEncoder)).to(device)
    else:
        raise NotImplementedError

    # decoders
    if cfg.train_cfg.decoder == 'AttnDecoderRNN':
        decoder = AttnDecoderRNN(output_lang.word2index['<SOS>'], cfg.max_length, output_lang.n_words, **vars(cfg.model_cfg.AttnDecoderRNN)).to(device)
    elif cfg.train_cfg.decoder == 'DecoderRNN':
        decoder = DecoderRNN(output_lang.word2index['<SOS>'], cfg.max_length, output_lang.n_words, **vars(cfg.model_cfg.DecoderRNN)).to(device)
    elif cfg.train_cfg.decoder == 'LinearDecoder':
        decoder = LinearDecoder(output_lang.n_words, **vars(cfg.model_cfg.LinearDecoder)).to(device)
    elif cfg.train_cfg.decoder == 'CnnDecoder':
        decoder = CnnDecoder(output_lang.n_words, **vars(cfg.model_cfg.CnnDecoder)).to(device)
    elif cfg.train_cfg.decoder == 'TinyCnnDecoder':
        decoder = CnnDecoder(output_lang.n_words, **vars(cfg.model_cfg.TinyCnnDecoder)).to(device)
    else:
        raise NotImplementedError
    return encoder, decoder


def train_it(cfg, train_dataloader, val_dataloader, input_lang, output_lang, encoder, decoder):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.train_cfg.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.train_cfg.lr)

    total_words = sum(output_lang.word2count.values())
    class_weight = None
    if cfg.train_cfg.enable_class_weight:
        class_weight = torch.tensor([total_words / output_lang.word2count.get(output_lang.index2word[ix], total_words) for ix in range(output_lang.n_words)])
    criterion = nn.NLLLoss(weight=class_weight)

    metrics = defaultdict(list)
    for epoch in range(1, cfg.train_cfg.n_epochs + 1):
        detailed_metrics = list()
        metric = dict(epoch=epoch, train_loss=0, eval_loss=0, precision=0, recall=0, f1_score=0, accuracy=0, precision_macro_avg=0, recall_macro_avg=0, f1_macro_avg=0, class_accuracy=0)
        for data in train_dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder, decoder = encoder.train(), decoder.train()

            input_tensor, target_tensor = data
            encoder_outputs, encoder_hidden = encoder.forward(input_tensor)
            decoder_outputs, _, _ = decoder.forward(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target_tensor.reshape(-1))
            # print(f"""629 train_epoch sample_9 loss: {loss}""")
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            metric['train_loss'] += loss.item() / len(train_dataloader)

        with torch.no_grad():
            for ix, data in enumerate(val_dataloader):
                input_tensor, target_tensor = data
                encoder, decoder = encoder.eval(), decoder.eval()
                encoder_outputs, encoder_hidden = encoder.forward(input_tensor)
                decoder_outputs, _, _ = decoder.forward(encoder_outputs, encoder_hidden, target_tensor)
                loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target_tensor.reshape(-1))
                metric['eval_loss'] += loss.item() / len(val_dataloader)
                for key, value in get_token_level_metrics(decoder_outputs, target_tensor).items():
                    metric[key] += value.item() / len(val_dataloader)

                _, decoded_ids = decoder_outputs.topk(1)
                bs = input_tensor.shape[0]
                for batch_ix in range(bs):
                    x = input_lang.sentenceFromIndexes(input_tensor[batch_ix].squeeze()).split(' ')
                    y = output_lang.sentenceFromIndexes(target_tensor[batch_ix].squeeze()).split(' ')
                    y_hat = output_lang.sentenceFromIndexes(decoded_ids[batch_ix].squeeze()).split(' ')
                    class_metric, detailed_metric = class_wise_metrics(y, y_hat, output_lang.word2index.keys())
                    for key, value in class_metric.items():
                        metric[key] += value / (bs * len(val_dataloader))
                    detailed_metrics.append(detailed_metric)
                    if epoch % (5 * cfg.train_cfg.ckpt_freq) == 0:
                        if batch_ix < 1:
                            ax, ay, ay_hat = align_data(x, y, y_hat)
                            print('>', ax)
                            print('=', ay)
                            print('<', ay_hat)
                            print('')

        metrics['general'].append(metric)
        for col in 'f1_score recall precision'.split():
            df = pd.DataFrame([d[col] for d in detailed_metrics]).mean().to_dict()
            df.update(epoch=epoch)
            metrics[col].append(df)
        print({k: f"{v:8.3f}" for k, v in metric.items()})
        if epoch % cfg.train_cfg.ckpt_freq == 0:
            torch.save(dict(encoder=encoder.state_dict(), decoder=decoder.state_dict(), input_lang=input_lang, output_lang=output_lang), f"{cfg.out_dir}/ckpt_{epoch:012d}.pth")
            plot_metrics(cfg.out_dir, metrics)
        if len(metrics['general']) > cfg.train_cfg.stop_patience:
            i_best = np.argmin([d[cfg.train_cfg.stop_metric] for d in metrics['general'][-cfg.train_cfg.stop_patience:]])
            if i_best == 0:
                print(f'early stopping at epoch {epoch}, no improvement for last {cfg.train_cfg.stop_patience} epochs')
                break
    torch.save(dict(encoder=encoder.state_dict(), decoder=decoder.state_dict(), input_lang=input_lang, output_lang=output_lang), f"{cfg.out_dir}/ckpt_{epoch:012d}.pth")
    plot_metrics(cfg.out_dir, metrics)


def showAttention(out_path, input_words, output_words, attentions):
    input_words = input_words + ['<EOS>']
    attentions = attentions[0, :len(output_words), :]

    fig = go.Figure(data=go.Heatmap(z=attentions.cpu().numpy(), x=input_words, y=output_words, colorscale="Blues", colorbar=dict(title="Attention Score"), ))
    fig.update_layout(title="Attention Heatmap", xaxis=dict(title="Input Words", tickangle=-45), yaxis=dict(title="Output Words", automargin=True), template="plotly_white", )
    fig.write_image(out_path, engine="kaleido")


def run_me_ner_train(cfg):
    cfg.out_dir = f"{cfg.out_dir}/{cfg.train_cfg.encoder}_{cfg.train_cfg.decoder}_{time.time_ns()}"
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(f'{cfg.out_dir}/config.yaml', 'w') as book:
        yaml.dump(dot_dict(cfg, reverse=True), book, default_flow_style=None, indent=4)
    pprint(cfg)
    input_lang, output_lang, pairs = prepareData(cfg)
    train_dataloader, test_dataloader, val_dataloader = get_dataloader(cfg, input_lang, output_lang, pairs)
    encoder, decoder = get_model(cfg, input_lang, output_lang)
    train_it(cfg, train_dataloader, val_dataloader, input_lang, output_lang, encoder, decoder)

    with torch.no_grad():
        for ix, data in enumerate(test_dataloader):
            input_tensor, target_tensor = data
            encoder, decoder = encoder.eval(), decoder.eval()

            encoder_outputs, encoder_hidden = encoder.forward(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder.forward(encoder_outputs, encoder_hidden)

            _, decoded_ids = decoder_outputs.topk(1)

            for batch_ix in range(input_tensor.shape[0]):
                x = input_lang.sentenceFromIndexes(input_tensor[batch_ix].squeeze()).split(' ')
                y = output_lang.sentenceFromIndexes(target_tensor[batch_ix].squeeze()).split(' ')
                y_hat = output_lang.sentenceFromIndexes(decoded_ids[batch_ix].squeeze()).split(' ')
                ax, ay, ay_hat = align_data(x, y, y_hat)
                print('>', ax)
                print('=', ay)
                print('<', ay_hat)
                print('')

                if decoder_attn is not None:
                    showAttention(f"{cfg.out_dir}/attention_{ix:012d}.jpg", x, y_hat, decoder_attn)
    print(f'out_dir: \n\n\n\t{cfg.out_dir}\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args for ner training")
    parser.add_argument('--config', type=str, default='ner_cfg.yaml', help="Path to the configuration YAML file")
    args = parser.parse_args()
    with open(args.config, 'r') as book:
        cfg = yaml.safe_load(book)
    run_me_ner_train(dot_dict(cfg))
