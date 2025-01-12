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
    pip install PyYAML==6.0.2 torch==1.13.1 torchvision==0.14.1 numpy==1.26.4 einops==0.8.0 pandas==2.2.3 plotly==5.24.1
'''

import time
import yaml
import argparse
import gradio as gr
from ner_utils import *
from pprint import pprint
from ner_trainer import get_model


def load_model(cfg, ckpt):
    cfg.out_dir = f"{cfg.out_dir}/{cfg.train_cfg.encoder}_{cfg.train_cfg.decoder}_{time.time_ns()}"
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(f'{cfg.out_dir}/config.yaml', 'w') as book:
        yaml.dump(dot_dict(cfg, reverse=True), book, default_flow_style=None, indent=4)
    pprint(cfg)
    ckpt = torch.load(ckpt, map_location='cpu')
    input_lang, output_lang = ckpt['input_lang'], ckpt['output_lang']
    encoder, decoder = get_model(cfg, input_lang, output_lang)
    print(encoder.load_state_dict(ckpt['encoder']))
    print(decoder.load_state_dict(ckpt['decoder']))
    encoder, decoder = encoder.to(device).eval(), decoder.to(device).eval()
    return encoder, decoder, input_lang, output_lang


@torch.no_grad()
def run_me_ner_demo(cfg, encoder, decoder, input_lang, output_lang, input_text):
    unknown_words = [word for word in input_text.split(' ') if word not in input_lang.word2index]
    if unknown_words:
        return f'unknown words detected, model cannot process: {unknown_words}'
    inp_ids = input_lang.indexesFromSentence(input_text)
    inp_padded = np.zeros(cfg.max_length, dtype=np.int32)
    inp_padded[:len(inp_ids)] = inp_ids
    input_tensor = torch.LongTensor(inp_padded).to(device)[None]

    encoder_outputs, encoder_hidden = encoder.forward(input_tensor)
    decoder_outputs, decoder_hidden, decoder_attn = decoder.forward(encoder_outputs, encoder_hidden)
    _, decoded_ids = decoder_outputs.topk(1)

    result = list()
    for batch_ix in range(input_tensor.shape[0]):
        y_hat = output_lang.sentenceFromIndexes(decoded_ids[batch_ix].squeeze(), add_eos=False)
        result.append(y_hat)
    return "\n".join(result)


def gradio_interface(input_text):
    result = run_me_ner_demo(cfg, encoder, decoder, input_lang, output_lang, input_text)
    return result


iface = gr.Interface(fn=gradio_interface,  #
                     inputs=gr.Textbox(label="Input Text", placeholder="Enter the text for NER", lines=2),  # User inputs text
                     outputs=gr.Textbox(label="NER Results"),  # Display the results
                     title="NER Demo",  # Title of the interface
                     description="Enter your text to run Named Entity Recognition.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for ner demo")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()
    with open(args.config, 'r') as book:
        cfg = dot_dict(yaml.safe_load(book))
    encoder, decoder, input_lang, output_lang = load_model(cfg, args.ckpt)
    iface.launch()
