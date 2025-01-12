import os
import sys
import yaml
from os.path import dirname

python = sys.executable
code_dir = dirname(__file__)


def run_me_auto_train():
    out_dir = '/tmp/auto_train'
    cfg = yaml.safe_load(open('ner_cfg.yaml', 'r'))
    for hidden_size in [64, 128, 256]:
        EncoderRNN = dict(dropout_p=0.1, hidden_size=hidden_size)
        AttnDecoderRNN = dict(dropout_p=0.1, hidden_size=hidden_size)
        cfg['model_cfg']['EncoderRNN'] = EncoderRNN
        cfg['model_cfg']['AttnDecoderRNN'] = AttnDecoderRNN
        cfg['out_dir'] = f"{out_dir}/{hidden_size}"
        cfg['train_cfg']['n_epochs'] = 200
        cfg['train_cfg']['ckpt_freq'] = 5

        os.makedirs(cfg["out_dir"], exist_ok=True)
        with open(f'{cfg["out_dir"]}/config.yaml', 'w') as book:
            yaml.dump(cfg, book, default_flow_style=None, indent=4)

        print(f'running training: {hidden_size}')
        os.system(f'cd {code_dir};{python} {code_dir}/ner_trainer.py --config={cfg["out_dir"]}/config.yaml > {cfg["out_dir"]}/train.log')


# run_me_auto_train()
config = '/home/ubuntu/aEye/old/ner_EncoderRNN_AttnDecoderRNN_1736701049424890334_train_with_test_data/config.yaml'
ckpt = '/home/ubuntu/aEye/old/ner_EncoderRNN_AttnDecoderRNN_1736701049424890334_train_with_test_data/ckpt_000000000100.pth'
os.system(f'cd {code_dir};{python} {code_dir}/ner_demo.py --config={config} --ckpt={ckpt}')
