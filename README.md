
# Named Entity Recognition (NER)

This repository provides tools to train and run a Named Entity Recognition (NER) model using Python. It includes:
- `ner_trainer.py`: For training the NER model.
- `ner_demo.py`: For running a Gradio-based demo of the trained model.

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vishnu-chand/named_entity_recognition.git
   cd named_entity_recognition/ner
   ```

2. **Install Dependencies:**
   Create and activate a Conda environment named `llm`:
   ```bash
   conda create --name llm python=3.10.14 -y
   conda activate llm
   ```
   Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the model, run the following command. The training process will automatically prepare the necessary data:

1. **Set Paths:**
   ```bash
   export out_dir=/path/to/output_directory
   export code_dir=/path/to/code_directory
   ```

2. **Run Training Command:**
   ```bash
   cd $code_dir; python $code_dir/ner_trainer.py --config=$out_dir/config.yaml > $out_dir/train.log
   ```

### Notes:
- Replace `/path/to/output_directory` with the desired output directory for logs, checkpoints, and other files.
- Replace `/path/to/code_directory` with the directory where the code is located.
- The training logs will be saved in `$out_dir/train.log`.

## Running the Demo

After training the model, you can start a Gradio-based demo to test it interactively:

1. **Set Paths:**
   ```bash
   export out_dir=/path/to/output_directory
   export code_dir=/path/to/code_directory
   ```

2. **Start the Demo:**
   ```bash
   cd $code_dir; python $code_dir/ner_demo.py --config=$out_dir/config.yaml --ckpt=$out_dir/ckpt_000000000100.pth
   ```

### Notes:
- Ensure that the checkpoint file (`ckpt_000000000100.pth`) exists in the output directory.
- The Gradio demo will be launched in your web browser.

## Repository Structure and Important Files
```
.
├── ner_trainer.py    # Script for training the NER model
├── ner_demo.py       # Script for running the Gradio demo
├── requirements.txt  # Dependencies for the project
├── config.yaml       # Example configuration file for training and demo
```

## Important YAML Configuration Fields

Below are some key fields from the configuration file (`config.yaml`):

```yaml
out_dir: /home/ubuntu/aEye/old/ner_result
max_length: 32       # Maximum sentence length: Limits the number of words in a sentence to 32

model_cfg:
  EncoderRNN:
    dropout_p: 0.1   # Dropout probability to prevent overfitting
    hidden_size: 128 # Hidden size of the encoder RNN
  DecoderRNN:
    hidden_size: 128 # Hidden size of the decoder RNN
  AttnDecoderRNN:
    dropout_p: 0.1   # Dropout probability to prevent overfitting
    hidden_size: 128 # Hidden size of the attention-based decoder RNN

train_cfg:
  bs: 16         # Batch size: Defines how many samples to process in each iteration
  lr: 0.0001     # Learning rate: Defines how quickly the model adjusts weights during training
  n_epochs: 100  # Number of epochs: The total number of times the model will see the entire dataset

  ckpt_freq: 5           # Frequency of saving model checkpoints (every 5 epochs)
  stop_patience: 15      # Patience before stopping training if the performance doesn't improve
  stop_metric: eval_loss # Metric to monitor for early stopping (based on evaluation loss)

  encoder: EncoderRNN     # Using EncoderRNN as the encoder model
  decoder: AttnDecoderRNN # Using AttnDecoderRNN as the decoder model

  enable_class_weight: false # Flag to enable class weighting during training (set to false)

data_cfg:
  n_pad: 0           # Padding value for sequences
  data_set: mac_2020 # The dataset being used for training

  simple_cleaners: false  # enable/disable simple cleaners
  advance_cleaners: false # enable/disable advance cleaners

  enable_phonemize: false            # Phonemization flag to convert text into phonetic representation
  enable_expand_abbreviations: false # Flag to expand abbreviations in the text
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Feel free to raise issues or create pull requests to improve the project.

## References
- [BRAT Standoff Format Documentation](https://brat.nlplab.org/standoff.html)
- [Bionlp 2021 Paper](https://aclanthology.org/2021.bionlp-1.16.pdf)
- [Named Entity Recognition Overview - GeeksforGeeks](https://www.geeksforgeeks.org/named-entity-recognition/)
- [Named Entity Recognition Code - GitHub](https://github.com/BeiqiZh/Named-Entity-Recognition/tree/main/code)
- [Named Entity Recognition with RNN - Kaggle Notebook](https://www.kaggle.com/code/ritvik1909/named-entity-recognition-rnn)
- [NER Code using Python - GitHub](https://github.com/maxslimb/Named-Entity-Recognition)
- [VITS GitHub Repository](https://github.com/jaywalnut310/vits.git)
- [NER Using Keras LSTM and Spacy - Medium Article](https://zhoubeiqi.medium.com/named-entity-recognition-ner-using-keras-lstm-spacy-da3ea63d24c5)
- [PyTorch Seq2Seq Translation Tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)

