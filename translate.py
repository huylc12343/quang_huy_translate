input_sentence = input("Please enter a sentence: ")
print(input_sentence)
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import spacy
from pyvi.ViTokenizer import tokenize
# Tải mô hình SpaCy tiếng Anh
nlp_en = spacy.load('en_core_web_sm')

# Hàm để token hóa câu tiếng Anh bằng SpaCy
def tokenize_en(text):
    return [token.text for token in nlp_en(text)]

# Hàm để token hóa câu tiếng Việt bằng PyVi
def tokenize_vi(text):
    return tokenize(text).split()

# Tải các mô hình đã lưu
encoder_model = load_model(r'.\encoder_model.h5')
decoder_model = load_model(r'.\decoder_model.h5')

# Tải từ điển token và các giá trị độ dài chuỗi với allow_pickle=True
npzfile = np.load(r'.\token_indices.npz', allow_pickle=True)
input_token_index = npzfile['input_token_index'].item()
target_token_index = npzfile['target_token_index'].item()
reverse_target_token_index = npzfile['reverse_target_token_index'].item()
max_encoder_seq_length = npzfile['max_encoder_seq_length'].item()
max_decoder_seq_length = npzfile['max_decoder_seq_length'].item()

# Hàm decode_sequence để dịch câu
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_token_index[sampled_token_index]

        decoded_sentence += ' ' + sampled_token

        if sampled_token == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence
    # return decoded_sentence.strip('<start> ').strip(' <end>')

# Hàm để dịch một câu từ tiếng Anh sang tiếng Việt
def translate_sentence(input_sentence):
    translations = []
    input_tokens = tokenize_en(input_sentence)
    input_seq = np.zeros((1, max_encoder_seq_length))
    for t, word in enumerate(input_tokens):
        if word in input_token_index:
            input_seq[0, t] = input_token_index[word]
    
    # decoded_sentence = decode_sequence(input_seq)

    decoded_sentence = decode_sequence(input_seq)
    # decoded_sentence = decoded_sentence.replace("_", " ")
    decoded_sentence = decoded_sentence.replace("start", "")
    decoded_sentence = decoded_sentence.replace("end", "")
    decoded_sentence = decoded_sentence.replace(">", "")
    decoded_sentence = decoded_sentence.replace("<", "")
    translations.append(decoded_sentence.strip())
    return decoded_sentence

# Nhập câu tiếng Anh từ bàn phím và dịch sang tiếng Việt
# input_sentence = "I will give you a hint"
translated_sentence = translate_sentence(input_sentence)
print("Input_sentence:",input_sentence)
print('Translated sentence:', translated_sentence)
