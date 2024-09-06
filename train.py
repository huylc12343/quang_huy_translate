import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
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

import pandas as pd
# Ví dụ dữ liệu huấn luyện
data = pd.read_csv("combined_translations.csv", encoding="utf-8")
# Ví dụ dữ liệu huấn luyện 
input_texts = data['source'].tolist()


target_texts = data['translation'].tolist()

# Thêm token đặc biệt
target_texts = ['<start> ' + text + ' <end>' for text in target_texts]

# Tokenize dữ liệu huấn luyện
input_tokens = [tokenize_en(text) for text in input_texts]
target_tokens = [tokenize_vi(text) for text in target_texts]
# Tạo từ điển cho token
input_token_index = {}
target_token_index = {'<start>': 1, '<end>': 2}


for tokens in input_tokens:
    for token in tokens:
        if token not in input_token_index:
            input_token_index[token] = len(input_token_index) + 1

for tokens in target_tokens:
    for token in tokens:
        if token not in target_token_index and token not in ['<start>', '<end>']:
            target_token_index[token] = len(target_token_index) + 1
# Đảo ngược từ điển token để chuyển chỉ số thành token
reverse_target_token_index = {i: token for token, i in target_token_index.items()}

# Tạo chuỗi đầu vào và đầu ra cho mô hình padding
max_encoder_seq_length = max([len(tokens) for tokens in input_tokens])
max_decoder_seq_length = max([len(tokens) for tokens in target_tokens])

encoder_input_data = np.zeros((len(input_tokens), max_encoder_seq_length), dtype='float32')
decoder_input_data = np.zeros((len(target_tokens), max_decoder_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(target_tokens), max_decoder_seq_length, len(target_token_index) + 1), dtype='float32')

for i, tokens in enumerate(input_tokens):
    for t, token in enumerate(tokens):
        encoder_input_data[i, t] = input_token_index[token]

for i, tokens in enumerate(target_tokens):
    for t, token in enumerate(tokens):
        decoder_input_data[i, t] = target_token_index[token]
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[token]] = 1.0

# Xây dựng mô hình LSTM
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=len(input_token_index) + 1, output_dim=256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(target_token_index) + 1, output_dim=256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(target_token_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=300, validation_split=0.2)

# Xây dựng mô hình encoder và decoder cho việc dự đoán
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Dự đoán chuỗi đầu ra
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

# Nhập câu tiếng Anh từ bàn phím và dịch sang tiếng Việt

input_sentence = "I like the beach."
input_tokens = tokenize_en(input_sentence)
input_seq = np.zeros((1, max_encoder_seq_length))
for t, word in enumerate(input_tokens):
    if word in input_token_index:
        input_seq[0, t] = input_token_index[word]

decoded_sentence = decode_sequence(input_seq)
print('Translated sentence:', decoded_sentence)

# Lưu mô hình encoder và decoder
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
model.save('seq2seq_model.h5')
np.savez('token_indices.npz', input_token_index=input_token_index, target_token_index=target_token_index, reverse_target_token_index=reverse_target_token_index, max_encoder_seq_length=max_encoder_seq_length, max_decoder_seq_length=max_decoder_seq_length)
#cao nhất 0.4273/4264/0.4286/0.4331/0.4356/0.4365/0.4373/0.4381/0.4404/0.4419/0.4421/0.4451/0.4472/0.4487/0.4499/0.4520/0.4562
# Lưu từ điển token
# np.savez('token_indices.npz', input_token_index=input_token_index, target_token_index=target_token_index, reverse_target_token_index=reverse_target_token_index)
# Lưu từ điển token và các giá trị độ dài chuỗi

print("Models and token indices have been saved.")
