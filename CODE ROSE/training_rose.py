import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#đọc dữ liệu vào # + xử lí dữ liệu đầu vào
words=[]
classes = []
documents = []
ignore_words = ['!','#','$','^','*','+','|','?','`','~']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # mã hóa mỗi từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Thêm tài liệu vào kho dữ liệu
        documents.append((w, intent['tag']))

        # thêm vào lớp danh sách
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# bổ đề và giảm bớt từng từ và xóa các từ trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# Tài liệu = sự kết hợp giữa patterns và intents
print (len(documents), "tài liệu")
# classes = intents
print (len(classes), "lớp", classes)
# Các từ = Tất cả các từ, từ vựng
print (len(words), "Tất cả từ vựng", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# tạo dữ liệu training
training = []
# Tạo mảng trống cho đầu ra 
output_empty = [0] * len(classes)
# Tập training,túi từ cho mỗi câu
for doc in documents:
    # khởi tạo túi
    bag = []
    # Các từ được mã hóa cho mẫu
    pattern_words = doc[0]
    # lemmatize từng từ, tạo từ cơ sở,biểu diễn các từ có liên quan với pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Tạo mảng túi từ bằng 1 và kết hợp từ với pattern hiện tại nếu tìm thấy
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # đầu ra là '0' cho mỗi tab và '1' cho tab hiện tại (cho mỗi pattern )
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    #output.append(output_row)
    training.append([bag, output_row])
# shuffle và biến thành np.array
random.shuffle(training)
training = np.array(training)
# Tạo danh sách training và kiểm tra. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training thành công !!!")


# Tạo model - 3 lớp: Lớp đầu tiên 128 tế bào nơ-ron.
#                    Lớp thứ hai 64 tế bào nơ-ron.
#                    Lớp thứ ba (Lớp OutPut) chứa số lượng tế bào nơ-ron đầu ra.
# Bằng số intents để đoán intent đầu ra với softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Biên dịch model. gradient ngẫu nhiên với gradient gia tốc Nesterov mang lại kết quả tốt cho model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting và lưu model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Tạo model Rose thành công,bla bla bla !!!")
