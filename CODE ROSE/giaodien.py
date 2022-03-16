import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Làm sạch câu
def clean_up_sentence(sentence):
    # mã hóa pattern - chia các từ thành mảng
    sentence_words = nltk.word_tokenize(sentence)
    # stem từng từ - tạo short form cho từ
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return mảng túi từ: 0 hoặc 1 cho mỗi từ trong túi tồn tại trong câu

def bow(sentence, words, show_details=True):
    # Mã hóa pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - ma trận N từ, ma trận từ vựng
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # chỉ định 1 nếu current word ở vị trí từ vựng
                bag[i] = 1
                if show_details:
                    print ("Tìm thấy trong bag: %s" % w)
    return(np.array(bag))

# Hàm dự đoán
def predict_class(sentence, model):
    # filter dự đoán dưới ngưỡng
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sắp xếp theo độ mạnh của xác suất
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Hàm nhận được phản hồi
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# hàm rose phản hồi ra giao diện
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# Tạo GUI với tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Rose: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 
# tạo cửa sổ tkinter

base = Tk()
base.title("TRỢ LÍ ẢO ROSE")
base.geometry("500x600")
base.resizable(width=FALSE, height=FALSE)
# Thêm widget vào cửa sổ gốc
#Label=Label(self,image=photoimage)
#Label(base, text = 'send', font =('Verdana', 15)).pack(side = TOP, pady = 10)
# Tạo cửa sổ chat
ChatLog = Text(base, bd=0, bg="#e04c83", height="8", width="50", font=("Arial",15,'bold'),foreground="black")

ChatLog.config(state=DISABLED)

# Liên kết thanh cuộn với cửa sổ Trò chuyện
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Nút tạo để gửi tin nhắn
SendButton = Button(base, font=("Verdana",15,'bold'), text="Gửi", width="12", height=5,
                    bd=0, bg="#e04c83",activeforeground="#181721", activebackground="#230aff",fg='#ffffff',
                    command= send )
# Tạo một đối tượng hình ảnh để sử dụng hình ảnh

# Tạo hộp để nhập tin nhắn
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", SendButton)

# Đặt tất cả các thành phần trên màn hình
scrollbar.place(x=476,y=5, height=476)
ChatLog.place(x=5,y=5, height=470, width=476)
EntryBox.place(x=6, y=500, height=90, width=348)
SendButton.place(x=320,y=500, height=90)

base.mainloop()
