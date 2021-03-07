import math
from collections import defaultdict
 
corpus = [
    "what is the weather like today",
    "what is for dinner tonight",
    "this is a question worth pondering",
    "it is a beautiful day today"
]
words = []
# 对corpus分词
for sentence in corpus:
    words.append(sentence.strip().split())

#print(words)

# 如果有自定义的停用词典，我们可以用下列方法来分词并去掉停用词
# f = ["is", "the"]
# for i in corpus:
#     all_words = i.split()
#     new_words = []
#     for j in all_words:
#         if j not in f:
#             new_words.append(j)
#     words.append(new_words)
# print(words)
 
# 进行词频统计  每个句子的词频
def Counter(words):
    word_count = []
    for sentence in words:
        word_dict = defaultdict(int)
        for word in sentence:
            word_dict[word] += 1
        word_count.append(word_dict)
    return word_count
 
word_count = Counter(words)
#print(word_count) #每个句子的词频
 
# 计算TF(word代表被计算的单词，word_dict是被计算单词所在句子分词统计词频后的字典)
def tf(word, word_dict):
    return word_dict[word] / sum(word_dict.values())
print(words,word_dict)
 
# 统计含有该单词的句子数
def count_sentence(word, word_count):
    return sum([1 for i in word_count if i.get(word)])  # i[word] >= 1
 
# 计算IDF
def idf(word, word_count):
    return math.log(len(word_count) / (count_sentence(word, word_count) + 1))
 
# 计算TF-IDF
def tfidf(word, word_dict, word_count):
    return tf(word, word_dict) * idf(word, word_count)
 
#p = 1
#for word_dict in word_count:
    #print("part:{}".format(p))
    #p += 1
   # for word, cnt in word_dict.items():
       # print("word: {} ---- TF-IDF:{}".format(word, tfidf(word, word_dict, word_count)))
