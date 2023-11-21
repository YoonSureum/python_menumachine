import pandas as pd
from gensim.models import Word2Vec

# CSV 파일 읽어오기
data = pd.read_csv('food_list_test.csv', encoding='CP949')

# 'taste'와 'made_with' 열의 값을 쉼표로 분리하여 리스트로 변환하고 공백 제거
data['taste'] = data['taste'].apply(lambda x: [word.strip() for word in x.split(',')])
data['made_with'] = data['made_with'].apply(lambda x: [word.strip() for word in x.split(',')])

# Word2Vec 모델을 훈련하기 위해 데이터 추출
all_words = []
for taste in data['taste']:
    all_words.extend(taste)  # 리스트 확장
for made_with in data['made_with']:
    all_words.extend(made_with)  # 리스트 확장

# Word2Vec 모델 훈련
word2vec_model = Word2Vec(sentences=[all_words], vector_size=100, window=5, min_count=1, sg=0)

# 모델 저장
word2vec_model.save('food_word2vec_model.model')