import pandas as pd
from gensim.models import Word2Vec
import random

# 모델 불러오기
model = Word2Vec.load('food_word2vec_model.model')

# CSV 파일 읽어오기
data = pd.read_csv('food_list_test.csv', encoding='CP949')

# 'taste'와 'made_with' 열의 값을 쉼표로 분리하여 리스트로 변환
data['taste'] = data['taste'].apply(lambda x: x.split(','))
data['made_with'] = data['made_with'].apply(lambda x: x.split(','))

# taste와 made_with를 입력받아 결과를 출력합니다.
taste = 'salty'
made_with = 'beef'

# taste와 made_with를 리스트로 변환, 빈 문자열일 경우 빈 리스트로 설정
taste_list = taste.split(',') if taste else []
made_with_list = made_with.split(',') if made_with else []

# 결과 메뉴를 저장할 빈 리스트
matching_menus = []

# 데이터프레임을 반복하면서 일치하는 항목을 찾음
for index, row in data.iterrows():
    menu_taste = row['taste']
    menu_made_with = row['made_with']

    # taste 또는 made_with가 비어있지 않고 모든 조건을 만족하는 경우 해당 메뉴를 추가
    if (not taste_list or all(t in menu_taste for t in taste_list)) and (not made_with_list or all(mw in menu_made_with for mw in made_with_list)):
        matching_menus.append((row['menu_num'], row['romanized_name']))

if matching_menus:
    # 유사한 메뉴 추천
    similarity_scores = {}
    for menu_num, _ in matching_menus:
        similarity_score = model.wv.n_similarity(taste_list + made_with_list, data.loc[data['menu_num'] == menu_num]['taste'].values[0] + data.loc[data['menu_num'] == menu_num]['made_with'].values[0])
        similarity_scores[menu_num] = similarity_score

    # 유사도가 높은 순서로 정렬
    sorted_similarities = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # 상위 3개 메뉴 중에서 랜덤으로 하나 선택
    top_3_similar_menus = sorted_similarities[:3]
    random_menu_num, similarity_score = random.choice(top_3_similar_menus)
    romanized_name = data.loc[data['menu_num'] == random_menu_num]['romanized_name'].values[0]

    print(f'menu_num: {random_menu_num}')
    print(f'romanized_name: {romanized_name}')
    print(f'Similarity Score: {similarity_score}')
else:
    print(f'No matching menu found for Taste "{taste}" and Made with "{made_with}".')