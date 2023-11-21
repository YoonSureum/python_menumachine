# 필요한 모듈을 import합니다.
from flask import Flask, request, jsonify  # Flask 모듈을 사용하여 웹 서버를 만듭니다.
import pandas as pd  # 데이터 프레임을 다루기 위한 Pandas 라이브러리를 import합니다.
from gensim.models import Word2Vec  # Word2Vec 모델을 사용하기 위한 Gensim 라이브러리를 import합니다.
import random  # 무작위 선택을 위한 random 모듈을 import합니다.

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)

# '/get_matching_menu' 엔드포인트를 생성하고 POST 메서드로 요청을 처리하는 함수를 정의합니다.
@app.route('/get_matching_menu', methods=['POST'])
def get_matching_menu():
    # POST 요청으로부터 'taste'와 'made_with' 값을 받아옵니다.
    print("------------model start------------")
    data = request.json
    taste = data.get('taste')
    made_with = data.get('made_with')

    # 이전 코드에서 데이터 프레임과 Word2Vec 모델을 로드합니다.
    model = Word2Vec.load('food_word2vec_model.model')
    data = pd.read_csv('food_list_test.csv', encoding='CP949')

    data['taste'] = data['taste'].apply(lambda x: x.split(','))
    data['made_with'] = data['made_with'].apply(lambda x: x.split(','))

    # 'taste'와 'made_with' 값을 쉼표로 분리하여 리스트로 변환합니다.
    taste_list = taste.split(',') if taste else []
    made_with_list = made_with.split(',') if made_with else []

    matching_menus = []  # 일치하는 메뉴를 저장할 빈 리스트를 만듭니다.

    # 데이터프레임을 반복하면서 일치하는 항목을 찾습니다.
    for index, row in data.iterrows():
        menu_taste = row['taste']
        menu_made_with = row['made_with']

        # taste 또는 made_with가 비어있지 않고 모든 조건을 만족하는 경우 해당 메뉴를 추가
        if (not taste_list or all(t in menu_taste for t in taste_list)) and (
                not made_with_list or all(mw in menu_made_with for mw in made_with_list)):
            matching_menus.append((row['menu_num'], row['romanized_name']))

    if matching_menus:
        # 유사한 메뉴 추천
        similarity_scores = {}
        for menu_num, _ in matching_menus:
            similarity_score = model.wv.n_similarity(taste_list + made_with_list,
                                                     data.loc[data['menu_num'] == menu_num]['taste'].values[0] +
                                                     data.loc[data['menu_num'] == menu_num]['made_with'].values[0])
            similarity_scores[menu_num] = similarity_score

        # 유사도가 높은 순서로 정렬
        sorted_similarities = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 3개 메뉴 중에서 랜덤으로 하나 선택
        top_3_similar_menus = sorted_similarities[:3]
        random_menu_num, similarity_score = random.choice(top_3_similar_menus)
        romanized_name = data.loc[data['menu_num'] == random_menu_num]['romanized_name'].values[0]
        result = {
            'menu_num': random_menu_num,
            'food_name': romanized_name

        }
    else:
        # 일치하는 메뉴가 없을 경우 해당 메시지를 반환합니다.
        result = {
            'message': f'No matching menu found for Taste "{taste}" or Made with "{made_with}".'
        }
    print("------------model end------------")
    # 결과를 JSON 형식으로 반환합니다.
    return jsonify(result)

# 스크립트를 직접 실행할 때만 Flask 애플리케이션을 실행합니다.
if __name__ == '__main__':
    app.run(port=8088)
