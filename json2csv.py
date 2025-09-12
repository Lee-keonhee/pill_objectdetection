import json
import pandas as pd


# with open('./predictions/test_predictions.json', 'r', encoding='utf-8') as f:
#     anno = json.load(f)
#
#     df1 = pd.read_json(anno)

df = pd.read_json('./predictions/test_predictions.json', encoding='utf-8', orient='records')
print(df.head())

bbox_expanded = df['bbox'].apply(pd.Series)

bbox_expanded.columns = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']

df_final = pd.concat([df, bbox_expanded], axis=1)

df_final = df_final.drop(['bbox','category_name'], axis=1)
df_final['annotation_id'] = df_final.index + 1
df_final = df_final[['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']]

print(df_final.head())

# CSV 파일로 저장할 경로 및 이름 지정
output_csv_file_path = "output_data.csv"

try:
    # to_csv() 메서드를 사용하여 DataFrame을 CSV 파일로 저장합니다.
    # index=False: DataFrame의 인덱스를 CSV 파일에 저장하지 않습니다. (보통 이렇게 합니다)
    # encoding='utf-8': 한글 등이 포함된 경우를 대비해 인코딩을 지정합니다.
    df_final.to_csv(output_csv_file_path, index=False, encoding='utf-8')

    print(f"\nDataFrame이 '{output_csv_file_path}' 파일로 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"오류: CSV 파일 저장 중 문제가 발생했습니다: {e}")