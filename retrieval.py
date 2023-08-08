import requests
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data import QueryLaw


def send_post_request(url, data):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def predict():
    url = "http://172.26.1.155:13005/law_search"
    file_path = "./test.json"
    data_set = QueryLaw(file_path)
    data_iter = DataLoader(data_set, batch_size=1, shuffle=False)

    predicts = []
    for data in tqdm(data_iter):
        query = data['query']
        response = send_post_request(url, query)
        if response:
            response_list = response.json()['body']
            predicts.append(response_list)
        else:
            print("Failed to send POST request.")
    with open('./predicts.json', 'w') as f:
        json.dump(predicts, f)


predict()
