import falcon

app = application = falcon.App(cors_enable=True)

from sagemaker.predictor import Predictor
endpoint = "tensorflow-inference-2021-04-28-15-31-51-433"
predictor = Predictor(endpoint_name=endpoint)

import tensorflow as tf
import json
from gpt2_tokenizer import GPT2Tokenizer
import time

tokenizer = GPT2Tokenizer(
    'CPM-Generate/bpe_3w_new/vocab.json',
    'CPM-Generate/bpe_3w_new/merges.txt',
    model_file='CPM-Generate/bpe_3w_new/chinese_vocab.model')

class Generate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_post(self, req, resp):
        body = json.loads(req.stream.read())
        text = body["text"]
        t0 = time.time()
        length = 300
        inputs = tf.constant([tokenizer.encode(text)], dtype=tf.int64)
        inputs = inputs.numpy().tolist()
        inputs = inputs[0]
        data = json.dumps({"inp":inputs, "length":length})
        ret = predictor.predict(data, initial_args={'ContentType': 'application/json'})
        ret = json.loads(ret)
        ret = ret['predictions']
        result = self.tokenizer.decode(ret[0]).replace(' ', '')
        result = {'result': str(result)}
        t1 = time.time()
        print("time:", t1-t0)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(result, ensure_ascii=False)
        

generate = Generate(tokenizer)
app.add_route('/predict', generate)