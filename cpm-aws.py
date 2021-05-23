#!/usr/bin/env python
# coding: utf-8

# pip install awscli sagemaker --user

# from sagemaker.tensorflow import TensorFlowModel
# 
# model = TensorFlowModel(framework_version="2.2.2", model_data='s3://model-deploy/cpm-lm-tf2-fp16.tar.gz', role='arn:aws-cn:iam::907067680148:role/service-role/AmazonSageMaker-ExecutionRole-20210422T153180')
# 
# predictor = model.deploy(initial_instance_count=1, instance_type='ml.g4dn.xlarge')

# In[1]:


from sagemaker.predictor import Predictor

endpoint = "tensorflow-inference-2021-04-28-15-31-51-433"

predictor = Predictor(endpoint_name=endpoint)


# In[2]:


import tensorflow as tf
import json
from gpt2_tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer(
    'CPM-Generate/bpe_3w_new/vocab.json',
    'CPM-Generate/bpe_3w_new/merges.txt',
    model_file='CPM-Generate/bpe_3w_new/chinese_vocab.model')


# In[3]:


length = 500
sentence = input("请输入: ")
inputs = tf.constant([tokenizer.encode(sentence)], dtype=tf.int64)
inputs = inputs.numpy().tolist()
inputs = inputs[0]
data = json.dumps({"inp":inputs, "length":length})
ret = predictor.predict(data, initial_args={'ContentType': 'application/json'})
ret = json.loads(ret)
ret = ret['predictions']
print(tokenizer.decode(ret[0]).replace(' ', ''))


# In[ ]:




