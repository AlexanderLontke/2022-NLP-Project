from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

query = "Download an image and save the content in output_dir"
query_vec = model(tokenizer(query,return_tensors='pt')['input_ids'])[1]
code_1="""
def f(image_url, output_dir):
    import requests
    r = requests.get(image_url)
    with open(output_dir, 'wb') as f:
        f.write(r.content)
"""
code1_vec = model(tokenizer(code_1,return_tensors='pt')['input_ids'])[1]
code_2="""
def f(image, output_dir):
    with open(output_dir, 'wb') as f:
        f.write(image)
"""
code2_vec = model(tokenizer(code_2,return_tensors='pt')['input_ids'])[1]
code_3="""
def f(image_url, output_dir):
    import requests
    r = requests.get(image_url)
    return r.content
"""
code3_vec = model(tokenizer(code_3,return_tensors='pt')['input_ids'])[1]
code_vecs=torch.cat((code1_vec,code2_vec,code3_vec),0)
codes = [code_1,code_2,code_3]
scores=torch.einsum("ab,cb->ac",query_vec,code_vecs)
scores=torch.softmax(scores,-1)
print("")
print("Query:",query)
for i in range(3):
    print("Code:",codes[i])
    print("Score:",scores[0,i].item())