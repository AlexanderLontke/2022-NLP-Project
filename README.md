# 2022-NLP-Project

# Getting started

To run the chatbot in this repository you first need to create a new python environment and install the project
requirements.

<span style="color:red">## Python 3.8 is required for the dialogue-bot ##</span>

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd bot/dialogue-bot-master
pip install .
```

Now place your `codebase.jsonl` and `embeddings.npy` files which can be found in [Uni SG OneDrive](https://universitaetstgallen-my.sharepoint.com/:f:/g/personal/alexander_lontke_student_unisg_ch/Evl1_xhQqu1ElSPfSn2JTzoBImN8O0wDEqXEz-TbiIWq-A?e=7YZMnH) in the [bot](./bot)
directory.
Furthermore, place the `config.json` and `pytorch_model.bin` files in the [bot/python_model](./bot/python_model)
directory.

Now you can start our chatbot from the root directory with the following commands:
```bash
cd bot
uvicorn our_bot_demo:app
```

and send queries to it with the following commands:

```bash
# code search
curl --location --request POST 'http://localhost:8000/code-search' \
--header 'Content-Type: application/json' \
--data-raw '{
    "user_input": "Create dataframe"
}'
```

```bash
# function explanation
curl --location --request POST 'http://localhost:8000/function-explanation' \
--header 'Content-Type: application/json' \
--data-raw '{
    "user_input": "Give me a better understanding of seaborn.pairplot()"
}'
```