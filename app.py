import matplotlib
matplotlib.use('Agg')
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, Response
import requests
from bs4 import BeautifulSoup
import torch
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from io import StringIO
import base64
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)

###The articles in the dataset have tags assigned to them, and only articles with at least one of these tags is considered
relavent_tags = ['Science', "Machine Learning", "Artificial Intelligence", "Health", "Coronavirus"]

def RemoveNewLines(text):
    text = text.replace("\n", " ")
    return text

###Checks whether rows in the dataset contain the above tags
def CheckTag(tags):
    for tag in relavent_tags:
        if tag in eval(tags):
            return True
    return False

###https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
def MeanPooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

###Converts the text into input ids, and attention masks, which the model can then process
def prepare_text(text, max_len):
    example = tokenizer(text, max_length=max_len, padding="max_length", truncation=True)
    example["input_ids"] = torch.tensor(example["input_ids"]).unsqueeze(0)
    example["attention_mask"] = torch.tensor(example["attention_mask"]).unsqueeze(0)
    return example

###Given the embedding of a text, it finds the n most similar embeddings in the dataset
def CheckSimilarity(text_embedding, num_sites):
    similarities = np.zeros(len(embeddings))
    for e in range(len(embeddings)):
        similarities[e] = cosine_similarity(embeddings[e].reshape(1, -1), text_embedding.detach().cpu().numpy().reshape(1, -1))
        indices = np.argsort(similarities)[::-1][:num_sites]
    return indices, similarities[indices]

###obtains the attention from a specific layer, and head of the model
def get_transformer_attention(model, text, text_start, text_end, max_len):
    text = " ".join(text.split(" ")[text_start:text_end])
    example = prepare_text(text, max_len)
    with torch.no_grad():
        embed = model(**example)[-1]
    input_id_list = example["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return embed, tokens

###Returns cleaned word tokens, and collects attention
def visualize_attention(model, text, text_start, text_end, num_head, num_layer):
    attention, tokens = get_transformer_attention(model, text, text_start, text_end, max_len=text_end-text_start)
    tokens = [token.replace("Ġ", "") for token in tokens]
    return attention[num_layer][0][num_head].detach().cpu().numpy(), tokens

###Gets the n most important sentences in the provided article
###Citation: https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/ch09
def SentenceWordImportance(text, num_display):
    sentences = tokenize.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    words_tfidf = vectorizer.fit_transform(sentences)
    important_sentences = []
    sent_sum = words_tfidf.sum(axis=1)
    important_sent = np.argsort(sent_sum, axis=0)[::-1]

    for i in range(0, len(sentences)):
        if i in important_sent[:num_display]:
            important_sentences.append(sentences[i])
    return important_sentences

###Get's the most frequent named entities
def GetEntities(text):
    doc = nlp(text)
    items = [e.text for e in doc.ents if e.label_ is "ORG"]
    items = [(x, y) for (x, y) in dict(Counter(items)).items()]
    return items

###Loads the pretrained embeddings from linghacks-training-notebook.ipynb, the model from distilrobertapretraining.ipynb, data, and the tokenizer

embeddings = np.load("medium_embeddings.npy")
data = pd.read_csv("medium_articles.csv")
data["contains_tag"] = data["tags"].apply(CheckTag)
data = data[data["contains_tag"] == True]

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("checkpoint-1000", output_attentions=True)
untrained_model = AutoModel.from_pretrained("distilroberta-base", output_attentions=True)

@app.route("/")
def home():
    return render_template("main_page.html")

@app.route("/", methods=['POST', 'GET'])
def collect_url():
    url = request.form['url_input']
    num_sites = int(request.form['num_sites'])
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    global text 
    text = soup.get_text()
    example = prepare_text(text, 512)
    with torch.no_grad():
        embed = model(**example)["last_hidden_state"].squeeze()
    pooler_output = MeanPooling(embed.unsqueeze(0), example["attention_mask"].reshape(1, 512))
    indices, similarities = CheckSimilarity(pooler_output, num_sites)
    datatable = {
            "Article Links" : [],
            "Similarity Scores" : [],
            "Article Numbers" : [],
            }
    all_tags = []
    for idx in range(len(indices)):
        datatable["Article Links"].append(data["url"].values[indices[idx]])
        datatable["Similarity Scores"].append(similarities[idx])
        datatable["Article Numbers"].append(idx+1)
        all_tags.append(eval(data["tags"].values[indices[idx]]))
    processed_tags = " ".join([x for y in all_tags for x in y]) + " "
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(processed_tags)
    plt.figure(figsize=(20, 10), facecolor="#CFF5C6")
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('static/foo2.png', bbox_inches='tight')
    datatable = pd.DataFrame(datatable)
    return render_template("main_page.html", columns = datatable.columns.values, data_values=list(datatable.values.tolist()), zip=zip, title="Results Similar To The Article You Provided:")
    
@app.route("/visuals",  methods=['POST'])
def get_visuals():
    layer_num = int(request.form['layerRange'])-1
    head_num = int(request.form['headRange'])-1
    text_start = int(request.form['textStart'])-1
    text_end = int(request.form['textEnd'])-1
    sns.set(style="darkgrid")
    attn, tokens = visualize_attention(model, text, text_start, text_end, head_num, layer_num)
    plt.figure(figsize=(20, 10), facecolor="#CFF5C6")
    sns.heatmap(attn, yticklabels=tokens, xticklabels=tokens)
    plt.savefig('static/foo.png', bbox_inches='tight')
    attn, tokens = visualize_attention(untrained_model, text, text_start, text_end, head_num, layer_num)
    plt.figure(figsize=(20, 10), facecolor="#CFF5C6")
    sns.heatmap(attn, yticklabels=tokens, xticklabels=tokens)
    plt.savefig('static/foo1.png', bbox_inches='tight')
    return render_template("visuals_page.html", user_image="templates/foo.png", text_len = len(text.split()), int=int, test_segment=" ".join(text.split(" ")[text_start:text_end]), num_head=head_num, num_layer=layer_num)

@app.route("/further_insight",  methods=['POST'])
def show_further_visuals():
    num_display =  int(request.form['displayNumber'])
    important_sents = SentenceWordImportance(text, num_display)
    all_entities = GetEntities(text)
    all_entities.sort(key=lambda x: x[1], reverse=True)
    all_entities = all_entities[:3]
    return render_template("additional_visuals.html", important_sents=important_sents, all_entities=all_entities)

@app.route("/visuals")
def visual_page():
    return render_template("visuals_page.html", text_len = len(text.split()))

@app.route("/about")
def show_about():
    return render_template("about_page.html")

@app.route("/future_ideas")
def show_future_ideas():
    return render_template("future_ideas_page.html")

@app.route("/methods")
def show_methods():
    return render_template("methods_page.html")

@app.route("/further_insight")
def show_additional_visuals():
    return render_template("additional_visuals.html")

if __name__ == '__main__':
	app.run(debug = True,port = 5001)