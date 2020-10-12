#%%
##### REQUIRMENTS
import spacy
nlp = spacy.load('en')
nlp.max_length = 1000000
from collections import Counter
import pandas as pd
import numpy as np
from utils.config import config_dict
import utils.db_toolbox as tb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.stats import pearsonr
from numpy import cov
from scipy.stats import spearmanr
from utils.sentiment import sentiment
from utils.query import sql_element, sql_price
#import textacy
import nltk
#geocodes
from functools import partial
from geopy.geocoders import Nominatim
#%%
##### SINGLE WORD FREQUENCY
def nlp_function(element,position,date_from,date_to):

    option = ['pos','neg']
    database = sql_element(element,date_from,date_to)
    
    #POSITIVE
    pwd = database[database['sentiment']=='pos']

    doc = nlp(' '.join(pwd['combined'].str.lower().tolist()))

    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True and len(token.text) > 3 and token.text != element and token.pos_ == position]

    word_freq = Counter(words)

    pwd = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
    pwd.columns = ['word', 'pos'+'_'+'frequency']
    pwd.index = pwd['word']
    del pwd['word']
    pos = pwd.sort_values(by=['pos'+'_'+'frequency'], ascending=False)
    
    #NEGATIVE
    nwd = database[database['sentiment']=='neg']

    doc = nlp(' '.join(nwd['combined'].str.lower().tolist()))

    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True and len(token.text) > 3 and token.text != element and token.pos_ == position]

    word_freq = Counter(words)

    nwd = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
    nwd.columns = ['word', 'neg'+'_'+'frequency']
    nwd.index = nwd['word']
    del nwd['word']
    neg = nwd.sort_values(by=['neg'+'_'+'frequency'], ascending=False)

    bar = pd.concat([pos, neg], axis=1)
    bar= bar.fillna(0)
    bar['total'] = bar['pos_frequency'] + bar['neg_frequency']
    
    return bar, pos, neg

#%% ngram function
def ngram_function(element,date_from,date_to):

    database = sql_element(element,date_from,date_to)
    
    #POSITIVE
    pwd = database[database['sentiment']=='pos']

    doc = nlp(' '.join(pwd['combined'].str.lower().tolist()))
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True and len(token.text) > 3 and token.text != element]
    #Create your bigrams
    bgs = nltk.bigrams(words)
    #compute frequency distribution for all the bigrams in the text
    pos_fdist = nltk.FreqDist(bgs)
    pos_key_list = []
    pos_item_list = []
    for k,v in pos_fdist.items():
        pos_key_list.append(' '.join(k))
        pos_item_list.append(v)
    pos_df = pd.DataFrame(list(zip(pos_key_list, pos_item_list)),
              columns=['bigram','pos_frequency'])
    pos_df.sort_values(by=['pos_frequency'], ascending=False, inplace=True)
    pos_df.index = pos_df['bigram']
    del pos_df['bigram']
    
    #NEGATIVE
    nwd = database[database['sentiment']=='neg']

    doc = nlp(' '.join(nwd['combined'].str.lower().tolist()))
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True and len(token.text) > 3 and token.text != element]
    #Create your bigrams
    bgs = nltk.bigrams(words)
    #compute frequency distribution for all the bigrams in the text
    neg_fdist = nltk.FreqDist(bgs)
    neg_key_list = []
    neg_item_list = []
    for k,v in neg_fdist.items():
        neg_key_list.append(' '.join(k))
        neg_item_list.append(v)
    neg_df = pd.DataFrame(list(zip(neg_key_list, neg_item_list)),
              columns=['bigram','neg_frequency'])
    neg_df.sort_values(by=['neg_frequency'], ascending=False, inplace=True)
    neg_df.index = neg_df['bigram']
    del neg_df['bigram']

    # TOTAL
    
    total_bigram = pd.concat([pos_df, neg_df], axis=1)
    total_bigram = total_bigram.fillna(0)
    total_bigram['total'] = total_bigram['pos_frequency'] + total_bigram['neg_frequency']
    total_bigram.sort_values(by=['total'], ascending=False, inplace=True)
    
    return total_bigram


#%%
    
##### NER WORDS
def ner_words(element,cat_type,date_from,date_to):
    
    option = ['pos','neg']
    database = sql_element(element,date_from,date_to)
    
    #NER
    for i in option: 
        
            db = database[database['sentiment']==i]
            doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
            
            # entitiy
            ents = [j.text for j in doc.ents if len(j.text) > 3 and j.label_ == cat_type]
    
    
            # five most common tokens
            ent_freq = Counter(ents)
            #common_words = ent_freq.most_common(5)
    
            et2 = pd.DataFrame.from_dict(ent_freq, orient='index').reset_index()
            et2.columns = ['ent', i+'_'+'frequency']
            et2.index = et2['ent']
            del et2['ent']
            globals()[i] = et2.sort_values(by=[i+'_'+'frequency'], ascending=False)
            
                
    et2 = pd.concat([pos, neg], axis=1)
    et2 = et2.fillna(0)
    et2['total'] = et2['pos_frequency'] + et2['neg_frequency']
    et2 = et2.sort_values(by=['total'], ascending=False)
    
    return et2

# 
#%%
#### NER FUNCTION
def ner_function(element,my_nlp,date_from,date_to):
    
    database = sql_element(element,date_from,date_to)
    
    if my_nlp == 'ENT':
        #positive
        db = database[database['sentiment']=='pos']
        doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
        
        # entitiy
        ents = [j.label_ for j in doc.ents if len(j.text) > 3]
    
        # five most common tokens
        ent_freq = Counter(ents)
        # common_words = ent_freq.most_common(5)
    
        ner_pos = pd.DataFrame.from_dict(ent_freq, orient='index').reset_index()
        ner_pos.columns = ['ent', 'pos_frequency']
        ner_pos.index = ner_pos['ent']
        del ner_pos['ent']
        #ner_pos = ner_pos.drop(index='WORK_OF_ART')
        ner_pos = ner_pos.sort_values(by=['pos_frequency'], ascending=False)
        
        #negative
        db = database[database['sentiment']=='neg']
        doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
        
        # entitiy
        ents = [j.label_ for j in doc.ents if len(j.text) > 3]
    
        # five most common tokens
        ent_freq = Counter(ents)
        # common_words = ent_freq.most_common(5)
    
        ner_neg = pd.DataFrame.from_dict(ent_freq, orient='index').reset_index()
        ner_neg.columns = ['ent', 'neg_frequency']
        ner_neg.index = ner_neg['ent']
        del ner_neg['ent']
        #ner_neg = ner_neg.drop(index='WORK_OF_ART')
        ner_neg = ner_neg.sort_values(by=['neg_frequency'], ascending=False)
                    
        et = pd.concat([ner_pos,ner_neg], axis=1)
        et = et.fillna(0)
        et['total'] = et['pos_frequency'] + et['neg_frequency']
        #et = et.drop(index='WORK_OF_ART')
        ey = et.sort_values(by=['total'], ascending=False)
    
    else:
        #positive
        db = database[database['sentiment']=='pos']
        doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
        
        # entitiy
        position = [j.pos_ for j in doc if len(j.text) > 3]
    
        # five most common tokens
        position_freq = Counter(position)
        # common_words = ent_freq.most_common(5)
    
        ner_pos = pd.DataFrame.from_dict(position_freq, orient='index').reset_index()
        ner_pos.columns = ['pos', 'pos_frequency']
        ner_pos.index = ner_pos['pos']
        del ner_pos['pos']
        ner_pos = ner_pos.sort_values(by=['pos_frequency'], ascending=False)
        
        #negative
        db = database[database['sentiment']=='neg']
        doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
        
        # entitiy
        position = [j.pos_ for j in doc if len(j.text) > 3]
    
        # five most common tokens
        position_freq = Counter(position)
        # common_words = ent_freq.most_common(5)
    
        ner_neg = pd.DataFrame.from_dict(position_freq, orient='index').reset_index()
        ner_neg.columns = ['pos', 'neg_frequency']
        ner_neg.index = ner_neg['pos']
        del ner_neg['pos']
        ner_neg = ner_neg.sort_values(by=['neg_frequency'], ascending=False)
                    
        et = pd.concat([ner_pos,ner_neg], axis=1)
        et = et.fillna(0)
        et['total'] = et['pos_frequency'] + et['neg_frequency']
        ey = et.sort_values(by=['total'], ascending=False)  
    
    return ey, ner_pos, ner_neg
        

#%% COUNTRY FREQUENCY
def comprehension(a, b):
    return [x for x in a if x in b]

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def country_frequency(element,date_from,date_to):
    
    nlp = spacy.load('en')
    nlp.max_length = 5000000
    
    database = sql_element(element,date_from,date_to)
    countries = pd.read_csv('countries.csv', sep=',')
    countries['name'] = countries['name'].str.lower()
    
    country_list = countries['name'].str.lower().tolist()
    
    #Positive
    pdb = database[database['sentiment']=='pos']
    pdoc = (' '.join(pdb['combined'].str.lower().tolist())) 
    pdoc = pdoc.split()
    
    pcountry_freq = []
    
    for i in pdoc:
        #print(i)
        if i in country_list:
            pcountry_freq.append(i)
            
    pcountry_freq = wordListToFreqDict(pcountry_freq)
    pcountry_freq = sortFreqDict(pcountry_freq)

    pcountry_freq = pd.DataFrame(pcountry_freq)
    pcountry_freq.columns = ['pos_frequency', 'country']
    pcountry_freq = pcountry_freq[['country', 'pos_frequency']]
    pcountry_freq.index =  pcountry_freq['country']
    del pcountry_freq['country']
    
    #Negative
    ndb = database[database['sentiment']=='neg']
    ndoc = (' '.join(ndb['combined'].str.lower().tolist())) 
    ndoc = ndoc.split()
    
    ncountry_freq = []
    
    for i in ndoc:
        #print(i)
        if i in country_list:
            ncountry_freq.append(i)
            
    ncountry_freq = wordListToFreqDict(ncountry_freq)
    ncountry_freq = sortFreqDict(ncountry_freq)

    ncountry_freq = pd.DataFrame(ncountry_freq)
    ncountry_freq.columns = ['neg_frequency', 'country']
    ncountry_freq = ncountry_freq[['country', 'neg_frequency']]
    ncountry_freq.index =  ncountry_freq['country']
    del ncountry_freq['country']
    
    #Total
    country_freq = pd.concat([pcountry_freq,ncountry_freq], axis=1)
    country_freq = country_freq.fillna(0)
    country_freq['total'] = country_freq['pos_frequency'] + country_freq['neg_frequency']
    country_freq = country_freq.sort_values(by=['total'], ascending=False)  

    # ISO code append
    country_freq = country_freq.merge(countries, left_on=country_freq.index, right_on='name')
    
    return country_freq

#%% DASH TESTING
    
##### DASH TEST
import dash
import plotly.figure_factory as ff
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table
from datetime import datetime

df = pd.DataFrame()


app = dash.Dash()

app.layout = html.Div([
    
    html.Div([
        
        ##### BLUE
        html.Div([
            html.H1('Data Visualisation Dashboard')
            ],style={#'background-color': 'rgb(0, 0, 0)',
                  'body':'0',
                  'margin': '1%',
                  'padding': '1',
                  'box-sizing':'border-box',
                  'font-family': 'Arial',
                  'color':'#0c1f38'
                  #box-shadow':'0px 1px 0px #b8b8b8',
                  #'border-radius': '5px',
                  # 'border-style': 'solid',
                  # 'border-width': '10px',
                  # 'border-color': 'rgb(200, 200, 200)',
                  }),
        
        ##### PINK
        html.Div([
            
            #
            html.Div([
                html.H4('Commodity')
                ],style={
                 'width': '5%',
                 'text-align': 'center',
                 'float':'left',
                 'margin':'1%',
                 'display': 'inline-block'}),
            html.Div([      
                dcc.Dropdown(
                            id='my_commodity_symbol',
                            options=[
                                {'label': 'Gold', 'value': 'gold'},
                                {'label': 'Silver', 'value': 'silver'},
                                {'label': 'Platinum', 'value': 'platinum'},
                                {'label': 'Palladium', 'value': 'palladium'},
                                {'label': 'Copper', 'value': 'copper'},
                                {'label': 'Iron', 'value': 'iron'}],
                            value='gold',
                            multi=False)
                ],style={
                 'width': '10%',
                 'height':'100%',
                 'float':'left',
                 'margin':'2%',
                 'display': 'inline-block'}),
            #
            html.Div([
                html.H4('Date Range')
                ],style={
                 'width': '5%',
                 'text-align': 'center',
                 'float':'left',
                 'margin':'1%',
                 'display': 'inline-block'}),
                html.Div([      
                    dcc.DatePickerRange(id='my_date_picker',
                                       min_date_allowed= str(datetime(2000,1,1).date()),
                                       max_date_allowed= str(datetime.today().date()),
                                       start_date=str(datetime(2019,1,1).date()),
                                       end_date = str(datetime.today().date())
                                       )
                ],style={
                 'width': '23%',
                 'height':'100%',
                 'float':'left',
                 'margin':'2%',
                 'display': 'inline-block'}),
            #
            html.Div([
                html.H4('Rank By')
                ],style={
                 'width': '5%',
                 'text-align': 'center',
                 'float':'left',
                 'margin':'1%',
                 'display': 'inline-block'}),
            html.Div([      
                dcc.Dropdown(
                            id='my_rank',
                            options=[
                                {'label': 'Total', 'value': 'Total'},
                                {'label': 'Positive', 'value': 'Positive'},
                                {'label': 'Negative', 'value': 'Negative'}],
                            value='Total',
                            multi=False)
                ],style={
                 'width': '10%',
                 'height':'100%',
                 'float':'left',
                 'margin':'2%',
                 'display': 'inline-block'}), 
            #
            html.Div([
                html.H4('Display Results')
                ],style={
                 'width': '5%',
                 'text-align': 'center',
                 'float':'left',
                 'margin':'1%',
                 'display': 'inline-block'}),
            html.Div([      
               dcc.Input(
                    id='display_number',
                    type='number',
                    value=25,
                    min=1,
                    max=100,
                    style={'width': '50%'})
                ],style={
                 'width': '10%',
                 'height':'100%',
                 'float':'left',
                 'margin':'3%',
                 'display': 'inline-block'}),  
                    
            
            #
            ],style={'background-color': 'rgb(254, 254, 254)',
                     'body':'0',
                      'margin-left': '2%',
                      'margin-bottom': '2%',
                      'padding': '0.25%',
                     'box-sizing':'border-box',
                     'font-family': 'Arial',
                     'height': '100px',
                     'width': '100%',
                     'padding-right': '0%',
                     'box-shadow':'1px 1px 1px #b8b8b8',
                     'border-radius': '5px',
                     'display': 'inline-block',
                     'float':'right'}),
                
                
                
                
        ##### GREEN             
        html.Div([
        #
        html.Div([
                dcc.Graph(
                    id='my_bar',
                    figure={}
                )
            ],style={'margin': '0%',
                     'padding':'1%',
                     'width':'99%',
                     #'height':'99%',
                     'display':'inline-block'}),
        
        #
        html.Div([
        html.Div([
               html.H4('Position')
               ],style={
                'width': '5%',
                'text-align': 'center',
                'float':'left',
                'margin':'2%',
                'display': 'inline-block'}),
           html.Div([      
               dcc.Dropdown(
                           id='my_position',
                           options=[
                               {'label': 'Noun', 'value': 'NOUN'},
                               {'label': 'Adjective', 'value': 'ADJ'},
                               {'label': 'Adverb', 'value': 'ADV'},
                               {'label': 'Verb', 'value': 'VERB'},
                               {'label': 'Bigram', 'value': 'Bigram'}],
                           value='NOUN',
                           multi=False)
               ],style={
                'width': '15%',
                #'height':'100%',
                'float':'left',
                'margin':'3%',
                'display': 'inline-block'}),
                
                ],style={#'height': '30px',
                         #'width': '100%',
                        'margin-top':'-8%',
                         'padding':'3%'})
                    
        # END TEST 1 GREEN
        ],style={'background-color': 'rgb(254, 254, 254)',
              'margin': '0%',
              'padding': '1',
             'box-sizing':'border-box',
             'font-family': 'Arial',
             'height': '500px',
             'width': '65%',
             'box-shadow':'1px 1px 1px #b8b8b8',
             'border-radius': '5px',
             'display': 'inline-block',
             'float':'left'}),
                 
        # START GREEN 2
                 
        html.Div([
        #
        html.Div([
                dcc.Graph(
                    id='my_bar2',
                    figure={}
                )
            ],style={'margin': '0%',
                     'width':'99%',
                     'padding':'0.5%',
                     #'height':'99%',
                     'display':'inline-block'}),
        #
        html.Div([
        #
        html.Div([
           html.H4('Type')
           ],style={
            'width': '10%',
            'text-align': 'center',
            'float':'left',
            'margin':'3%',
            'display': 'inline-block'}),
        html.Div([      
            dcc.Dropdown(
                        id='my_nlp',
                        options=[
                            {'label': 'Position', 'value': 'POS'},
                            {'label': 'Entity', 'value': 'ENT'}],
                        value='ENT',
                        multi=False)
            ],style={
             'width': '30%',
             #'height':'100%',
             'float':'left',
             'margin':'5%',
             'display': 'inline-block'}),
            #    
            ],style={#'height': '30px',
             #'width': '100%',
            'margin-top':'-10%',
             'padding':'3%'})
         
        ],style={'background-color': 'rgb(254, 254, 254)',
                     'body':'0',
                      'margin': '0%',
                      'padding': '0.25%',
                     'box-sizing':'border-box',
                     'font-family': 'Arial',
                     'height': '500px',
                     'width': '33%',
                     'box-shadow':'1px 1px 1px #b8b8b8',
                     'border-radius': '5px',
                     'display': 'inline-block',
                     'float':'right'}),
                 
        # END GREEN
                 
  ##### RED             
    html.Div([
        #
        html.Div([
                dcc.Graph(
                    id='my_map',
                    figure={}
                )
            ],style={'margin': '0%',
                     'padding':'1%',
                     'width':'99%',
                     #'height':'99%',
                     'display':'inline-block'}),
        
        #
                    
        # END TEST 1 RED
        ],style={'background-color': 'rgb(254, 254, 254)',
            'margin-top':'2%',
              #'margin': '0%',
              'padding': '1',
             'box-sizing':'border-box',
             'font-family': 'Arial',
             'height': '500px',
             'width': '65%',
             'box-shadow':'1px 1px 1px #b8b8b8',
             'border-radius': '5px',
             'display': 'inline-block',
             'float':'left'}),
                 
        # START RED 2
                 
        html.Div([
        #
        html.Div([
                dcc.Graph(
                    id='my_bar3',
                    figure={}
                )
            ],style={'margin': '0%',
                     'width':'99%',
                     'padding':'0.5%',
                     #'height':'99%',
                     'display':'inline-block'}),
        #
        html.Div([
        #
        html.Div([
           html.H4('Type')
           ],style={
            'width': '10%',
            'text-align': 'center',
            'float':'left',
            'margin':'3%',
            'display': 'inline-block'}),
        html.Div([      
            dcc.Dropdown(
                        id='my_ent',
                        options=[
                            {'label': 'Organisation', 'value': 'ORG'},
                            {'label': 'Location', 'value': 'GPE'},
                            {'label': 'Date', 'value': 'DATE'},
                            {'label': 'Person', 'value': 'PERSON'}],
                        value='ORG',
                        multi=False)
            ],style={
             'width': '40%',
             #'height':'100%',
             'float':'left',
             'margin':'5%',
             'display': 'inline-block'}),
            #    
            ],style={#'height': '30px',
             #'width': '100%',
            'margin-top':'-10%',
             'padding':'3%'})
         
        ],style={'background-color': 'rgb(254, 254, 254)',
                 'margin-top':'2%',
                     'body':'0',
                      #'margin': '0%',
                      'padding': '0.25%',
                     'box-sizing':'border-box',
                     'font-family': 'Arial',
                     'height': '500px',
                     'width': '33%',
                     'box-shadow':'1px 1px 1px #b8b8b8',
                     'border-radius': '5px',
                     'display': 'inline-block',
                     'float':'right'}),
                 
        # END RED
        
     ])
                 
 ],style={'background-color': 'rgb(245, 245, 245)',
          'height': '1700px',
          'body':'0',
          'position': "absolute",
          'top': '0',
          'right': '0',
          'bottom': '0',
          'left': '0',
          'margin': '0',
          'padding-top': '0.5%',
          'padding-right': '5%',
          'padding-left': '5%',
          'padding-bottom': '0.5%',})
          
         
                


##### CALLBACKS
@app.callback(
    Output('my_bar', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('my_rank','value'),
    Input('my_position','value'),
    Input('display_number','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph(commodity_ticker,my_rank,position,display_number,start_date,end_date):

    
    if position == 'Bigram':
        tt = ngram_function(commodity_ticker,start_date,end_date)
        
        if my_rank == 'Total':
            bar = tt.iloc[:display_number].sort_values(by=['total'], ascending=False)
        
            trace2 = go.Bar(x=bar.index,
                            y=bar['pos_frequency'],
                            name='Positive',
                            marker={'color':'green'})
        
            trace3 = go.Bar(x=bar.index,
                            y=bar['neg_frequency'],
                            name='Negative',
                            marker={'color':'red'})
        
            data = [trace2,trace3]
            layout = go.Layout(title=f'Word Frequency split by Sentiment: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
    
        elif my_rank == 'Positive':
            bar = tt.iloc[:display_number].sort_values(by=['pos_frequency'], ascending=False)
        
            trace2 = go.Bar(x=bar.index,
                            y=bar['pos_frequency'],
                            name='Positive',
                            marker={'color':'green'})
        
            data = [trace2]
            layout = go.Layout(title=f'Positive Word Frequency: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
    
        elif my_rank == 'Negative':
            bar = tt.iloc[:display_number].sort_values(by=['neg_frequency'], ascending=False)
    
            trace3 = go.Bar(x=bar.index,
                            y=bar['neg_frequency'],
                            name='Negative',
                            marker={'color':'red'})
        
            data = [trace3]
            layout = go.Layout(title=f'Negative Word Frequency: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
        
        else:
            pass
        
        
        
    else:
        tt, ps, ng = nlp_function(commodity_ticker,position,start_date,end_date)
        
        if my_rank == 'Total':
            bar = tt.iloc[:display_number].sort_values(by=['total'], ascending=False)
        
            trace2 = go.Bar(x=bar.index,
                            y=bar['pos_frequency'],
                            name='Positive',
                            marker={'color':'green'})
        
            trace3 = go.Bar(x=bar.index,
                            y=bar['neg_frequency'],
                            name='Negative',
                            marker={'color':'red'})
        
            data = [trace2,trace3]
            layout = go.Layout(title=f'Word Frequency split by Sentiment: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
    
        elif my_rank == 'Positive':
            bar = ps.iloc[:display_number].sort_values(by=['pos_frequency'], ascending=False)
        
            trace2 = go.Bar(x=bar.index,
                            y=bar['pos_frequency'],
                            name='Positive',
                            marker={'color':'green'})
        
            data = [trace2]
            layout = go.Layout(title=f'Positive Word Frequency: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
    
        elif my_rank == 'Negative':
            bar = ng.iloc[:display_number].sort_values(by=['neg_frequency'], ascending=False)
    
            trace3 = go.Bar(x=bar.index,
                            y=bar['neg_frequency'],
                            name='Negative',
                            marker={'color':'red'})
        
            data = [trace3]
            layout = go.Layout(title=f'Negative Word Frequency: {commodity_ticker.title()}',
                                barmode='stack',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(title='Frequency'),
                                xaxis=dict(title='Word'))
            fig = go.Figure(data=data,layout=layout)
        
        else:
            pass
            
    return fig

#
    
@app.callback(
    Output('my_bar2', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('my_rank','value'),
    Input('my_nlp','value'),
    Input('display_number','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph(commodity_ticker,my_rank,my_nlp,display_number,start_date,end_date):
    
    ep, ner_pos, ner_neg = ner_function(commodity_ticker,my_nlp,start_date,end_date)
    
    if my_rank == 'Total':   
        et = ep.iloc[:display_number].sort_values(by=['total'], ascending=False)
    
        trace2 = go.Bar(y=et.index,
                        x=et['pos_frequency'],
                        name='Positive',
                        orientation='h',
                        marker={'color':'green'})
    
        trace3 = go.Bar(y=et.index,
                        x=et['neg_frequency'],
                        name='Negative',
                        orientation='h',
                        marker={'color':'red'})
    
        data = [trace2,trace3]
        layout = go.Layout(title=f'Word Frequency split by {my_nlp}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)
    
    elif my_rank == 'Positive':  
        ner_pos = ner_pos.iloc[:display_number].sort_values(by=['pos_frequency'], ascending=False)
    
        trace2 = go.Bar(y=ner_pos.index,
                        x=ner_pos['pos_frequency'],
                        name='Positive',
                        orientation='h',
                        marker={'color':'green'})
    
        data = [trace2]
        layout = go.Layout(title=f'Word Frequency split by {my_nlp}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)
        
    elif my_rank == 'Negative':
        ner_neg = ner_neg.iloc[:display_number].sort_values(by=['neg_frequency'], ascending=False)

        trace3 = go.Bar(y=ner_neg.index,
                        x=ner_neg['neg_frequency'],
                        name='Negative',
                        orientation='h',
                        marker={'color':'red'})
    
        data = [trace3]
        layout = go.Layout(title=f'Word Frequency split by {my_nlp}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)

    
    return fig

@app.callback(
    Output('my_bar3', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('my_rank','value'),
    Input('my_ent','value'),
    Input('display_number','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph(commodity_ticker,my_rank,my_ent,display_number,start_date,end_date):
    
    ep = ner_words(commodity_ticker,my_ent,start_date,end_date)
    
    if my_rank == 'Total':   
        et = ep.iloc[:display_number].sort_values(by=['total'], ascending=False)
    
        trace2 = go.Bar(y=et.index,
                        x=et['pos_frequency'],
                        name='Positive',
                        orientation='h',
                        marker={'color':'green'})
    
        trace3 = go.Bar(y=et.index,
                        x=et['neg_frequency'],
                        name='Negative',
                        orientation='h',
                        marker={'color':'red'})
    
        data = [trace2,trace3]
        layout = go.Layout(title=f'Word Frequency split by {my_ent}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)
    
    elif my_rank == 'Positive':  
        et = ep.iloc[:display_number].sort_values(by=['pos_frequency'], ascending=False)
    
        trace2 = go.Bar(y=et.index,
                        x=et['pos_frequency'],
                        name='Positive',
                        orientation='h',
                        marker={'color':'green'})
    
        data = [trace2]
        layout = go.Layout(title=f'Word Frequency split by {my_ent}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)
        
    elif my_rank == 'Negative':
        et = ep.iloc[:display_number].sort_values(by=['neg_frequency'], ascending=False)

        trace3 = go.Bar(y=et.index,
                        x=et['neg_frequency'],
                        name='Negative',
                        orientation='h',
                        marker={'color':'red'})
    
        data = [trace3]
        layout = go.Layout(title=f'Word Frequency split by {my_ent}: {commodity_ticker.title()}',
                            barmode='stack',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title='Frequency'),
                            yaxis=dict(title='Word',autorange="reversed"))
        fig = go.Figure(data=data,layout=layout)

    
    return fig

# CHLOROPLETH
@app.callback(
    Output('my_map', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('my_rank','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_map(commodity_ticker,my_rank,start_date,end_date):
    df = country_frequency(commodity_ticker,start_date,end_date)
    
    if my_rank == 'Total':
        fig = go.Figure(data=go.Choropleth(
            locations = df['alpha-3'],
            z = df['total'],
            text = df['name'],
            colorscale = 'Blues',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            #colorbar_tickprefix = f'{my_rank}',
            colorbar_title = f'{my_rank} Frequency',
        ))
        
        fig.update_layout(
            title_text=f'Country Overview for {commodity_ticker} by {my_rank}',
            geo=dict(
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
        )
    elif my_rank =='Positive':
        fig = go.Figure(data=go.Choropleth(
            locations = df['alpha-3'],
            z = df['pos_frequency'],
            text = df['name'],
            colorscale = 'Greens',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            #colorbar_tickprefix = f'{my_rank}',
            colorbar_title = f'{my_rank} Frequency',
        ))
        
        fig.update_layout(
            title_text=f'Country Overview for {commodity_ticker} by {my_rank}',
            geo=dict(
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
        )
    elif my_rank == 'Negative':
        
        fig = go.Figure(data=go.Choropleth(
            locations = df['alpha-3'],
            z = df['neg_frequency'],
            text = df['name'],
            colorscale = 'Reds',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            #colorbar_tickprefix = f'{my_rank}',
            colorbar_title = f'{my_rank} Frequency',
        ))
        
        fig.update_layout(
            title_text=f'Country Overview for {commodity_ticker} by {my_rank}',
            geo=dict(
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
        )
    else:
        pass
        
    return fig
    
##### SERVER
if __name__ == '__main__':
    app.run_server()
    

#%% TESTING
# element='gold'
# date_from = '2018-01-01'
# date_to = '2020-01-01'
# et = ner_words(element,date_from,date_to)