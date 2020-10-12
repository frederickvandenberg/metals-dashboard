#%%
##### REQUIRMENTS
import spacy
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
#%%
##### SENTIMENT ML 
def sentiment(dataframe):
    df = pd.read_csv('mining_headlines_500.csv', sep=',')
    
    X = df['headline']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    clf = LinearSVC()
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', clf),
    ])
    
    # Feed the training data through the pipeline
    text_clf.fit(X_train, y_train)  
    
    # Form a prediction set
    predictions = text_clf.predict(X_test)
    
    #Report the confusion matrix
    from sklearn import metrics
    
    #print(metrics.confusion_matrix(y_test,predictions))
    
    # You can make the confusion matrix less confusing by adding labels:
    dfc = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['neg','pos'], columns=['neg','pos'])
    print(dfc)
    
    # Print a classification report
    print(metrics.classification_report(y_test,predictions))
    
    # Print the overall accuracy
    print(metrics.accuracy_score(y_test,predictions))
    
    # Returns databased
    predictions = text_clf.predict(dataframe['combined'])

    return predictions

#%%
con = tb.db_con(config_dict)
con.force_close()
#%%
##### SINGLE ELEMENT NLP QUERY
def sql_element(element,date_from,date_to):
    con = tb.db_con(config_dict)

    # Headlines
    dfw = pd.DataFrame(con.read_query(f"""select pub_date, heading, sub_heading
                                        from articles
                                        where (unique_text like '%{element}%' AND pub_date BETWEEN '{date_from}' AND '{date_to}')
                                        order by pub_date desc;"""),
                                        columns=['pub_date','heading','sub_heading'])

    dfw['combined'] = dfw['heading'] + '. ' + dfw['sub_heading']

    del dfw['heading']
    del dfw['sub_heading']

    # Sentiment Predictions
    predictions = sentiment(dfw)
    #predictions = text_clf.predict(dfw['combined'])

    # Date format
    prediction_results = pd.DataFrame(predictions)
    dfw['sentiment'] = prediction_results
    dfw['pub_date'] = pd.to_datetime(dfw['pub_date'])

    # close connection
    con.force_close()
    
    return dfw

#%%
##### SINGLE ELEMENT PRICE-FREQ-SENT QUERY
    
def sql_price(element,date_from,date_to):
    
        con = tb.db_con(config_dict)
    
        # Headlines
        df = pd.DataFrame(con.read_query(f"""select pub_date, heading, sub_heading
                                            from articles
                                            where (unique_text like '%{element}%' AND pub_date BETWEEN '{date_from}' AND '{date_to}')
                                            order by pub_date desc;"""),
                                            columns=['pub_date','heading','sub_heading'])
    
        df['combined'] = df['heading'] + '. ' + df['sub_heading']
    
        del df['heading']
        del df['sub_heading']
    
        # Price
        pf = pd.DataFrame(con.read_query(f"""select spot_date, am_price
                                            from metal_price
                                            where (commodity like '%{element}%' AND spot_date BETWEEN '{date_from}' AND '{date_to}')
                                            order by spot_date desc;"""),
                                            columns=['spot_date','am_price'])
    
        # Sentiment Predictions
        predictions = sentiment(df)
    
        prediction_numerics = []
    
        for j in predictions:
            if j == 'neg':
                prediction_numerics.append(-1)
            else:
                prediction_numerics.append(1)
    
        # Date format
        prediction_results = pd.DataFrame(predictions)
        prediction_numerics = pd.DataFrame(prediction_numerics)
        df['sentiment'] = prediction_results
        df['num_sentiment'] = prediction_numerics
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        pf['spot_date'] = pd.to_datetime(pf['spot_date'])
    
        df.index = df['pub_date']
        df.index = pd.to_datetime(df.index)
        pf.index = pf['spot_date']
        pf.index = pd.to_datetime(pf.index)
    
        #### Time period W or M switch
        time_period = 'M'
    
        dfs = df['num_sentiment'].resample(time_period).mean().rename('Sentiment')
        dff = df['num_sentiment'].resample(time_period).count().rename('Frequency')
        pfp = pf['am_price'].resample(time_period).mean().rename('Price')
    
        # Assign element name
        element_name_list = []
    
        for e in dff.index:
            element_name_list.append(element)
    
        ef = pd.DataFrame(data=element_name_list, index = dfs.index, columns=['element'])
    
        # Concat dataframes
        df1 = pd.concat([dfs,dff,pfp,ef], axis=1)
        df1['date'] = df1.index
        df1['year'], df1['month'] = df1['date'].dt.year, df1['date'].dt.month
        #from statsmodels.tsa.holtwinters import ExponentialSmoothing
        # df1['Freq_TES_mul_12'] = ExponentialSmoothing(df1['Frequency'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
        # df1['Sent_TES_add_12'] = ExponentialSmoothing(df1['Sentiment'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
        # df1['Price_TES_mul_12'] = ExponentialSmoothing(df1['Price'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
        
        # df1['Freq_SMA_12']= df1['Frequency'].rolling(window=12).mean()
        # df1['Sent_SMA_12']= df1['Sentiment'].rolling(window=12).mean()
        # df1['Price_SMA_12']= df1['Price'].rolling(window=12).mean()
        
        # close connection
        con.force_close()
        return df1

#%%
##### NLP TOKEN LEMMA ENT SPACY
def nlp_function(element,date_from,date_to):
    nlp = spacy.load('en')
    nlp.max_length = 5000000

    option = ['pos','neg']
    database = sql_element(element,date_from,date_to)
    
    for i in option:

        wd = database[database['sentiment']==i]

        doc = nlp(' '.join(wd['combined'].str.lower().tolist()))
        # all tokens that arent stop words or punctuations
        words = [token.text for token in doc if token.is_stop != True and token.is_punct != True and len(token.text) > 3]

        # noun tokens that arent stop words or punctuations
        nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]

        # five most common tokens
        word_freq = Counter(words)
        #common_words = word_freq.most_common(5)

        # five most common noun tokens
        noun_freq = Counter(nouns)
        #common_nouns = noun_freq.most_common(5)

        wd = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
        wd.columns = ['word', i+'_'+'frequency']
        wd.index = wd['word']
        del wd['word']
        globals()[i] = wd.sort_values(by=[i+'_'+'frequency'], ascending=False)

    bar = pd.concat([pos, neg], axis=1)
    bar= bar.fillna(0)
    bar['total'] = bar['pos_frequency'] + bar['neg_frequency']
    return bar

#%%
#### NER FUNCTION
def ner_function(element,date_from,date_to):
    nlp = spacy.load('en')
    nlp.max_length = 5000000
    
    option = ['pos','neg']
    database = sql_element(element,date_from,date_to)
    
    #NER
    for i in option: 
        
            db = database[database['sentiment']==i]
            doc = nlp(' '.join(db['combined'].str.lower().tolist())) 
            
            # entitiy
            ents = [j.label_ for j in doc.ents if len(j.text) > 3]
    
    
            # five most common tokens
            ent_freq = Counter(ents)
            common_words = ent_freq.most_common(5)
    
            et = pd.DataFrame.from_dict(ent_freq, orient='index').reset_index()
            et.columns = ['ent', i+'_'+'frequency']
            et.index = et['ent']
            del et['ent']
            globals()[i] = et.sort_values(by=[i+'_'+'frequency'], ascending=False)
            
                
    et = pd.concat([pos, neg], axis=1)
    et = et.fillna(0)
    et['total'] = et['pos_frequency'] + et['neg_frequency']
    et = et.sort_values(by=['total'], ascending=False)
    
    return et

#%%
##### WORD POSITION
def pos_function(element,date_from,date_to):
    nlp = spacy.load('en')
    nlp.max_length = 5000000

    option = ['pos','neg']
    database = sql_element(element,date_from,date_to)
    
    #NER
    for i in option: 
        db = database[database['sentiment']==i]
        doc = nlp(' '.join(db['combined'].str.lower().tolist())) 

        # entitiy
        position = [j.pos_ for j in doc if len(j.text) > 3]


        # five most common tokens
        position_freq = Counter(position)
        #common_words = pos_freq.most_common(5)

        ps = pd.DataFrame.from_dict(position_freq, orient='index').reset_index()
        ps.columns = ['position', i+'_'+'frequency']
        ps.index = ps['position']
        del ps['position']
        globals()[i] = pd.DataFrame(ps)
                         
    ps = pd.concat([pd.DataFrame(pos), pd.DataFrame(neg)], axis=1)
    ps = ps.fillna(0)
    ps['total'] = ps['pos_frequency'] + ps['neg_frequency']
    ps = ps.sort_values(by=['total'], ascending=False)
    
    return ps
#%%
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

##### LAYOUT, INPUTS & OUTPUTS
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
        #html.H1('Gauge 1')
            #html.Div([
                dcc.Graph(
                id='bullet',
                figure={
                    "data": [],
                    "layout": {
                        "height": 50,  # px
                                },
                        },
                    )
                  
           # ],style={'margin': '2%'})
        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'float':'left',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'width': '48%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
        html.Div([
        #html.H1('Gauge 2')
           # html.Div([
                dcc.Graph(
                id='bullet2',
                figure={
                    "data": [],
                    "layout": {
                        "height": 50,  # px
                                },
                        },
                    )
                  
           # ],style={'margin': '2%'})
        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'float':'right',
              #'height': '25px',
              'width': '48%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
        html.Div([],style={'height': '10px'}), 
        
        ##### GREEN             
        html.Div([
            
        #html.H1('Test 1'),
        #
        html.Div([
                dcc.Graph(
                    id='my_graph',
                    figure={}
                )
            ],style={'margin': '1%'})
                    
        # END TEST 1 GREEN
        ],style={'background-color': 'rgb(254, 254, 254)',
              'margin': '1%',
              'padding': '1',
             'box-sizing':'border-box',
             'font-family': 'Arial',
             'height': '600px',
             'width': '64%',
             'box-shadow':'1px 1px 1px #b8b8b8',
             'border-radius': '5px',
             'display': 'inline-block',
             'float':'left'}),
                 
        #
                 
        html.Div([
            #
            html.Div([
                html.H3('Required Parameters')
                ],style={
                     'width': '100%',
                     #'height': '80px',
                     'background-color': 'rgb(254, 254, 254)',
                     'text-align': 'left',
                     'float':'left',
                     'margin':'0%',
                     'padding':'0%',
                     'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H4('Commodity')
                    ],style={
                     'width': '30%',
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
                     'width': '60%',
                     'height':'100%',
                     'float':'left',
                     'margin':'4%',
                     'display': 'inline-block'})
                    
                    ]),
            #            
            html.Div([
                html.Div([
                    html.H4('Date Range')
                    ],style={
                     'width': '30%',
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
                     'width': '60%',
                     'height':'100%',
                     'float':'left',
                     'margin':'4%',
                     'display': 'inline-block'})
                    
                    ]),
            #
            html.Div([
                html.Div([
                    html.H4('Y1')
                    ],style={
                     'width': '10%',
                     'text-align': 'center',
                     'float':'left',
                     'margin':'1%',
                     'display': 'inline-block'}),
                html.Div([      
                        dcc.Dropdown(
                            id='plot_type',
                            options=[
                                {'label': 'Frequency', 'value': 'Frequency'},
                                {'label': 'Price', 'value': 'Price'},
                                {'label': 'Sentiment', 'value': 'Sentiment'}],
                            value='Price',
                            multi=False
                        )
                    ],style={
                     'width': '30%',
                     'height':'100%',
                     'float':'left',
                     'margin':'4%',
                     'display': 'inline-block'})
                    
                    ]),
                #
                html.Div([
                    html.Div([
                        html.H4('Y2')
                        ],style={
                         'width': '10%',
                         'text-align': 'center',
                         'float':'left',
                         'margin':'1%',
                         'display': 'inline-block'}),
                    html.Div([      
                        dcc.Dropdown(
                            id='plot_type2',
                            options=[
                                {'label': 'Frequency', 'value': 'Frequency'},
                                {'label': 'Price', 'value': 'Price'},
                                {'label': 'Sentiment', 'value': 'Sentiment'}],
                            value='Frequency',
                            multi=False
                        )
                    ],style={
                     'width': '30%',
                     'height':'100%',
                     'float':'left',
                     'margin':'4%',
                     'display': 'inline-block'})
                    
                    ]),
                    
                    #
                html.Div([
                html.H3('Additional Parameters')
                ],style={
                     'width': '100%',
                     #'height': '80px',
                     'background-color': 'rgb(254, 254, 254)',
                     'text-align': 'left',
                     'float':'left',
                     'margin':'0%',
                     'padding':'0%',
                     'display': 'inline-block'}),
                #
                html.Div([
                    html.Div([
                        html.H4('Moving Average')
                        ],style={
                         'width': '30%',
                         'text-align': 'center',
                         'float':'left',
                         'margin':'1%',
                         'display': 'inline-block'}),
                    html.Div([      
                            dcc.Dropdown(
                                        id='moving_average',
                                        options=[
                                            {'label': 'None', 'value': 'None'},
                                            {'label': 'SMA', 'value': 'SMA'}],
                                        value='None',
                                        multi=False)
                        ],style={
                         'width': '60%',
                         'height':'100%',
                         'float':'left',
                         'margin':'4%',
                         'display': 'inline-block'}),
                #
                html.Div([
                    html.Div([
                        html.H4('Window')
                        ],style={
                         'width': '10%',
                         'text-align': 'center',
                         'float':'left',
                         'margin':'1%',
                         'display': 'inline-block'}),
                    html.Div([      
                             dcc.Input(
                                id='windows',
                                type='number',
                                value=0,
                                style={'width': '30%'})
                        ],style={
                         'width': '30%',
                         #'height':'100%',
                         'float':'left',
                         'margin':'4%',
                         'display': 'inline-block'})
                    
                    ]),
                #
                html.Div([
                    html.Div([
                        html.H4('Y2 Lag')
                        ],style={
                         'width': '10%',
                         'text-align': 'center',
                         'float':'left',
                         'margin':'1%',
                         'display': 'inline-block'}),
                    html.Div([      
                       dcc.Input(
                                id='lag',
                                type='number',
                                value=0,
                                style={'width': '30%'})
                    ],style={
                     'width': '30%',
                     #'height':'100%',
                     'float':'left',
                     'margin':'4%',
                     'display': 'inline-block'})
                    
                    ]),
                #
              
        # END MAIN DIV (GREEN)              
        ]) 
                        
                             
        ],style={'background-color': 'rgb(254, 254, 254)',
                     'body':'0',
                      'margin': '1%',
                      'padding': '0.25%',
                     'box-sizing':'border-box',
                     'font-family': 'Arial',
                     'height': '600px',
                     'width': '32%',
                     'box-shadow':'1px 1px 1px #b8b8b8',
                     'border-radius': '5px',
                     'display': 'inline-block',
                     'float':'right'}),
                 
        # END GREEN

                
        # html.Div([],style={'height': '1000px'}),         
        
        ##### RED
        html.Div([
            #html.H1('Test 3'),
                html.Div([
                dcc.Graph(
                    id='my_scatter',
                    figure={}
                )
            ],style={'margin': '1%'}),
            #
            ],style={'background-color': 'rgb(254, 254, 254)',
                  'body':'0',
                  'margin': '1%',
                  'padding': '1',
                  'box-sizing':'border-box',
                  'font-family': 'Arial',
                  'height': '475px',
                  'width': '31.25%',
                  'box-shadow':'1px 1px 1px #b8b8b8',
                  'border-radius': '5px',
                  'display': 'inline-block'}),
                     
        html.Div([
        #html.H1('Test 4')
            html.Div([
                dcc.Graph(
                    id='my_hist',
                    figure={}
                )
            ],style={'margin': '1%'})    
        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'height': '475px',
              'width': '31.25%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),

        html.Div([
       # html.H1('Test 5')
            html.Div([
                dcc.Graph(
                    id='my_hist2',
                    figure={}
                )
            ],style={'margin': '1%'})
        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'height': '475px',
              'width': '31.25%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
        html.Div([],style={'height': '10px'}),   
        
        ##### YELLOW
        html.Div([
        #html.H1('Test 6')
        dash_table.DataTable(id='table1',
                            columns=[{"name":'Statistic', "id": 'statistic'},
                                     {"name": 'Value', "id": 'value'}],
                            data=df.to_dict("rows"),
                            page_action='none',
                            fixed_rows={'headers': True},
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            style_cell={'textAlign': 'left','padding': '5px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            },
                            style_data={'whiteSpace': 'normal'},
                            style_header= {'whiteSpace':'normal'},
                            css=[{
                                    'selector': '.dash-cell div.dash-cell-value',
                                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                }]
                            )
        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'height': '300px',
              'width': '31.25%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
 
        html.Div([
        #html.H1('Test 7')
        dash_table.DataTable(id='table2',
                            columns=[{"name":'Statistic', "id": 'statistic'},
                                     {"name": 'Value', "id": 'value'}],
                            data=df.to_dict("rows"),
                            page_action='none',
                            fixed_rows={'headers': True},
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            style_cell={'textAlign': 'left','padding': '5px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            },
                            style_data={'whiteSpace': 'normal'},
                            style_header= {'whiteSpace':'normal'},
                            css=[{
                                    'selector': '.dash-cell div.dash-cell-value',
                                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                }]
        )

        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'height': '300px',
              'width': '31.25%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
        html.Div([
        #html.H1('Test 8')
        dash_table.DataTable(id='table3',
                            columns=[{"name":'Statistic', "id": 'statistic'},
                                     {"name": 'Value', "id": 'value'}],
                            data=df.to_dict("rows"),
                            page_action='none',
                            fixed_rows={'headers': True},
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            style_cell={'textAlign': 'left','padding': '5px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            },
                            style_data={'whiteSpace': 'normal'},
                            style_header= {'whiteSpace':'normal'},
                            css=[{
                                    'selector': '.dash-cell div.dash-cell-value',
                                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                }]
                            )

        ],style={'background-color': 'rgb(254, 254, 254)',
              'body':'0',
              'margin': '1%',
              'padding': '1',
              'padding': '1',
              'box-sizing':'border-box',
              'font-family': 'Arial',
              'height': '300px',
              'width': '31.25%',
              'box-shadow':'1px 1px 1px #b8b8b8',
              'border-radius': '5px',
              'display': 'inline-block'}),
                 
                 
                 
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
                 
##### CALL BACK FUNCTIONS
        
@app.callback(
    Output('my_graph', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('moving_average','value'),
    Input('windows','value'),
    Input('lag','value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph(commodity_ticker, moving_average, windows, lag, plot_type, plot_type2, start_date, end_date):
    #
    windows = int(windows)
    lag=int(lag)
    title_name =  commodity_ticker.title()
    
    
    if moving_average == 'None':
        if lag > 0:
            df = sql_price(commodity_ticker,str(datetime(2000,1,1).date()),str(datetime.today().date()))
            df[plot_type2] = df[plot_type2].shift(lag)
            df = df.loc[start_date:end_date]
        else:
            df = sql_price(commodity_ticker,start_date,end_date)
        
        # TRACES
        trace1 = go.Scatter(x=df.index,
                            y= df[plot_type],
                            name=plot_type,
                            mode='lines',
                            yaxis='y1',
                            line=dict(color="red"))
        trace2 = go.Scatter(x=df.index,
                            y= df[plot_type2],
                            name=plot_type2,
                            mode='lines',
                            yaxis='y2',
                            line=dict(color="blue"))
        
        data = [trace1, trace2]
        
        layout = go.Layout(title=(f'Time Series Plot: {title_name}'),
                           xaxis = dict(
                                        #rangeslider = {'visible': True},
                                        title='Date'
                                        ),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           height=550,
                           yaxis=dict(title=plot_type),
                           yaxis2=dict(title=plot_type2,
                                       overlaying='y',
                                       side='right'))
        
        fig = {'data': data,
            'layout': layout}
    #
    elif moving_average == 'SMA':
        df = sql_price(commodity_ticker,str(datetime(2000,1,1).date()),str(datetime.today().date()))
        df[plot_type2] = df[plot_type2].shift(lag)
        df['Frequency']= df['Frequency'].rolling(window=windows).mean()
        df['Sentiment']= df['Sentiment'].rolling(window=windows).mean()
        df['Price']= df['Price'].rolling(window=windows).mean()
        df = df.loc[start_date:end_date]

    
        # TRACES
        trace1 = go.Scatter(x=df.index,
                            y= df[plot_type],
                            name=plot_type,
                            mode='lines',
                            yaxis='y1',
                            line=dict(color="red"))
        trace2 = go.Scatter(x=df.index,
                            y= df[plot_type2],
                            name=plot_type2,
                            mode='lines',
                            yaxis='y2',
                            line=dict(color="blue"))
        
        data = [trace1, trace2]
        
        layout = go.Layout(title=(f'Time Series Plot: {title_name}'),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            #autosize=False,
                            #width=400,
                            height=550,
                            yaxis=dict(title=plot_type),
                            xaxis = dict(
                                        #rangeslider = {'visible': True},
                                        title='Date'
                                        ),
                            yaxis2=dict(title=plot_type2,
                                        overlaying='y',
                                        side='right'))
        
        fig = {
            'data': data,
            'layout': layout
        }    
    else:
        pass
    
    return fig

#
@app.callback(
    Output('my_scatter', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('lag','value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph2(commodity_ticker,lag,plot_type,plot_type2,start_date,end_date):

    if lag > 0: #LAG
        df1 = sql_price(commodity_ticker,str(datetime(2000,1,1).date()),str(datetime.today().date()))
        df1 = df1.shift(lag)
        df1 = df1.loc[start_date:end_date]

        # TRACES ERROR WHEN USING BUBBLE CHART, DOESNT SEEM TO UPDATE WHEN CHANGING COMMODITIES
        #df1.drop(['element', 'date','Freq_SMA_12','Sent_SMA_12','Price_SMA_12'], axis=1, inplace=True)
        
        trace1 = go.Scatter(x=df1[plot_type2],
                            y=df1[plot_type],
                            mode='markers',
                            marker=dict(size=5,
                            color= df1['year'],
                            colorscale='rdbu',
                            showscale=True
                            )
                            )
        
        layout2 = go.Layout(title=(f'Scatter Matrix Plot for {commodity_ticker}'),
                            xaxis=dict(title=(f'{plot_type2}')),
                            yaxis=dict(title=(f'{plot_type}')),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')
        
        data2 = [trace1]
        
        fig2 = {
            'data': data2,
            'layout': layout2
                }
    else: #NO LAG
        df1 = sql_price(commodity_ticker,start_date,end_date)

        # TRACES ERROR WHEN USING BUBBLE CHART, DOESNT SEEM TO UPDATE WHEN CHANGING COMMODITIES
        #df1.drop(['element', 'date','Freq_SMA_12','Sent_SMA_12','Price_SMA_12'], axis=1, inplace=True)
        
        trace1 = go.Scatter(x=df1[plot_type2],
                            y=df1[plot_type],
                            mode='markers',
                            marker=dict(size=5,
                            color= df1['year'],
                            colorscale='rdbu',
                            showscale=True
                            )
                            )
        
        layout2 = go.Layout(title=(f'Scatter Matrix Plot for {commodity_ticker}'),
                            xaxis=dict(title=(f'{plot_type2}')),
                            yaxis=dict(title=(f'{plot_type}')),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')
        
        data2 = [trace1]        
    
    
        fig2 = {
            'data': data2,
            'layout': layout2
        }
    return fig2

#
@app.callback(
    Output('my_hist', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph3(commodity_ticker,plot_type,plot_type2,start_date,end_date):
    #INPUTS
    df1 = sql_price(commodity_ticker,start_date,end_date)

    trace1 = go.Histogram(x=df1[plot_type],
                          autobinx=True,
                          opacity=0.80,
                          marker_color='red')
    
    data2 = [trace1]
    
    layout2 = go.Layout(title=(f'Histogram for {commodity_ticker} {plot_type}'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title=(f'{plot_type} Bins')),
                        yaxis=dict(title=('Frequency')),
                        shapes= [{'line': {'color': 'black', 'dash': 'dash', 'width': 2},
                                    'type': 'line',
                                    'x0': df1[plot_type].mean(),
                                    'x1': df1[plot_type].mean(),
                                    'xref': 'x',
                                    'y0': 0,
                                    'y1': 1,
                                    'yref': 'paper'}],
                        # annotations=[
                        #     dict(
                        #         x=df1[plot_type].mean(),
                        #         y= 5,
                        #         xref="x",
                        #         yref="y",
                        #         text=("Mean = {:,.0f}").format(df1[plot_type].mean()),
                        #         showarrow=True,
                        #         arrowhead=7,
                        #         # ax = 2,
                        #         # ay = 2
                        #         )]
                            )
    fig2 = {
        'data': data2,
        'layout': layout2
    }
    return fig2

@app.callback(
    Output('my_hist2', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph3(commodity_ticker,plot_type,plot_type2,start_date,end_date):
    #INPUTS 
    
    df1 = sql_price(commodity_ticker,start_date,end_date)
    #df1 = df1[df1['element'] == commodity_ticker]
    # bin size (size=), start and end values
    trace1 = go.Histogram(x=df1[plot_type2],
                          autobinx=True,
                          opacity=0.80,
                          marker_color='blue') #xbins=dict(size=10)
    
    data2 = [trace1]
    
    layout2 = go.Layout(title=(f'Histogram for {commodity_ticker} {plot_type2}'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title=(f'{plot_type2} Bins')),
                        yaxis=dict(title=('Frequency')),
                        shapes= [{'line': {'color': 'black', 'dash': 'dash', 'width': 2},
                                    'type': 'line',
                                    'x0': df1[plot_type2].mean(),
                                    'x1': df1[plot_type2].mean(),
                                    'xref': 'x',
                                    'y0': 0,
                                    'y1': 1,
                                    'yref': 'paper'}],
                        # annotations=[
                        #     dict(
                        #         x=df1[plot_type2].mean(),
                        #         y= 10,
                        #         xref="x",
                        #         yref="y",
                        #         text=("Mean = {:,.0f}").format(df1[plot_type2].mean()),
                        #         showarrow=True,
                        #         arrowhead=7,
                        #         # ax = 2,
                        #         # ay = 2
                        #         )]
                            )
    
    fig2 = {
        'data': data2,
        'layout': layout2
    }
    return fig2




@app.callback(
    Output('bullet', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph3(commodity_ticker,plot_type,plot_type2,start_date,end_date):

    df = sql_price(commodity_ticker,start_date,end_date)
    
    fig = go.Figure(go.Indicator(
          mode = "gauge",
          gauge = {'shape': "bullet",
                   'axis': {'range': [df[plot_type].min(),df[plot_type].max()]},
                   'threshold': {
                       'line': {'color': "red", 'width': 2},
                      'thickness': 0.75,
                      'value': df[plot_type].mean()},
                       'bar': {'color': "black"},
                     'steps': [
                    {'range': [df[plot_type].min(),df[plot_type].quantile(0.25)], 'color': "lightblue"},
                    {'range': [df[plot_type].quantile(0.25), df[plot_type].quantile(0.75)], 'color': "blue"},
                    {'range': [df[plot_type].quantile(0.75), df[plot_type].max()], 'color': "darkblue"}]
                      },
          value = df[plot_type].mean(),
          domain = {'x': [0, 1], 'y': [0, 1]},
          ))
    
    fig.update_layout(height = 100, 
                       margin=dict(l=40, r=40, t=40, b=40),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       title = (f'{plot_type} (Quantile-Mean Bullet Chart)'))


    return fig

@app.callback(
    Output('bullet2', 'figure'),
    [Input('my_commodity_symbol', 'value'),
    Input('plot_type','value'),
    Input('plot_type2','value'),
    Input('my_date_picker','start_date'),
    Input('my_date_picker','end_date')])
def update_graph3(commodity_ticker,plot_type,plot_type2,start_date,end_date):

    df = sql_price(commodity_ticker,start_date,end_date)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {'shape': "bullet",
                 'axis': {'range': [df[plot_type2].min(),df[plot_type2].max()]},
                 'threshold': {
                     'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': df[plot_type2].mean()},
                     'bar': {'color': "black"},
                     'steps': [
                    {'range': [df[plot_type2].min(),df[plot_type2].quantile(0.25)], 'color': "lightblue"},
                    {'range': [df[plot_type2].quantile(0.25), df[plot_type2].quantile(0.75)], 'color': "blue"},
                    {'range': [df[plot_type2].quantile(0.75), df[plot_type2].max()], 'color': "darkblue"}]
                    },
        value = df[plot_type2].mean(),
        domain = {'x': [0, 1], 'y': [0, 1]},
        ))
    fig.update_layout(height = 100, 
                      margin=dict(l=40, r=40, t=40, b=40),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title = (f'{plot_type2} (Quantile-Mean Bullet Chart)'))
    


    return fig

#
@app.callback(
[Output('table2', 'data'),
 Output('table2', 'columns'),
 Output('table3', 'data'),
 Output('table3', 'columns'),
 Output('table1', 'data'),
 Output('table1', 'columns')],
[Input('my_commodity_symbol', 'value'),
Input('lag', 'value'),
Input('plot_type','value'),
Input('plot_type2','value'),
Input('my_date_picker','start_date'),
Input('my_date_picker','end_date')])
def update_markdown(commodity_ticker,lag,plot_type,plot_type2,start_date,end_date):
    
    df = sql_price(commodity_ticker,start_date,end_date)
    
    table2 = {'count':df[plot_type].count(),
             'sum':df[plot_type].sum(),
             'mean':df[plot_type].mean(),
             'median':df[plot_type].median(),
             'min':df[plot_type].min(),
             'q1': df[plot_type].quantile(.25),
             'q3':df[plot_type].quantile(.75),
             'max':df[plot_type].max(),
             'iqr': abs(df[plot_type].quantile(.75)-df[plot_type].quantile(.25)),
             'Stdev':df[plot_type].std(),
             'Skew':df[plot_type].skew(),
             'Kurtosis':df[plot_type].kurt()}
    
    table2 = pd.DataFrame(table2.items(), columns=[f'Statistic', f'Value'])
    table2.reset_index(drop= False)
    columns = [{'name': col, 'id': col} for col in table2.columns]
    data=table2.to_dict("rows")
    
    table3 = {'count':df[plot_type2].count(),
             'sum':df[plot_type2].sum(),
             'mean':df[plot_type2].mean(),
             'median':df[plot_type2].median(),
             'min':df[plot_type2].min(),
             'q1': df[plot_type2].quantile(.25),
             'q3':df[plot_type2].quantile(.75),
             'max':df[plot_type2].max(),
             'iqr': abs(df[plot_type2].quantile(.75)-df[plot_type].quantile(.25)),
             'Stdev':df[plot_type2].std(),
             'Skew':df[plot_type2].skew(),
             'Kurtosis':df[plot_type2].kurt()}
    
    table3 = pd.DataFrame(table3.items(), columns=[f'Statistic', f'Value'])
    table3.reset_index(drop= False)
    columns2 = [{'name': col, 'id': col} for col in table3.columns]
    data2=table3.to_dict("rows")
    
    if lag > 0:
        df1 = sql_price(commodity_ticker,str(datetime(2000,1,1).date()),str(datetime.today().date()))
        df1 = df1.dropna()
        df1 = df1.shift(lag)
        df1 = df1.loc[start_date:end_date]
    else: 
        df1 = sql_price(commodity_ticker,start_date,end_date)
        df1 = df1.dropna()

    #covariance = cov(df1[plot_type], df1[plot_type2])
    pcorr, pprob = pearsonr(df1[plot_type], df1[plot_type2])
    scorr, sprob = spearmanr(df1[plot_type], df1[plot_type2])
    
    table4 = {"Pearson's r":pcorr,
              "Pearson's p-value":pprob,
              "Spearman's r": scorr,
              "Spearman's p-value": sprob}
    

    table4 = pd.DataFrame(table4.items(), columns=[f'Statistic', f'Value'])
    table4.reset_index(drop= False)
    columns3 = [{'name': col, 'id': col} for col in table4.columns]
    data3=table4.to_dict("rows")
    

    return data, columns, data2, columns2, data3, columns3

##### SERVER
if __name__ == '__main__':
    app.run_server()
