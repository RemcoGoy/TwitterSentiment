import asyncio
import datetime
import os
import re
from collections import Counter, deque

import dash
import dash_core_components as dcc
import dash_html_components as html
import nltk
import numpy as np
import plotly
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from data_gather.api import get_tweet_data
from data_gather.streaming import keywords_to_hear, run_twitter_stream

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)

sid = SentimentIntensityAnalyzer()

stops = stopwords.words('english')
stops.append('https')

app = dash.Dash(__name__, meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app_color = {
    "graph_bg": "rgb(201, 201, 201)",
    "graph_line": "rgb(8, 70, 151)",
    "graph_font": "rgb(41, 41, 41)"
}

chart_colors = [
    '#ef553b',  # Red
    '#00cc96',  # Green
    '#ab63fa',  # Purple
    '#7fdeff',
    '#ffa15a',  # Orange
    '#ff6692',  # Pink
    '#19d3f3',  # Light blue
]

num_tags_scatter = 5
scatter_dict = {}
sentiment_dict = {}

X_universal = deque(maxlen=30)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Interval(
                            id="query_update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                        html.Div(
                            [html.Div("Number of tweets",
                                      className="graph_title")]
                        ),
                        html.Div(
                            [
                                html.P(
                                    "Total number of tweets streamed during last 60 seconds: 0",
                                    id="bin-size",
                                    className="auto__p",
                                ),
                            ],
                            className="auto__container",
                        ),
                        dcc.Graph(
                            id="number_of_tweets",
                            animate=False,
                            figure=go.Figure(
                                layout=go.Layout(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                    ],
                    className="number_of_tweets",
                ),
                html.Div(
                    [
                        html.Div(
                            [html.Div("Sentiment score",
                                      className="graph_title")]
                        ),
                        html.Div(
                            [
                                html.P(
                                    "Positive: >= 0.05",
                                    className="auto__p",
                                ),
                                html.P(
                                    "Neutral: < 0.05 and > -0.05",
                                    className="auto__p",
                                ),
                                html.P(
                                    "Negative: <= -0.05",
                                    className="auto__p",
                                ),
                            ],
                            className="auto__container",
                        ),
                        dcc.Graph(
                            id="sentiment_scores",
                            figure=go.Figure(
                                layout=go.Layout(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                    ],
                    className="sentiment_graph",
                ),
            ],
            className="app__content",
        )
    ],
    className="app__container",
)


def hashtag_counter(series):
    cnt = {keyword: 0 for keyword in keywords_to_hear}
    for row in series:
        for keyword in keywords_to_hear:
            if keyword.lower() in row.lower():
                cnt[keyword] += 1
    return cnt


def preprocess_nltk(row):
    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(row.lower()) if word.isalpha()]

    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]

    # remove all stopwords
    no_stop = [word for word in tokens if word not in stops]

    return ' '.join(no_stop)


@app.callback(
    Output('number_of_tweets', 'figure'),
    [Input('query_update', 'n_intervals')])
def update_graph_scatter(n):
    df = get_tweet_data()
    cnt = Counter(df['keyword'])

    # get the current time for x-axis
    time = datetime.datetime.now().strftime('%D, %H:%M:%S')
    X_universal.append(time)

    to_pop = []
    for keyword, cnt_queue in scatter_dict.items():
        if cnt_queue:
            while cnt_queue and (cnt_queue[0][1] < X_universal[0]):
                cnt_queue.popleft()
        else:
            to_pop.append(keyword)

    for keyword in to_pop:
        scatter_dict.pop(keyword)

    top_N = cnt.most_common(num_tags_scatter)

    for keyword, cnt in top_N:
        if keyword not in scatter_dict:
            scatter_dict[keyword] = deque(maxlen=30)
            scatter_dict[keyword].append([cnt, time])
        else:
            scatter_dict[keyword].append([cnt, time])

    new_colors = chart_colors[:len(scatter_dict)]

    data = [go.Scatter(
        x=[time for cnt, time in cnt_queue],
        y=[cnt for cnt, time in cnt_queue],
        name=keyword,
        mode='lines+markers',
        opacity=1,
        marker=dict(
            size=10,
            color=color,
        ),
        line=dict(
            width=6,
            # dash='dash',
            color=color,
        ),
        textfont=dict(
            color=color
        )
    ) for color, (keyword, cnt_queue) in list(zip(new_colors, scatter_dict.items()))]

    # specify the layout
    layout = go.Layout(
        xaxis={
            'automargin': False,
            'range': [min(X_universal), max(X_universal)],
            'title': 'Current Time (GMT)',
            'nticks': 6,
            'color': '#292929',
            'gridcolor': '#adadad'
        },
        yaxis={
            'type': 'log',
            'autorange': True,
            'title': 'Number of Tweets',
            'color': '#292929',
            'gridcolor': '#adadad'
        },
        height=700,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": app_color["graph_font"]},
        autosize=False,
        legend={
            'orientation': 'h',
            'xanchor': 'center',
            'yanchor': 'top',
            'x': 0.5,
            'y': 1.025
        },
        margin=go.layout.Margin(
            l=75,
            r=25,
            b=45,
            t=25,
            pad=4
        ),
    )

    return go.Figure(
        data=data,
        layout=layout,
    )


@app.callback(
    Output('sentiment_scores', 'figure'),
    [Input('query_update', 'n_intervals')])
def update_graph_sentiment(interval):
    df = get_tweet_data()
    cnt = Counter(df['keyword'])

    top_N = cnt.most_common(num_tags_scatter)
    top_N_words = [keyword for keyword, cnt in top_N]

    df['text'] = df.text.apply(preprocess_nltk)

    sentiments = {keyword: [] for keyword in top_N_words}
    for row in df['text']:
        for keyword in top_N_words:
            if keyword.lower() in row.lower():
                sentiments[keyword].append(
                    sid.polarity_scores(row)['compound'])

    avg_sentiments = {}
    for keyword, score_list in sentiments.items():
        avg_sentiments[keyword] = [np.mean(score_list), np.std(score_list)]

    time = datetime.datetime.now().strftime('%D, %H:%M:%S')
    X_universal.append(time)

    to_pop = []
    for keyword, score_queue in sentiment_dict.items():
        if score_queue:
            while score_queue and (score_queue[0][1] <= X_universal[0]):
                score_queue.popleft()
        else:
            to_pop.append(keyword)

    for keyword in to_pop:
        sentiment_dict.pop(keyword)

    for keyword, score in avg_sentiments.items():
        if keyword not in sentiment_dict:
            sentiment_dict[keyword] = deque(maxlen=30)
            sentiment_dict[keyword].append([score, time])
        else:
            sentiment_dict[keyword].append([score, time])

    new_colors = chart_colors[:len(sentiment_dict)]

    # plot the scatter plot
    data = [go.Scatter(
        x=[time for score, time in score_queue],
        y=[score[0] for score, time in score_queue],
        name=keyword,
        mode='lines+markers',
        opacity=1,
        marker=dict(color=color)
    ) for color, (keyword, score_queue) in list(zip(new_colors, sentiment_dict.items()))]

    # specify the layout
    layout = go.Layout(
        xaxis={
            'automargin': False,
            'range': [min(X_universal), max(X_universal)],
            'title': 'Current Time (GMT)',
            'nticks': 2,
            'color': '#292929',
            'gridcolor': '#adadad'
        },
        yaxis={
            'autorange': True,
            'title': 'Sentiment Score',
            'color': '#292929',
            'gridcolor': '#adadad'
        },
        height=400,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": app_color["graph_font"]},
        autosize=False,
        legend={
            'orientation': 'v',
            # 'xanchor': 'right',
            # 'yanchor': 'middle',
            # 'x': 0.5,
            # 'y': 1.025
        },
        margin=go.layout.Margin(
            l=75,
            r=25,
            b=70,
            t=25,
            pad=4
        ),
    )

    return go.Figure(
        data=data,
        layout=layout,
    )


@app.callback(
    Output("bin-size", "children"),
    [Input("query_update", "n_intervals")],
)
def show_num_bins(slider_value):
    df = get_tweet_data()
    total_tweets = len(df)

    return "Total number of tweets streamed during last 60 seconds: " + str(int(total_tweets))


if __name__ == '__main__':
    asyncio.run(run_twitter_stream())
    app.run_server(debug=False, host='0.0.0.0', port=80)
