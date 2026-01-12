from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import numpy as np

# Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

extract = URLExtract()
analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# BASIC STATS
# --------------------------------------------------
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


# --------------------------------------------------
# MOST BUSY USERS
# --------------------------------------------------
def most_busy_users(df):
    x = df['user'].value_counts().head()

    percent_df = (
        df['user'].value_counts() / df.shape[0] * 100
    ).round(2).reset_index()

    percent_df.columns = ['name', 'percent']
    return x, percent_df


# --------------------------------------------------
# WORDCLOUD
# --------------------------------------------------
def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') &
              (df['message'] != '<Media omitted>\n')]

    def remove_stop_words(message):
        return " ".join(
            word for word in message.lower().split()
            if word not in stop_words
        )

    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white'
    )

    temp = temp.copy()
    temp['message'] = temp['message'].apply(remove_stop_words)

    return wc.generate(temp['message'].str.cat(sep=" "))


# --------------------------------------------------
# MOST COMMON WORDS
# --------------------------------------------------
def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') &
              (df['message'] != '<Media omitted>\n')]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))


# --------------------------------------------------
# EMOJI ANALYSIS
# --------------------------------------------------
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    return pd.DataFrame(Counter(emojis).most_common())


# --------------------------------------------------
# TIMELINES
# --------------------------------------------------
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = (
        df.groupby(['year', 'month_num', 'month'])
          .count()['message']
          .reset_index()
    )

    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


# --------------------------------------------------
# ACTIVITY MAPS
# --------------------------------------------------
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)


# ==================================================
# 🔥 EMOTION INTENSITY MODEL (IDEA 1 – FINAL)
# ==================================================

def compute_emotion_intensity(selected_user, df):
    """
    Returns hourly emotion intensity scores
    """

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df.copy()

    # Sentiment per message
    df['sentiment'] = df['message'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )

    # Hourly bucket
    df['hour_block'] = df['date'].dt.floor('H')

    grouped = df.groupby('hour_block').agg(
        avg_sentiment=('sentiment', 'mean'),
        message_count=('message', 'count'),
        unique_users=('user', 'nunique')
    ).reset_index()

    # Sentiment change
    grouped['sentiment_delta'] = grouped['avg_sentiment'].diff()

    # Emotion intensity score
    grouped['emotion_intensity'] = (
        grouped['sentiment_delta'].abs()
        * np.log1p(grouped['message_count'])
        * grouped['unique_users']
    )

    return grouped


def detect_emotional_events(emotion_df, top_percentile=95):
    """
    Detect top emotional events based on intensity
    """

    threshold = np.percentile(
        emotion_df['emotion_intensity'].dropna(),
        top_percentile
    )

    events = emotion_df[
        emotion_df['emotion_intensity'] >= threshold
    ].copy()

    events['event_type'] = events['sentiment_delta'].apply(
        lambda x: 'Positive Surge 😊' if x > 0 else 'Negative Surge 😠'
    )

    return events[
        ['hour_block', 'emotion_intensity',
         'message_count', 'unique_users', 'event_type']
    ]
