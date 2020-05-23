from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists
from tweepy import API, OAuthHandler, Stream
from urllib3.exceptions import ProtocolError

from data_gather.key_secret import (access_token, access_token_secret, consumer_key,
                                    consumer_secret)
from data_gather.slistener import SListener

keywords_to_hear = ['Fortnite',
                    'LeagueOfLegends',
                    'ApexLegends',
                    ]


async def run_twitter_stream():
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = API(auth)

    listen = SListener(api, keywords=keywords_to_hear)

    stream = Stream(auth, listen)

    engine = create_engine("sqlite:///tweets.sqlite")
    if not database_exists(engine.url):
        create_database(engine.url)

    stream.filter(track=keywords_to_hear, is_async=True)
