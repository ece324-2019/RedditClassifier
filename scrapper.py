# https://api.pushshift.io/reddit/search/submission/?subreddit=learnpython&sort=desc&sort_type=created_utc&before=1523934121&size=1000
import requests

subreddit_list = ["nfl", "AskReddit", "worldnews", "politics", "fantasyfootball", "dankmemes", "memes", "playboicarti",
              "freefolk", "explainlikeimfive", "AmITheAsshole", "nba", "The_Donald", "teenagers", "tifu", "LifeProTips,"
              "dataisbeautiful", "unpopularopinion", "WatchPeopleDieInside", "legaladvice"]
base = "https://api.pushshift.io/reddit/search/submission/?subreddit="
#before=1523934121&size=1000
trail = "&sort=desc&sort_type=created_utc&size=1000&score=>100"
i =0
while i < len(subreddit_list):
    subreddit = subreddit_list[i]

    r = requests.get(base + subreddit + trail)
    print(r.status_code)
    posts = r.json()
    j = 0
    while j < len(posts['data']):
        #print(posts['data'][i]['title'])
        try:
            print(posts['data'][j]['title'])
            print(posts['data'][j]['url'])
            #print(posts['data'][j]['selftext'])
            print(posts['data'][j]['score']) ## must be > 100
            #print(posts['data'][j]['selftext'])

        except:
            print(posts['data'][j])
        j += 1
    i += 1


