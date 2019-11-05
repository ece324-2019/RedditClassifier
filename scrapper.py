# https://api.pushshift.io/reddit/search/submission/?subreddit=learnpython&sort=desc&sort_type=created_utc&before=1523934121&size=1000
import requests
import pandas as pd

subreddit_list = ["nfl", "AskReddit", "worldnews", "politics", "fantasyfootball", "AskMen", "interestingasfuck", "showerthoughts",
              "freefolk", "explainlikeimfive", "AmITheAsshole", "nba", "The_Donald", "teenagers", "tifu", "LifeProTips,"
              "dataisbeautiful", "unpopularopinion", "movies", "legaladvice", "wallstreetbets"]
base = "https://api.pushshift.io/reddit/search/submission/?subreddit="
#before=1523934121&size=1000
i = 0
sub_data = []
earliest_utc = 1000000000000000
while i < len(subreddit_list):
    subreddit = subreddit_list[i]
    if earliest_utc == 1000000000000000:
        trail = "&sort=desc&sort_type=created_utc&size=1000&score=>100"
        r = requests.get(base + subreddit + trail)
    else:
        trail = "&before=" + str(earliest_utc) + "&sort=desc&sort_type=created_utc&size=1000&score=>100"
        r = requests.get(base + subreddit + trail)
    posts = r.json()

    print(r.status_code)
    j = 0
    while j < len(posts['data']):
        if j == 0:
            print(earliest_utc)
        earliest_utc = min(earliest_utc, int(posts['data'][j]['created_utc']))
        data_point = [posts['data'][j]['title']]
        if 'url' in posts['data'][j]:
            data_point.append(posts['data'][j]['url'])
        else:
            data_point.append("")

        if 'selftext' in posts['data'][j]:
            data_point.append(posts['data'][j]['selftext'][0:300].replace("\n", ""))
        else:
            data_point.append("")
        if len(posts['data'][j]['title']) > 15:
            sub_data.append(data_point)
        j += 1
    if len(sub_data) >= 5000:
        df = pd.DataFrame(sub_data[0:5000])
        df.to_csv(str(i) + "_raw_data.csv")
        sub_data = []
        earliest_utc = 1000000000000000
        i += 1

# implement pager --> 
# save raw data

