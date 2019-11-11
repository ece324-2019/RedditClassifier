# https://api.pushshift.io/reddit/search/submission/?subreddit=learnpython&sort=desc&sort_type=created_utc&before=1523934121&size=1000
import requests
import pandas as pd

subreddit_list = ["nfl", "AskReddit", "worldnews", "politics", "gaming", "science", "interestingasfuck", "showerthoughts",
              "freefolk", "todayilearned", "anime", "nba", "The_Donald", "teenagers", "soccer", "jokes",
              "EarthPorn", "askreddit", "movies",  "mma"]


base = "https://api.pushshift.io/reddit/search/submission/?subreddit="
#before=1523934121&size=1000
i = 19
sub_data = []
earliest_utc = 1000000000000000
threshold = 50000
while i < len(subreddit_list):
    print(subreddit_list[i])
    subreddit = subreddit_list[i]
    if earliest_utc == 1000000000000000:
        trail = "&sort=desc&sort_type=created_utc&size=1000&score=>50"
        r = requests.get(base + subreddit + trail)
    else:
        trail = "&before=" + str(earliest_utc) + "&sort=desc&sort_type=created_utc&size=1000&score=>50"
        r = requests.get(base + subreddit + trail)
    posts = r.json()

    print(r.status_code)
    if len(posts['data']) == 0:
        print(len(sub_data))
        i += 1
        sub_data = []
        earliest_utc = 1000000000000000
        continue
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
    if len(sub_data) >= threshold:
        df = pd.DataFrame(sub_data[0:threshold])
        df.to_csv(str(i) + "_raw_data.csv")
        sub_data = []
        earliest_utc = 1000000000000000
        i += 1

# implement pager --> 
# save raw data

