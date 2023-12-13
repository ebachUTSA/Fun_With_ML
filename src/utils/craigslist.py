import pandas as pd
from src.utils import getPageSource, getSoup, insertSQLPandas, selectSQLPandas
from src.config import Config
config = Config()

def getRegions():
    with open(f"{config.base_directory}/data/regions.txt",'r') as f:
        regions = f.read().split('\n')
        regions = [region.lower().strip() for region in regions]
    return regions

def getAlreadyCollected():
    df = selectSQLPandas('select distinct url from craigslist')
    urlList = df['url'].values.tolist()
    return urlList

def getPostContent(url):
    postid = url.split('/')[-1].split('.')[0]
    content = getPageSource(url=url)
    soup = getSoup(content)
    expiration = soup.find('meta',{'name':'robots'})
    if expiration is not None:
        expiration = expiration.get('content').split(' ')[-1]
    title = soup.find('span',{'id':'titletextonly'})
    if title is not None:
        title = title.text.strip()
    dtg = soup.find('time',class_='date timeago')
    if dtg is not None:
        dtg = dtg.text.strip() #preferred right now
        # dtg = soup.find('time',{'class':['date','timeago']}).get('datetime')
    body = soup.find('section',{'id':'postingbody'})
    if body is not None:
        body.find('div').decompose()
        body = body.text.strip()
    geo_el = soup.find('div',{'id':'map'})
    if geo_el is not None:
        lat = geo_el.get('data-latitude')
        long = geo_el.get('data-longitude')
        gmaps_url = soup.find('p',{'class':'mapaddress'})
        if gmaps_url is not None:
            gmaps_url = gmaps_url.find('a').get('href')
        else:
            gmaps_url = 'https://www.google.com/maps/search/'+str(lat)+','+str(long)
    else:
        lat = None
        long = None
        gmaps_url = None
    return (title,body,dtg,lat,long,gmaps_url,postid,expiration)

def getAllLinks(url,alreadyCollected):
    url = url + '?sort=dateoldest'
    p1='#search=1~list~'
    p2= '~0'
    pcount = 0
    results = []
    oldurl = ''
    run = True
    while run:
        url = url+f"{p1}{pcount}{p2}"
        content = getPageSource(url=url)
        soup = getSoup(content)
        ol = soup.find('ol')
        rowcount = 0
        for a in ol.find_all('a'):
            subUrl = a.get('href')
            if rowcount == 0:
                if subUrl == oldurl:
                    run = False
                    break
                else:
                    oldurl = subUrl
            if subUrl not in (alreadyCollected):
                results.append(subUrl)
            rowcount += 1
        pcount += 1
    return results

def insertCraigslistData(region_url,alreadyCollected=[]):
    post_urls = getAllLinks(region_url,alreadyCollected)
    data = {'url':[]
            ,'title':[]
            ,'body':[]
            ,'dtg':[]
            ,'lat':[]
            ,'long':[]
            ,'gmaps_url':[]
            ,'postid':[]
            ,'expiration':[]}
    for url in post_urls:
        title,body,dtg,lat,long,gmaps_url,postid,expiration = getPostContent(url)
        data['url'].append(url)
        data['title'].append(title)
        data['body'].append(body)
        data['dtg'].append(dtg)
        data['lat'].append(lat)
        data['long'].append(long)
        data['gmaps_url'].append(gmaps_url)
        data['postid'].append(postid)
        data['expiration'].append(expiration)

    df = pd.DataFrame.from_dict(data)
    insertSQLPandas(df,'craigslist')