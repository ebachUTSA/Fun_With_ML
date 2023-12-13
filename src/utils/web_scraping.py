import requests
import time
from bs4 import BeautifulSoup as bs
from src.config import Config
config = Config()

def checkReqLimit(mw_tf,max_reqs):
    currtime = time.time()
    newreqList = []
    for req in config.request_list:
        if currtime-req <= mw_tf:
            newreqList.append(req)
    config.request_list = newreqList
    return max_reqs-len(config.request_list)

def getPageSource(url,mw_tf=3600,max_reqs=400,justSource=True,userAgent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'):
    reqs_allowed = checkReqLimit(mw_tf,max_reqs)
    if reqs_allowed <= 0:
        sleeptime = (config.request_list[abs(reqs_allowed)]+mw_tf+5) - time.time()
        print("Needing to sleep for",sleeptime,"seconds.")
        print("Because we are",1-reqs_allowed,"over our limit!")
        time.sleep(sleeptime)
    headers = {'User-Agent':userAgent}
    r = requests.get(url,headers=headers)
    config.request_list.append(time.time())
    if justSource:
        results = r.text
    else:
        results = r
    return results

def getSoup(content,parser='html.parser'):
    soup = bs(content,parser)
    return soup