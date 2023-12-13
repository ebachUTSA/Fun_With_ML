from src.utils import insertCraigslistData, getAlreadyCollected, getRegions

alreadyCollected = getAlreadyCollected()
regions = getRegions()
boards = ['ccc','jjj','ggg','bbb','rrr']

for region in regions:
    print('Working',region)
    for board in boards:
        print('\tWorking',board)
        region_url = f"https://{region}.craigslist.org/search/{board}"
        insertCraigslistData(region_url,alreadyCollected=alreadyCollected)