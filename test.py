from src.utils import insertCraigslistData

regions = ['asheville','columbus','neworleans','austin']

for region in regions:
    print('Working ',region)
    region_url = f"https://{region}.craigslist.org/search/mis"
    insertCraigslistData(region_url)