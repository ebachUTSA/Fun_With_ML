from src.utils import insertCraigslistData, getResumeContent, getAllLinks, getCraigslistResumeData
from src.config import Config
import os 

config = Config()

with open(f"{config.base_directory}/data/active_craigslist_regions.txt","r") as f:
    regions = f.readlines()

output_file = f"{config.base_directory}/output/craigslist_resumes.csv"
for region in regions:
    region = region.strip()
    print('Working ',region)
    region_url = f"https://{region}.craigslist.org/search/rrr"
    df = getCraigslistResumeData(region_url)
    df['region'] = region
    if os.path.isfile(output_file):
        df.to_csv(output_file,'|',index=False)
    else:
        df.to_csv(output_file,'|',index=False,mode='a',header=False)

# to import the craigslist_resumes.csv use the following command:
# df = pd.read_csv(f"{config.base_directory}/output/craigslist_resumes.csv", sep='|')