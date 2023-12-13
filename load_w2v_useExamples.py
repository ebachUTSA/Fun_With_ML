from src.utils import wordEquation, similarWords, leastSimilar, wordDistance
from src.config import Config
config = Config()



for word in ("isnt",'New York','New_York'):
    print(word,wordDistance(word,'banana'))
# print('ls',leastSimilar(['Car','Truck','Tractor','Banana']))
# print('sim',similarWords(['Affirmative','Yes','Understood'],topn=3,positive=True))
# print('we',wordEquation(['Cyber','Intrusion','Detection','Physical'],['Attack']))