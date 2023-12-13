from src.utils import wordEquation, similarWords, leastSimilar, wordDistance
from src.config import Config
config = Config()

print('wd',wordDistance('Peanut','Almond'))
print('ls',leastSimilar(['Car','Truck','Tractor','Banana']))
print('sim',similarWords(['Affirmative','Yes','Understood'],topn=3,positive=True))
print('we',wordEquation(['Cyber','Intrusion','Detection','Physical'],['Attack']))