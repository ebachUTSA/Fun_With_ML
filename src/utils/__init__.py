from src.utils.query_helper import selectSQLPandas, insertSQLPandas
from src.utils.informationFunctions import getEntropy, getInformationGain
from src.utils.web_scraping import getPageSource, getSoup
from src.utils.fed_benefits_funcs import getFedBennitsUrls, parseFedBenefitsDataTable
from src.utils.craigslist import getAllLinks, getPostContent, insertCraigslistData, getAlreadyCollected, getRegions
from src.utils.youtubDownloader import downloadYoutube
from src.utils.videoSlicer import sliceVideo
from src.utils.feature_generation import generateTextFeatures
from src.utils.word2vec import similarWords, leastSimilar, wordDistance, wordEquation, compareTexts