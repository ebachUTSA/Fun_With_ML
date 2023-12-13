from src.utils import downloadYoutube
from src.config import Config
config = Config()

downLoadDir = f'{config.base_directory}/output/'
fName = 'UnderPressure'
url = 'https://www.youtube.com/watch?v=a01QQZyl-_I'
#comment
downloadYoutube(url,f"{downLoadDir}{fName}")