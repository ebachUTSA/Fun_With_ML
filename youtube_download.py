from src.utils import downloadYoutube, sliceVideo
from src.config import Config
config = Config()

downLoadDir = f'{config.base_directory}/output/'
fName = 'UnderPressure'
url = 'https://www.youtube.com/watch?v=a01QQZyl-_I'

downloadYoutube(url,f"{downLoadDir}{fName}")

sliceVideo(f"{downLoadDir}",fName+'.mp4',fName+"_slice.mp4",32,40)