from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

baseDir = 'c:/development/other/yt_downloader/'

fname_in = '/downloads/UnderPressure.mp4'

fname_out = '/downloads/UnderPressure_Clip.mp4'

start_time = 32
end_time = 60

ffmpeg_extract_subclip(f"{baseDir}{fname_in}", start_time, end_time, targetname=f"{baseDir}{fname_out}")