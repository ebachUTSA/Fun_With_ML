from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def sliceVideo(baseDir,fname_in,fname_out,start_time,end_time):
    try:
        ffmpeg_extract_subclip(f"{baseDir}{fname_in}", start_time, end_time, targetname=f"{baseDir}{fname_out}")
        return True
    except Exception as e:
        return False