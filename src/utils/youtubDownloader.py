import yt_dlp

def downloadYoutube(url,fName,ratelimit=5000000,myformat='best[ext=mp4]'):
    #download all videos on a channel
    if url.startswith((
        'https://www.youtube.com/c/', 
        'https://www.youtube.com/channel/', 
        'https://www.youtube.com/user/')):
        ydl_opts = {
            'ignoreerrors': True,
            'abort_on_unavailable_fragments': True,
            'format': myformat,
            'outtmpl': fName + '/Channels/%(uploader)s/%(title)s ## %(uploader)s ## %(id)s.%(ext)s',
            'ratelimit': ratelimit,
        }
    # Download all videos in a playlist
    elif url.startswith('https://www.youtube.com/playlist'):
        ydl_opts = {
            'ignoreerrors': True,
            'abort_on_unavailable_fragments': True,
            'format': myformat,
            'outtmpl': fName + '/Playlists/%(playlist_uploader)s ## %(playlist)s/%(title)s ## %(uploader)s ## %(id)s.%(ext)s',
            'ratelimit': ratelimit,
        }
    # Download single video from url
    elif url.startswith((
        'https://youtu.be/',
        'https://www.youtube.com/watch', 
        'https://www.twitch.tv/', 
        'https://clips.twitch.tv/')):
        ydl_opts = {
            'ignoreerrors': True,
            'abort_on_unavailable_fragments': True,
            'format': myformat,
            'outtmpl': fName + '.%(ext)s',
            'ratelimit': ratelimit,
        }
    # Downloads depending on the options set above
    if ydl_opts is not None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)