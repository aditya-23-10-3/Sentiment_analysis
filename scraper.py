from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_KEY = 'AIzaSyDO9G81oP2NZBkunAdFMaIZRh9J6FQXJhM'  # Replace with your YouTube Data API key

def fetch_youtube_comments(video_id):
    """
    Fetch comments from a YouTube video using the YouTube Data API.
    :param video_id: The ID of the YouTube video
    :return: A list of comments
    """
    try:
        # Initialize YouTube API client
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # Request to fetch comment threads
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        # Extract comments from the response
        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response.get("items", [])
        ]

        # Return comments or an empty list if no comments found
        return comments

    except HttpError as e:
        if e.resp.status == 404:
            raise ValueError(f"Video with ID '{video_id}' not found. Please check the video link.")
        elif e.resp.status == 403:
            raise ValueError("Access to this video is restricted. Please try a different video.")
        else:
            raise ValueError(f"An error occurred: {e}")

