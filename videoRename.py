from moviepy.editor import VideoFileClip
import os

def convert_and_rename_videos(folder_path):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return
    
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv'))]
    video_files.sort()
    
    for index, filename in enumerate(video_files):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"Squat_New{index}.mp4"
        new_path = os.path.join(folder_path, new_filename)
        
        if filename.lower().endswith('.mp4'):
            os.rename(old_path, new_path)
        else:
            try:
                clip = VideoFileClip(old_path)
                clip.write_videofile(new_path, codec="libx264", audio_codec="aac")
                clip.close()
                os.remove(old_path)
                print(f"Converted & Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# Example usage
folder = "AllVids"
convert_and_rename_videos(folder)