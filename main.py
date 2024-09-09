from utils import save_video, read_video
from tracker import Tracker


def main():
    #read video
    video_frames = read_video('input_video/3507660-uhd_3840_2160_30fps.mp4')
    
    #intialize tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl') 
    
    
    #draw annotation
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    #save video
    save_video(output_video_frames, 'output_video/output_video.avi')
    
if __name__ == '__main__':
    main()