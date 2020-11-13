import numpy as np
import subprocess as sp
import json

"""
A class to handle video using FFMPEG
"""
class VideoHandler():
    def __init__(self, ffprobe_exe, ffmpeg_exe, process_frame_interval=0):
        self.ffprobe = ffprobe_exe
        self.ffmpeg = ffmpeg_exe
        self.process_frame_interval = process_frame_interval

    def get_video_meta(self, video_file):
        """Internal method to get video meta
        :return: a list containing [audio_exit, video_exit, duration, frame_count, height, width, fps]
        """
        cmd = [self.ffprobe, '-i', video_file, '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format']
        ffprobe_output = json.loads(sp.check_output(cmd).decode('utf-8'))

        # audio_exits = False
        video_exits = False
        duration = 0
        frame_count = 0
        height = 0
        width = 0
        fps = 0
        bitrate = 0

        stream_type = 'streams'
        codec_type = 'codec_type'
        if stream_type in ffprobe_output:
            for i in range(len(ffprobe_output[stream_type])):
                if codec_type in ffprobe_output[stream_type][i]:
                    # if ffprobe_output[stream_type][i][codec_type] == 'audio':
                    #     audio_exits = True
                    if ffprobe_output[stream_type][i][codec_type] == 'video':
                        video_exits = True
                        frame_rate = ffprobe_output[stream_type][i]['avg_frame_rate']
                        if '/' in frame_rate:
                            fps_temp = [float(item) for item in frame_rate.split('/')]
                            fps = fps_temp[0] / fps_temp[1]
                        else:
                            fps = float(frame_rate)
                        if 'duration' not in ffprobe_output[stream_type][i]:
                            if 'format' in ffprobe_output:
                                duration = float(ffprobe_output['format']['duration'])
                        else:
                            duration = float(ffprobe_output[stream_type][i]['duration'])
                        frame_count = int(duration * fps)
                        height = ffprobe_output[stream_type][i]['height']
                        width = ffprobe_output[stream_type][i]['width']
                        if 'bit_rate' not in ffprobe_output[stream_type][i]:
                            if 'format' in ffprobe_output:
                                bitrate = int(ffprobe_output['format']['bit_rate'])
                        else:
                            bitrate = int(ffprobe_output[stream_type][i]['bit_rate']) / 1000

        if not video_exits:
            return None
        return [video_exits, duration, frame_count, height, width, fps, bitrate]

    def get_frames(self, video_file, convert_to_gray=False):
        """
        Get video frames in a Numpy array
        :param video_file: video path
        :param convert_to_gray: flag to convert to gray or not
        :return: frames in an array
        """
        meta = self.get_video_meta(video_file)
        video_height = meta[3]
        video_width = meta[4]
        video_size = video_height * video_width * 3
        # print('Start reading {}'.format(video_file))
        if self.process_frame_interval > 0:
            fps = 'fps=1/' + str(self.process_frame_interval)
            cmd = [self.ffmpeg, '-i', video_file, '-f', 'image2pipe', '-vf', fps, '-pix_fmt', 'rgb24', '-vcodec',
                   'rawvideo', '-']
        else:
            cmd = [self.ffmpeg, '-i', video_file, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-hide_banner', '-loglevel',
                   'panic', '-vcodec', 'rawvideo', '-']
        pipe = sp.Popen(cmd, stdout=sp.PIPE)

        images = []
        try:
            while True:
                try:
                    raw_image = pipe.stdout.read(video_size)
                    if len(raw_image) != video_size:
                        break
                    image = np.fromstring(raw_image, dtype='uint8')
                    image = image.reshape((video_height, video_width, 3))

                    if convert_to_gray:
                        image = np.array(image, dtype=np.float32)
                        image = np.dot(image, [0.2989, 0.587, 0.114])

                    images.append(image.astype(np.uint8))
                except Exception as e1:
                    print(e1)
                    continue
        except Exception as e2:
            print(e2)
        pipe.stdout.flush()

        return images
