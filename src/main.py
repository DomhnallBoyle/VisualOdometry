"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

import argparse
import cv2
import numpy as np

from calibration.calibration import calibrate_camera, undistort
from visual_odometry import VisualOdometry

WIDTH, HEIGHT = (1280, 720)


def main(args):
    video = cv2.VideoCapture(args.video_path)
    calibration = calibrate_camera(verbose=args.debug)
    visual_odometry = VisualOdometry()
    trajectory = np.zeros((600, 600, 3), dtype=np.uint8)

    frame_id = 1
    requires_reshape = False

    while video.isOpened():
        success, frame = video.read()

        if frame_id == 1:
            if frame.shape[:2] != (HEIGHT, WIDTH):
                requires_reshape = True

        if success:
            if requires_reshape:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = undistort(frame, calibration)
            visual_odometry.pipeline(frame=frame)

            current_translation = visual_odometry.current_translation
            if frame_id > 2:
                x, y, z = current_translation[:3]
            else:
                x, y, z = (0., 0., 0.)

            draw_x, draw_y = int(x) + 290, int(z) + 90

            cv2.circle(trajectory, (draw_x, draw_y), 1, (frame_id * 255 / 4540, 255 - frame_id * 255 / 4540,
                                                               0), 1)
            cv2.rectangle(trajectory, (10, 20), (600, 60), (0, 0, 0), -1)

            text = 'Coordinates: x=%2fm y=%2fm z=%2fm' % (x, y, z)

            cv2.putText(trajectory, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Road facing camera', frame)
            cv2.imshow('Trajectory', trajectory)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame_id += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    main(args)
