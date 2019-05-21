"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""
import cv2
import numpy as np

MIN_FEATURES = 1500
PROCESS_FIRST, PROCESS_SECOND, PROCESS_DEFAULT = (1, 2, 3)


class VisualOdometry:

    def __init__(self):
        self.current_frame = None
        self.previous_frame = None
        self.current_features = None
        self.previous_features = None
        self.current_rotation = None
        self.current_translation = None
        self.focal = 718.8560
        self.pp = (607.1928, 185.2157)
        self.stage_id = PROCESS_FIRST
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def detect_features(self, frame):
        features = self.detector.detect(frame, None)

        return np.array([x.pt for x in features], dtype=np.float32)

    def track_features(self):
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        current_features, st, err = cv2.calcOpticalFlowPyrLK(self.previous_frame,
                                                             self.current_frame,
                                                             self.previous_features, None, **lk_params)

        st = st.reshape(st.shape[0])
        previous_features = self.previous_features[st == 1]
        current_features = current_features[st == 1]

        return previous_features, current_features

    def tracking_and_pose_estimation(self):
        self.previous_features, self.current_features = self.track_features()
        E, mask = cv2.findEssentialMat(self.previous_features, self.current_features,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)

        if E is None and mask is None:
            raise Exception('Essential matrix not found')

        _, rotation, translation, mask = cv2.recoverPose(E, self.current_features, self.previous_features,
                                                         focal=self.focal, pp=self.pp)

        return rotation, translation

    def process_first(self, frame):
        self.current_features = self.detect_features(frame=frame)
        self.stage_id = PROCESS_SECOND

    def process_second(self):
        try:
            self.current_rotation, self.current_translation = self.tracking_and_pose_estimation()
            self.stage_id = PROCESS_DEFAULT
        except Exception as e:
            print('Processing second', e)
            self.stage_id = PROCESS_FIRST

    def process_default(self, frame):
        try:
            rotation, translation = self.tracking_and_pose_estimation()

            self.current_translation = self.current_translation + self.current_rotation.dot(translation)
            self.current_rotation = rotation.dot(self.current_rotation)

            if self.previous_features.shape[0] < MIN_FEATURES:
                self.current_features = self.detect_features(frame=frame)
        except Exception as e:
            print('Processing first', e)
            self.stage_id = PROCESS_FIRST

    def pipeline(self, frame):
        self.current_frame = frame

        if self.stage_id == PROCESS_FIRST:
            self.process_first(frame=frame)
        elif self.stage_id == PROCESS_SECOND:
            self.process_second()
        else:
            self.process_default(frame=frame)

        self.previous_features = self.current_features
        self.previous_frame = self.current_frame
