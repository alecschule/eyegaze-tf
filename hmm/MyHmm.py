import numpy as np
import sys
import csv

class hmm:
    def __init__(self, len_hist, model_frames, annotated_frames,\
            looking_accuracy, not_looking_accuracy):
        self.len_hist = len_hist
        self.mask = (2 ** len_hist) - 1 # bit mask of form 000..00111
        self.annotated_prob = self.frame_prob(annotated_frames)
        self.model_prob = self.frame_prob(model_frames)
        # model accuracy
        self.accuracy = [not_looking_accuracy, looking_accuracy]

    # returns a list of probabilities of size 2^n
    def frame_prob(self, frames): 
        frame_counts = np.zeros(2**self.len_hist)

        frame_hist = 0
        frame_num = 0
        for frame in frames:
            frame_hist = frame_hist << 1
            frame_hist += frame
            frame_num += 1
            if frame_num >= self.len_hist:
                index = frame_hist & self.mask # last n bits only
                frame_counts[index] += 1

        num_frames = np.sum(frame_counts)
        frame_counts /= num_frames
        return frame_counts

    # returns the probability that the n frames passed to HMM are all correct
    def get_probability(self, frame_history):
        # frame_history is an int where the last n bits are the last n predictions
        frame_history = frame_history & self.mask;
        # model_correct_prob is the probability that model was correct on previous frames
        model_correct_prob = 1
        for i in range(self.len_hist):
            frame_prediction = (frame_history >> i) & 0x01
            model_correct_prob *= self.accuracy[frame_prediction]
        if self.model_prob[frame_history] == 0:
            print ("No data on this combination of frames")
            return None
        return model_correct_prob * self.annotated_prob[frame_history] / self.model_prob[frame_history]
