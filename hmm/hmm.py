import numpy as np
import sys
import csv

num_frames = 7 # 2^n possibilities
mask = 0x7F

# returns a list of probabilities of size 2^n
def frame_prob(n, frames): 
    mask = 0x7F
    frame_counts = np.zeros(2**n)
    print (frame_counts)

    frame_hist = 0
    frame_num = 0
    for frame in frames:
        frame_hist = frame_hist << 1
        frame_hist += frame
        frame_num += 1
        if frame_num >= n:
            index = frame_hist & mask # last n bits only
            print ("index", index)
            frame_counts[index] += 1

    print(frame_counts)
    num_frames = np.sum(frame_counts)
    frame_counts /= num_frames
    return frame_counts

labels = []
with open(sys.argv[1]) as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        labels.append(int(row[-1]))

probs = frame_prob(7, labels)
for i,prob in enumerate(probs):
    print (i, ":", prob)

class hmm:
    def __init__(self, model_frames, annotated_frames,\
            looking_accuracy, not_looking_accuracy):
        self.annotated_prob = frame_prob(num_frames, annotated_frames)
        self.model_prob = frame_prob(num_frames, model_frames)
        # model accuracy
        self.accuracy = [not_looking_accuracy, looking_accuracy]

    def get_probability(self, frame_history):
        # frame_history is an int where the last 7 bits are the last 7 predictions
        # probability model was correct on previous frames
        model_correct_prob = 1
        for i in range(num_frames):
            frame_prediction = (frame_history >> i) & 0x01
            model_correct_prob *= self.accuracy[frame_prediction]
        if self.model_prob[frame_history] == 0:
            print ("No data on this combination of frames")
            return
        return model_correct_prob * self.annotated_prob[frame_history] / self.model_prob[frame_history]
