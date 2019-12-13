from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
import os
import glob
import threading

currentImage = NONE


class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class Main:

    def __init__(self, data_path):
        self.originals_path = os.path.join(data_path, 'originals')
        self.predictions_path = os.path.join(data_path, 'predictions')

        originals_length = len(glob.glob(os.path.join(self.originals_path, 'frame_*.png')))
        predictions_length = len(glob.glob(os.path.join(self.originals_path, 'frame_*.png')))
        assert originals_length == predictions_length

        self.video_length = originals_length
        self.current_frame_index = -1
        self.current_prediction_image = None
        self.current_original_image = None

        self.timer = RepeatTimer(1, self.play_timer)

        root = Tk()

        top_frame = Frame(root).pack(side=BOTTOM)
        bottom_frame = Frame(root).pack(side=BOTTOM)

        canvas_originals = Canvas(top_frame, width=320, height=240)
        canvas_predictions = Canvas(top_frame, width=320, height=240)
        button_next = ttk.Button(bottom_frame, text="Play next", command=self.play_next)
        button_prev = ttk.Button(bottom_frame, text="Play prev", command=self.play_prev)
        button_play = ttk.Button(bottom_frame, text="Play", command=self.play_button)
        button_stop = ttk.Button(bottom_frame, text="Stop", command=self.stop_button)

        index_label = ttk.Label(bottom_frame, text='-1')
        canvas_originals.pack(side=LEFT)
        canvas_predictions.pack(side=LEFT)
        button_next.pack(side=TOP)
        button_prev.pack(side=TOP)
        button_play.pack(side=TOP)
        button_stop.pack(side=TOP)
        index_label.pack(side=TOP)

        self.canvas_originals = canvas_originals
        self.canvas_predictions = canvas_predictions
        self.index_label = index_label
        root.mainloop()

    def play_next(self):

        if self.current_frame_index < self.video_length - 1:
            self.current_frame_index += 1

        originals_image_path = os.path.join(self.originals_path, 'frame_{}.png'.format(self.current_frame_index))
        predictions_image_path = os.path.join(self.predictions_path, 'frame_{}.png'.format(self.current_frame_index))

        self.current_original_image = ImageTk.PhotoImage(Image.open(originals_image_path))
        self.current_prediction_image = ImageTk.PhotoImage(Image.open(predictions_image_path))

        self.canvas_originals.create_image(20, 20, anchor=NW, image=self.current_original_image)
        self.canvas_predictions.create_image(20, 20, anchor=NW, image=self.current_prediction_image)

        self.index_label['text'] = str(self.current_frame_index)

    def play_prev(self):

        if self.current_frame_index > 0:
            self.current_frame_index -= 1

        originals_image_path = os.path.join(self.originals_path, 'frame_{}.png'.format(self.current_frame_index))
        predictions_image_path = os.path.join(self.predictions_path, 'frame_{}.png'.format(self.current_frame_index))

        self.current_original_image = ImageTk.PhotoImage(Image.open(originals_image_path))
        self.current_prediction_image = ImageTk.PhotoImage(Image.open(predictions_image_path))

        self.canvas_originals.create_image(20, 20, anchor=NW, image=self.current_original_image)
        self.canvas_predictions.create_image(20, 20, anchor=NW, image=self.current_prediction_image)

        self.index_label['text'] = str(self.current_frame_index)



    def play_button(self):
        self.timer.start()

    def stop_button(self):
        self.timer.cancel()

    def play_timer(self):
        if self.video_length-1 == self.current_frame_index:
            self.timer.cancel()
        else:
            self.play_next()


main = Main('../data/view')
