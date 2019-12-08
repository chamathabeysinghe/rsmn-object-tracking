from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Sequential, Model
from model.ROIPoolingLayer import ROIPoolingLayer

# TODO make this 2x2 for with PCA classification
pooled_height = 1
pooled_width = 1


class RandomKernelModel:

    @staticmethod
    def build_model(input_shape, rois_shape, kernel_count=64):
        input_future = Input(input_shape)
        input_rois_future = Input(rois_shape)

        input_current = Input(input_shape)
        input_rois_current = Input(rois_shape)

        cnn = Sequential()
        cnn.add(Conv2D(kernel_count, (10, 10), activation='relu', input_shape=input_shape, padding='same'))

        features_future = cnn(input_future)
        roi_features_future = ROIPoolingLayer(pooled_height, pooled_width)([features_future, input_rois_future])

        features_current = cnn(input_current)
        roi_features_current = ROIPoolingLayer(pooled_height, pooled_width)([features_current, input_rois_current])

        test_model = Model(inputs=[input_current, input_rois_current, input_future, input_rois_future],
                           outputs=[roi_features_current, roi_features_future])
        return test_model


# RandomKernelModel.build_model((240, 320, 6), (2, 4)).summary()

