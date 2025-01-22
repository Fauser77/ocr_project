import cv2
import typing
import numpy as np
import tensorflow as tf
from keras.models import load_model

from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class ImageToWordModel:
    def __init__(self, char_list: typing.Union[str, list], model_path: str):
        self.char_list = char_list
        self.model = load_model(model_path, compile=False)
        self.input_shape = self.model.input_shape[1:3]  # Get height, width from input shape

    def predict(self, image: np.ndarray):

        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.predict(image_pred, verbose=0)

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # Load configurations
    configs = BaseModelConfigs.load("Models/04_sentence_recognition/202501201958/configs.yaml")

    # Initialize model with .h5 path
    model = ImageToWordModel(
        model_path="Models/04_sentence_recognition/202501201958/model.h5",
        char_list=configs.vocab
    )

    # Load validation dataset
    df = pd.read_csv("Models/04_sentence_recognition/202501201958/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

        cv2.imshow(prediction_text, image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q'):
                break

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")