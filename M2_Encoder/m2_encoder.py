from typing import Union

import torch
from nn4k.executor import LLMExecutor


class M2EncoderExecutor(LLMExecutor):

    @classmethod
    def from_config(cls, nn_config: Union[str, dict]) -> "M2EncoderExecutor":
        executor = cls(nn_config)
        return executor

    def load_model(self, args=None, mode=None, **kwargs):
        from nn4k.consts import NN_DEVICE_KEY
        from nn4k.utils.config_parsing import get_string_field

        nn_config: dict = args or self.init_args
        if self._model is None:
            model_config = get_string_field(nn_config, 'model_config', '')
            device = nn_config.get(NN_DEVICE_KEY)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            from vlmo.utils.beit_utils import load_from_config
            model, processors = load_from_config(model_config)
            model.to(device).eval()
            self._model = model
            self._tokenizer, self._img_processor = processors

    def text_inference(self, texts):
        # logit_scale = self._model.logit_scale.exp()
        # print(logit_scale)
        # exit()
        txt_encoding = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self._model.hparams.config["max_text_len"],
            return_special_tokens_mask=True,
        )
        txt_data = {
            "text_ids": torch.tensor(txt_encoding["input_ids"]).to(self._device),
            "text_masks": torch.tensor(txt_encoding["attention_mask"]).to(self._device),
            "text_labels": None,
        }
        txt_feats = self._model.infer_text(txt_data)["cls_vlffn_feats"]
        return txt_feats
    
    def image_inference(self, images):
        data = {"image": [images]}
        img_feats = self._model.infer_image(data)["cls_vlffn_feats"]
        return img_feats
        

    def inference(self, data, args=None, encoding_type="text", **kwargs):
        # print(self._img_processor)
        # exit()
        if encoding_type=="text":
            return self.text_inference(data)
        elif encoding_type=="image":
            return self.image_inference(data)
        else:
            raise ValueError(f"wrong choice for inference encoding type....")
