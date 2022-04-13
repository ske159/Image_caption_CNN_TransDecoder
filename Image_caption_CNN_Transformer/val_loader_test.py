import json
import re
import os.path
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json


# set up file names and pathes

dataDir = '.'
dataType = 'val2014'
algName = 'transformer'
annFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)


@torch.no_grad()
def val_coco(model, val_image_folder, device,  result_i, transform, batch_size, val_dataset):

    hypotheses = list()  # hypotheses (predictions)

    subtypes = ['results', 'evalImgs', 'eval']
    [resFile, evalImgsFile, evalFile] = \
        ['%s/results/captions_%s_%s_%s_%d.json' %
         (dataDir, dataType, algName, subtype, result_i) for subtype in subtypes]
    # COCO official result format

    # dict_for_receive = [{
    #     "image_id": int, "caption": str,
    # }]
    # Clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())

    model.eval()

    data = pd.read_csv('./coco/val2014_imgID.txt')
    if result_i > 315:
        result_i = 0
    else:
        pass

    for i in range(result_i * batch_size, (batch_size + result_i * batch_size), 1):
        dict_for_receive = {"image_id": int(data['id'][i]), "caption": str}
        img = Image.open(os.path.join(val_image_folder, data['image'][i])).convert("RGB")
        img = transform(img).unsqueeze(0)

        dict_for_receive["caption"] = bioclean(" ".join(model.caption_image(img.to(device), val_dataset.vocab)))
        # [1:-1]
        hypotheses.append(dict_for_receive)
        print(hypotheses[i]['caption'])
    json.dump(hypotheses, open(resFile, 'w'))

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    for metric, score in cocoEval.eval.items():
        print('%s, %.3f' % (metric, score))
    # save evaluation results
    json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
    json.dump(cocoEval.eval, open(evalFile, 'w'))

    return cocoEval.eval['Bleu_4']

    model.train()
