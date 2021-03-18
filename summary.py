import torch
import time
import numpy as np
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from models.builder import SummarizerLayer


def preprocess(sourcefile):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    with open(sourcefile) as source:
        rawtext = source.read().replace("\n", " ").replace("[CLS] [SEP]", " ")
    sents = sent_tokenize(rawtext)
    processedtext = "[CLS] [SEP]".join(sents)
    return processedtext, len(sents)



def loadtext(processedtext, maxpos, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sepid = tokenizer.vocab["[SEP]"]
    clsid = tokenizer.vocab["[CLS]"]

    def processsrc(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        srcsubtokens = tokenizer.tokenize(raw)
        srcsubtokens = ["[CLS]"] + srcsubtokens + ["[SEP]"]
        srcsubtokenids = tokenizer.convert_tokens_to_ids(srcsubtokens)
        srcsubtokenids = srcsubtokenids[:-1][:maxpos]
        srcsubtokenids[-1] = sepid
        segs1 = [-1] + [i for i, t in enumerate(srcsubtokenids) if t == sepid]
        segs = [segs1[i] - segs1[i - 1] for i in range(1, len(segs1))]
        
        segmentids = []
        segs = segs[:maxpos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segmentids += s * [0]
            else:
                segmentids += s * [1]

        src = torch.tensor(srcsubtokenids)[None, :].to(device)
        srcmask = (1 - (src == 0).float()).to(device)
        clsids = [[i for i, t in enumerate(srcsubtokenids) if t == clsid]]
        clss = torch.tensor(clsids).to(device)
        clsmask = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, srcmask, segmentids, clss, clsmask

    src, srcmask, segmentids, clss, clsmask = processsrc(processedtext)
    segs = torch.tensor(segmentids)[None, :].to(device)
    srctext = [[sent.replace("[SEP]", "").strip() for sent in processedtext.split("[CLS]")]]
    return src, srcmask, segs, clss, clsmask, srctext



def process(model, inputdata, resultpath, maxlength, blocktrigram=True):
    def getngrams(n, text):
        ngram = set()
        textlen = len(text)
        maxngramstart = textlen - n
        for i in range(maxngramstart + 1):
            ngram.add(tuple(text[i : i + n]))
        return ngram

    def blocktri(c, p):
        tric = getngrams(3, c.split())
        for s in p:
            tri = getngrams(3, s.split())
            if len(tric.intersection(tri)) > 0:
                return True
        return False

    with open(resultpath, "w") as savepred:
        with torch.no_grad():
            src, mask, segs, clss, clsmask, source = inputdata
            sent_scores, mask = model(src, segs, clss, mask, clsmask)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selectedids = np.argsort(-sent_scores, 1)

            pred = []
            for i, ids in enumerate(selectedids):
                pred1 = []
                if len(source[i]) == 0:
                    continue
                for j in selectedids[i][: len(source[i])]:
                    if j >= len(source[i]):
                        continue
                    cand = source[i][j].strip()
                    if blocktrigram:
                        if not blocktri(cand, pred1):
                            pred1.append(cand)
                    else:
                        pred1.append(cand)

                    if len(pred1) == maxlength:
                        break

                pred1 = " ".join(pred1)
                pred.append(pred1)

            for i in range(len(pred)):
                savepred.write(pred[i].strip() + "\n")


def summarize(srctext, resultpath, model, maxlength=3, maxpos=512, returnsummary=True):
    model.eval()
    processedtext, fullen = preprocess(srctext)
    inputdata = loadtext(processedtext, maxpos, device="cpu")
    process(model, inputdata, resultpath, maxlength, blocktrigram=True)
    if returnsummary:
        return open(resultpath).read().strip()
        