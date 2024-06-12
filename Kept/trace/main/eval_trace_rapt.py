import logging
import os
import sys
import time

import torch

sys.path.append("..")
sys.path.append("../../")
from transformers import BertConfig
from common.init_eval import get_eval_args, test
from common.data_loader import load_examples
from common.utils import MODEL_FNAME
from common.models import Rapt

if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model = Rapt(BertConfig(), args.code_bert)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))
        #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise Exception("evaluation model not found")
    #logger.info("model loaded device:",device.type)
    model = model.to(device)
    model.eval()
    start_time = time.time()
    test_dir = os.path.join(args.data_dir,args.project, "test")
    test_examples = load_examples(test_dir, model=model, num_limit=args.test_num,args=args,type='test')
    test_examples.update_embd(model)
    m = test(args, model, test_examples)
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
