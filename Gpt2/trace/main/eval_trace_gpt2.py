import logging
import os
import sys
import time
import torch
sys.path.append("..")
sys.path.append("../../")
from transformers import T5Config,GPT2Config
from common.init_eval import get_eval_args, test
from common.data_loader import load_examples
from common.utils import MODEL_FNAME
from common.models import Gpt2



if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = Gpt2(GPT2Config.from_pretrained(args.gpt2,local_files_only=True), args.gpt2)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception("evaluation model not found")
    logger.info("model loaded")
    model = model.to(device)
    start_time = time.time()
    test_dir = os.path.join(args.data_dir,args.project, "test")
    test_examples = load_examples(test_dir, model=model, num_limit=args.test_num, type='test')
    test_examples.update_embd(model)
    m = test(args, model, test_examples)
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
