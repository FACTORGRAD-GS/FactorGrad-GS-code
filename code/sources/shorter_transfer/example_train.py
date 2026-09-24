from argparse import ArgumentParser, Namespace
import random
import numpy as np
import torch
import sys

import litegs
import litegs.config
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_default_arg()
    fp_cdo=litegs.arguments.FactorGradParams.get_class_default_obj()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    litegs.arguments.FactorGradParams.add_cmdline_arg(fp_cdo,parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    lp=litegs.arguments.ModelParams.extract(args)
    op=litegs.arguments.OptimizationParams.extract(args)
    pp=litegs.arguments.PipelineParams.extract(args)
    dp=litegs.arguments.DensifyParams.extract(args)
    fp=litegs.arguments.FactorGradParams.extract(args)



    litegs.training.start(
        lp,
        op,
        pp,
        dp,
        args.test_epochs,
        args.save_epochs,
        args.checkpoint_epochs,
        args.start_checkpoint,
        fp,
    )
