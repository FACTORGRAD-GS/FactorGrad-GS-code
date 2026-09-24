#!/usr/bin/env python3
"""Evaluate frozen paper models, or replay their exact training arguments."""
import argparse
import ast
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = {'fastgs_base', 'skipgs_base', 'full'}
BASELINES = {'3dgs', 'mini', 'speedy', 'taming', 'dash'}


def json_read(path):
    return json.loads(path.read_text())


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_cfg(path):
    tree = ast.parse(path.read_text().strip(), mode='eval').body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != 'Namespace' or tree.args:
        raise ValueError('Expected a literal argparse Namespace: '+str(path))
    if any(k.arg is None for k in tree.keywords):
        raise ValueError('Unpacked expressions are not permitted in cfg_args')
    return {k.arg: ast.literal_eval(k.value) for k in tree.keywords}


def selected(r, args):
    primary = 'primary' in r['groups']
    suites = {
        'selected': r.get('visualization_selected',False),
        'table1': (primary and r['method'] in MAIN) or ('baselines' in r['groups'] and r['method'] in BASELINES),
        'main': primary and r['method'] in MAIN,
        'ablation': primary and r['method'] not in {'fastgs_base','skipgs_base'},
        'transfer': bool(set(r['groups']) & {'transfer','shorter_transfer','vanilla_full_boundary'}) or r['method']=='factorgrad_fastgs',
        'sensitivity': 'sensitivity' in r['groups'],
        'mechanism': 'mechanism' in r['groups'],
        'all': True,
    }
    return suites[args.suite] and (not args.run_id or r['id'] in args.run_id) and (not args.scenes or r['scene'] in args.scenes.split(','))


def metric_values(model):
    path = model/'results.json'
    if path.exists():
        result = json_read(path)
        if not result:
            raise RuntimeError('Empty results.json: '+str(path))
        key = 'ours_30000' if 'ours_30000' in result else sorted(result)[-1]
        row = {k.lower():float(result[key][k]) for k in ('PSNR','SSIM','LPIPS')}
    else:
        path = model/'metrics_testset.txt'
        if not path.exists():
            raise FileNotFoundError('Evaluation did not create metrics: '+str(model))
        row = {}
        for line in path.read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                if k.strip().lower() in ('psnr','ssim','lpips'):
                    row[k.strip().lower()] = float(v.strip().split()[0])
    if set(row) != {'psnr','ssim','lpips'} or not all(math.isfinite(v) for v in row.values()):
        raise ValueError('Invalid metric values: '+str(path))
    return row


def run_process(command, cwd, environment, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_write(prefix.with_suffix('.command.json'), {'argv':command, 'cwd':str(cwd)})
    start = time.monotonic()
    with prefix.with_suffix('.stdout.log').open('w') as stdout, prefix.with_suffix('.stderr.log').open('w') as stderr:
        status = subprocess.run(command, cwd=str(cwd), env=environment, stdout=stdout, stderr=stderr)
    json_write(prefix.with_suffix('.status.json'), {'returncode':status.returncode, 'process_wall_s':time.monotonic()-start})
    if status.returncode:
        raise RuntimeError('Command failed; see '+str(prefix.with_suffix('.stderr.log')))


def replace_paths(tokens, data, output):
    return [str(data) if t == '{data}' else str(output) if t == '{output}' else t for t in tokens]


def image_dir(r):
    args = r['train_args']
    for flag in ('-i','--images'):
        if flag in args:
            return args[args.index(flag)+1]
    return 'images'


def executable(r, args):
    path = args.conda_root/'envs'/r['environment']/'bin/python'
    if not path.exists():
        raise FileNotFoundError('Missing environment interpreter: '+str(path))
    return str(path)


def locate_weight(r, name, args):
    bundled = args.artifacts/r['artifact']/name
    if bundled.is_file():
        return bundled
    if args.use_local_archive:
        mapping = ROOT.parent/'local_private/original_locations.json'
        if mapping.exists():
            locations = {row['id']:Path(row['original_path']) for row in json_read(mapping)}
            original = locations.get(r['id'])
            if original is not None and (original/name).is_file():
                return original/name
    raise FileNotFoundError('Weight omitted from lean delivery: '+r['id']+'/'+name+
                            '. Use --suite selected, supply the full artifacts, or explicitly use --use-local-archive on the original machine.')


def benchmark_fps(r, args, output, data, cfg, source, python, env):
    env = dict(env, FACTORGRAD_BENCHMARK_SOURCE=str(source), FACTORGRAD_BENCHMARK_HOST=r['method'])
    common = ['-s',str(data),'-m',str(output),'-i',cfg.get('images',image_dir(r)),
              '--warmup','10','--repeats','3','--output',str(output/'fps_benchmark.json')]
    if 'shorter' in r['source']:
        command = [str(ROOT/'scripts/benchmark_shorter_fps.py')] + common + ['--sh_degree','3']
    else:
        script = source/'benchmark_fps.py'
        if not script.exists():
            script = ROOT/'scripts/benchmark_legacy_fps.py'
            env['FACTORGRAD_GAUSSIAN_MODE'] = 'taming' if r['source'].endswith('/taming') else 'legacy'
        command = [str(script)] + common + ['--iteration','30000','--quiet']
        if cfg.get('eval',True):
            command += ['--eval']
        if 'fastgs-factorgrad' in r['source'] or r['source'].endswith('/legs'):
            command += ['--mult','0.7' if r['dataset'] in ('tnt','db') else '0.5']
    run_process([python]+command, source, env, output/'fps')
    result = json_read(output/'fps_benchmark.json')
    if not math.isfinite(result['fps']) or result['fps'] <= 0:
        raise RuntimeError('FPS benchmark returned invalid data')
    return result


def evaluate(r, args):
    archived = args.artifacts/r['artifact']
    original_model = archived/r['model_subdir']
    output = args.output/r['id']
    if output.exists():
        raise FileExistsError('Refusing to mix outputs; choose a new --output: '+str(output))
    output.mkdir(parents=True)
    data = args.data_root/r['dataset']/r['scene']
    if not (data/'sparse').is_dir():
        raise FileNotFoundError('Missing COLMAP data: '+str(data))
    source = ROOT/'sources'/r['source']
    if r['source'] == 'baselines/mini':
        mini_source = os.environ.get('FACTORGRAD_MINI_SOURCE')
        if not mini_source:
            raise FileNotFoundError(
                'Mini-Splatting source is not bundled. Set FACTORGRAD_MINI_SOURCE '
                'to your separately obtained Mini-Splatting checkout.')
        source = Path(mini_source).expanduser().resolve()
        if not (source/'ms').is_dir():
            raise FileNotFoundError('Invalid FACTORGRAD_MINI_SOURCE: '+str(source))
    if not source.is_dir():
        raise FileNotFoundError('Missing source directory: '+str(source))
    python = executable(r, args)
    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TORCH_HOME'] = str(args.artifacts/'pretrained/torch')
    python_paths = [str(source), str(ROOT/'sources/factorgrad')]
    if 'fastgs-factorgrad' in r['source']:
        skipgs_source = os.environ.get('FACTORGRAD_SKIPGS_SOURCE')
        if skipgs_source:
            skipgs_source = Path(skipgs_source).expanduser().resolve()
            if not (skipgs_source/'skipgs/__init__.py').is_file():
                raise FileNotFoundError('Invalid FACTORGRAD_SKIPGS_SOURCE: '+str(skipgs_source))
            python_paths.append(str(skipgs_source))
        elif args.action == 'train':
            raise FileNotFoundError(
                'SkipGS source is not bundled. Set FACTORGRAD_SKIPGS_SOURCE '
                'to your separately obtained SkipGS checkout.')
    python_paths.append(env.get('PYTHONPATH',''))
    env['PYTHONPATH'] = os.pathsep.join(python_paths)
    if args.action == 'train':
        cwd = source/'ms' if r['source']=='baselines/mini' else source
        command = [python] + replace_paths(r['train_args'], data, output)
        run_process(command, cwd, env, output/'train')
        # Training generates new weights, never replacing the archived weights.
    else:
        for relative in r['weights']:
            weight = locate_weight(r,relative,args)
            dest = output/Path(relative).relative_to(r['model_subdir'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(weight.resolve())
    cfg_path = original_model/'cfg_args'
    cfg = load_cfg(cfg_path) if cfg_path.exists() else {}
    cfg.update(source_path=str(data), model_path=str(output))
    if cfg_path.exists():
        output.joinpath('cfg_args').write_text('Namespace('+', '.join(k+'='+repr(v) for k,v in sorted(cfg.items()))+')\n')
    if 'shorter' in r['source']:
        tokens = r['eval_args'] or ['example_metrics.py','-s','{data}','-m','{output}','-i',image_dir(r),'--sh_degree','3','--save_images']
        run_process([python]+replace_paths(tokens, data, output), source, env, output/'evaluation')
    else:
        tokens = ['render.py','--iteration','30000','-s',str(data),'-m',str(output),'-i',cfg.get('images',image_dir(r)), '--quiet','--skip_train']
        if cfg.get('eval', True):
            tokens += ['--eval']
        if 'fastgs-factorgrad' in r['source'] or r['source'].endswith('/legs'):
            tokens += ['--mult', '0.7' if r['dataset'] in ('tnt','db') else '0.5']
        run_process([python]+tokens, source, env, output/'render')
        pairs = output/'test/ours_30000'
        rendered = {p.name for p in (pairs/'renders').glob('*.png')}
        truth = {p.name for p in (pairs/'gt').glob('*.png')}
        if not rendered or rendered != truth:
            raise RuntimeError('Empty or unmatched test renders: '+str(output))
        original_pairs = original_model/'test/ours_30000/renders'
        if original_pairs.exists() and rendered != {p.name for p in original_pairs.glob('*.png')}:
            raise RuntimeError('Test-view set differs from the archived evaluation')
        if r.get('test_image_names') and rendered != set(r['test_image_names']):
            raise RuntimeError('Test-view set differs from the retained paper view manifest')
        run_process([python,'metrics.py','-m',str(output)], source, env, output/'metrics')
    values = metric_values(output)
    expected = metric_values(original_model)
    delta = {k:values[k]-expected[k] for k in values}
    verdict = all(abs(delta[k]) <= (1e-4 if k=='psnr' else 1e-5) for k in delta)
    fps = benchmark_fps(r,args,output,data,cfg,source,python,env) if args.fps else None
    result = dict(id=r['id'], method=r['method'], repetition=r['repetition'], scene=r['scene'], dataset=r['dataset'],
                  metrics=values, recorded=expected, difference=delta, matches_recorded_within_tolerance=verdict,
                  action=args.action, fps_new_measurement=fps)
    json_write(output/'verification.json', result)
    return result


def aggregate_records(records, output):
    # Scene means first, then equal-weight execution means; training time is summed per execution.
    grouped = {}
    for r in records:
        for ds in (r['dataset'],'all13'):
            key = (r['group'], r['method'], ds, r['repetition'])
            grouped.setdefault(key, []).append(r)
    execution_rows = []
    for (group, method, ds, repetition), items in sorted(grouped.items()):
        row = dict(group=group, method=method, dataset=ds, repetition=repetition, scenes=len(items))
        for k in ('psnr','ssim','lpips'):
            row[k] = statistics.mean(x[k] for x in items)
        times = [x['wall_time_s'] for x in items]
        row['wall_time_sum_s'] = sum(times) if all(t is not None for t in times) else None
        execution_rows.append(row)
    summary = []
    keys = sorted({(r['group'],r['method'],r['dataset']) for r in execution_rows})
    for group,method,ds in keys:
        items = [r for r in execution_rows if (r['group'],r['method'],r['dataset'])==(group,method,ds)]
        row = dict(group=group,method=method,dataset=ds,executions=len(items))
        for k in ('psnr','ssim','lpips','wall_time_sum_s'):
            values = [r[k] for r in items]
            row[k+'_mean'] = statistics.mean(values) if all(v is not None for v in values) else None
            row[k+'_std'] = statistics.stdev(values) if len(values)>1 and all(v is not None for v in values) else None
        summary.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for name, records_ in [('per_scene.csv',records),('per_execution.csv',execution_rows),('summary.csv',summary)]:
        with (output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records_[0]))
            writer.writeheader()
            writer.writerows(records_)
    json_write(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = ROOT.parent/'local_private/machine.json'
    machine = json_read(local) if local.exists() else {}
    parser.add_argument('--action', choices=['evaluate','recorded','train','list'],default='evaluate')
    parser.add_argument('--suite', choices=['selected','table1','main','ablation','transfer','sensitivity','mechanism','all'],default='selected')
    parser.add_argument('--use-local-archive',action='store_true',help='Explicitly read omitted weights from original directories on the research machine')
    parser.add_argument('--fps',action='store_true',help='Also remeasure synchronized render-only FPS (not training peak memory)')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'artifacts')
    parser.add_argument('--data-root',type=Path,default=os.environ.get('FACTORGRAD_DATA_ROOT',machine.get('data_root',str(ROOT/'data/datasets'))))
    parser.add_argument('--conda-root',type=Path,default=os.environ.get('FACTORGRAD_CONDA_ROOT',machine.get('conda_root',str(Path.home()/'miniconda3'))))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--run-id',action='append')
    parser.add_argument('--scenes',help='Comma-separated optional subset for verification')
    args=parser.parse_args()
    for name in ('artifacts','data_root','conda_root','output'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    runs=[r for r in json_read(ROOT/'configs/runs.json') if selected(r,args)]
    if not runs:
        parser.error('No runs selected')
    if args.action=='list':
        for r in runs:
            print(r['id'])
        return
    if args.action=='evaluate':
        for r in runs:
            for name in r['weights']:
                locate_weight(r,name,args)
    records=[]
    discrepancies=[]
    for i,r in enumerate(runs,1):
        print('[{}/{}] {}'.format(i,len(runs),r['id']),flush=True)
        if args.action=='recorded':
            values=metric_values(args.artifacts/r['artifact']/r['model_subdir'])
        else:
            result=evaluate(r,args)
            values=result['metrics']
            if not result['matches_recorded_within_tolerance']:
                discrepancies.append(result)
        historical=r['expected'].get('wall_time_s')
        records.append(dict(group=r['id'].split('/')[0],id=r['id'],method=r['method'],repetition=r['repetition'],
                            dataset=r['dataset'],scene=r['scene'],psnr=values['psnr'],ssim=values['ssim'],lpips=values['lpips'],
                            wall_time_s=float(historical) if historical not in ('',None) else None))
    aggregate_records(records,args.output/'tables')
    json_write(args.output/'evaluation_report.json',dict(action=args.action,runs=len(runs),discrepancies=discrepancies,
               time_note='wall_time_s in tables is HISTORICAL training process wall time, not a new measurement. Missing timing stays null.'))
    print('Results: '+str(args.output))
    if discrepancies and args.action=='evaluate':
        raise SystemExit('Metric differences detected. See evaluation_report.json; no values were overwritten.')


if __name__=='__main__':
    main()
