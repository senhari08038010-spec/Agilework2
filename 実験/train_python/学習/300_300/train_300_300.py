#!/usr/bin/env python3
# 100_100のノートブックを複製し、3つのLinearの次元だけ300へ変更して実行する。
# このフォルダの重み等を出力する。比較中は再学習せず保存済み重みを使う。
from pathlib import Path
import contextlib, hashlib, io, json, os, platform, sys, time
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch, cv2
root=Path(__file__).resolve().parent
os.chdir(root)
notebook=root/'train_L1_normalization_300_300.ipynb'
nb=json.loads(notebook.read_text())
class Tee(io.TextIOBase):
 def __init__(self,buffer):self.buffer=buffer
 def write(self,s):self.buffer.write(s);sys.__stdout__.write(s);sys.__stdout__.flush();return len(s)
 def flush(self):sys.__stdout__.flush()
scope={'__name__':'__main__'};logs=[];start=time.monotonic();count=0
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 code='\n'.join(line for line in ''.join(c.get('source',[])).splitlines() if not line.lstrip().startswith(('%','!')))
 if not code.strip():continue
 count+=1;buffer=io.StringIO()
 with contextlib.redirect_stdout(Tee(buffer)):
  exec(compile(code,str(notebook)+f':cell{i}','exec'),scope)
 output=buffer.getvalue();logs.append(output)
 c['execution_count']=count
 c['outputs']=[{'output_type':'stream','name':'stdout','text':output.splitlines(keepends=True)}] if output else []
 plt.close('all')
(root/'training_log.txt').write_text(''.join(logs))
notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
model=scope['model'];X=scope['X'];Y=scope['Y']
prior=json.loads((root.parent/'100_100/training_data.json').read_text())
assert np.array_equal(X,np.array(prior['X'],dtype=np.float32))
assert np.array_equal(Y,np.array(prior['Y'],dtype=np.float32))
assert X.shape==Y.shape==(33,3)
with torch.no_grad():mse=torch.mean((model(torch.tensor(X))-torch.tensor(Y))**2).item()
assert np.isfinite(mse)
torch.save(model.state_dict(),root/'model_state.pt')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=root.parents[2]
files=['scanner_ver2/before_1669_59.ppm','train_python/week3_6_6.ppm','train_python/img/reference_image1_cmyk_large.png','train_python/img/reference_image2_cmyk_large.png']
report={'architecture':[3,300,300,3],'training_pixels':33,'epochs':50000,'optimizer':'Adam','learning_rate':.01,'lambda_l1':1e-6,'seed':0,'cpu_threads':1,'deterministic_algorithms':True,'training_mse_normalized_rgb':mse,'elapsed_seconds':time.monotonic()-start,'source_notebook_100':str(root.parent/'100_100/train_L1_normalization_100_100.ipynb'),'training_data_equal_to_100_100':True,'training_data_sha256':{f:sha(base/f) for f in files},'model_header_sha256':sha(root/'model_parameters.h'),'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cv2':cv2.__version__,'matplotlib':matplotlib.__version__},'physical_measurement_performed':False}
(root/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(root/'training_data.json').write_text(json.dumps({'X':X.tolist(),'Y':Y.tolist()},ensure_ascii=False,indent=2)+'\n')
print('TRAINING_COMPLETE '+json.dumps(report,ensure_ascii=False),flush=True)
