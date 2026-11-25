import hashlib, json, os, zipfile, io, csv
from pathlib import Path
import numpy as np
r=Path.cwd()
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
protocol=load(r/'frozen_protocol.json')
manifest=load(r/'inputs/input_manifest.json')
subjects=next(v for v in manifest.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'output_sha256' in v[0])
summary=load(r/'scores/summary.json')
assert len(subjects)==6 and len(summary['per_animal_method'])==12
report_rows = {(x['animal'],x['method']):x for x in summary['per_animal_method']}
check={}
for item in subjects:
    a=item['id']; c=protocol['animals'][a]
    inp=r/'inputs'/item['npz']
    assert sha(inp)==item['output_sha256']
    with np.load(inp,allow_pickle=False) as d:
        assert np.array_equal(d['roi_ids'],np.arange(c['n_rois']))
        rng=np.random.default_rng(c['seed'])
        permutations=np.stack([rng.permutation(c['n_rois']) for _ in range(c['frames'])])
        times=d['times']
        assert np.all(np.any(permutations[1:]!=permutations[:-1],axis=1))
    out=r/'results'/a
    arrays=[]
    for m in ('dual_anchor','cpd_reference'):
        zipout=out/(m+'.npz')
        details=load(out/(m+'.json'))
        assert details['input_sha256']==sha(inp)
        assert details['protocol_sha256']==sha(r/'frozen_protocol.json')
        score=report_rows[a,m]
        assert details['assignment_file_sha256']==score['npz_sha256']==sha(zipout)
        assert details['method']==m and details['permutation_seed']==c['seed'] and details['independent_integer_agreement']
        assert all(details[k]==score[k] for k in ('correct','total','identity_switches','wrong_identity_episode_count','wrong_identity_frames','maximum_wrong_identity_duration_selected_scans'))
        with np.load(zipout,allow_pickle=False) as d:
            assert np.array_equal(d['truth_indices'],permutations)
            assert np.array_equal(d['frame_indices'],np.arange(c['frames']))
            assert np.array_equal(d['frame_times'],times)
            arrays.append(d['assignments'])
    check[a]={'input_sha256':sha(inp),'truth_permutations_reconstructed':True,'methods_identical_assignments':bool(np.array_equal(*arrays)),
              'transitions_with_changed_anonymous_row_order':len(permutations)-1,
              'tracker_seconds':{m:load(out/(m+'.json'))['tracker_seconds'] for m in ('dual_anchor','cpd_reference')}}
target=r/'scores/final_checks.json'
target.write_text(json.dumps({'input_manifest_sha256':sha(r/'inputs/input_manifest.json'),'protocol_sha256':sha(r/'frozen_protocol.json'),'by_animal':check},indent=2)+'\n')
files=[r/'run_frozen.py',r/'frozen_protocol.json',r/'independent_score.py',r/'input_motion_diagnostics.json',r/'inputs/extract_sk.py',r/'inputs/input_manifest.json',
       *(r/'inputs').glob('SK*.npz'),*(r/'scores').glob('*.json'),r/'scores/per_animal.csv',*(r/'results').glob('*/*')]
# Include the archived tracker implementations and frozen configuration for this run.
hist=Path.home()/'wha-paper-phase2-20261004-iPztef/project'
local_members={p.relative_to(r).as_posix():p for p in files if p.is_file()}
for rel,h in protocol['historical_source_hashes'].items():
    if rel.startswith('src/') or rel in ('configs/wormid_prospective.yaml','configs/paper_phase2.yaml'):
        src=hist/rel
        assert sha(src)==h
        local_members['frozen_project/'+rel]=src
for rel in ['requirements-paper-pilot-lock.txt','work/results/paper_phase2_20261004/cpd_selection.json']:
    local_members['frozen_project/'+rel]=hist/rel
archive=r/'sk-comparison-evidence.zip'
hashlist={name:sha(p) for name,p in sorted(local_members.items())}
with zipfile.ZipFile(archive.with_suffix('.tmp'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for name,p in sorted(local_members.items()):z.write(p,name)
    z.writestr('checksums.json',json.dumps(hashlist,indent=2)+'\n')
os.replace(archive.with_suffix('.tmp'),archive)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest()==s for n,s in hashlist.items())
source=summary['method_means']
print('Cases',len(report_rows),'identical whole-recording assignment arrays',sum(row['methods_identical_assignments'] for row in check.values()))
print('Total identities per method',sum(x['total'] for x in summary['per_animal_method'] if x['method']=='dual_anchor'))
print('Archive',archive.stat().st_size,sha(archive),len(hashlist))
