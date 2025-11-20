import os
import json
from glob import glob
from collections import defaultdict

PHEME_ROOT = 'pheme_dataset'
EVENTS = [
    'charliehebdo-all-rnr-threads',
    'ebola-essien-all-rnr-threads',
    'ferguson-all-rnr-threads',
    'germanwings-crash-all-rnr-threads',
    'gurlitt-all-rnr-threads',
    'ottawashooting-all-rnr-threads',
    'prince-toronto-all-rnr-threads',
    'putinmissing-all-rnr-threads',
    'sydneysiege-all-rnr-threads',
]

# Output: list of dicts, one per rumor
output = []


count = 0
for event in EVENTS:
    for label in ['rumours', 'non-rumours']:
        base = os.path.join(PHEME_ROOT, event, label)
        if not os.path.exists(base):
            continue
        thread_ids = [tid for tid in os.listdir(base) if not tid.startswith('.')]
        for thread_id in thread_ids[:50]:
            thread_path = os.path.join(base, thread_id)
            if not os.path.isdir(thread_path):
                continue
            # annotation
            ann_path = os.path.join(thread_path, 'annotation.json')
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, encoding='utf8') as f:
                annotation = json.load(f)
            # structure
            struct_path = os.path.join(thread_path, 'structure.json')
            if not os.path.exists(struct_path):
                continue
            with open(struct_path, encoding='utf8') as f:
                structure = json.load(f)
            # source tweet
            src_dir = os.path.join(thread_path, 'source-tweets')
            src_files = glob(os.path.join(src_dir, '*.json'))
            if not src_files:
                continue
            with open(src_files[0], encoding='utf8') as f:
                source_tweet = json.load(f)
            # reactions
            react_dir = os.path.join(thread_path, 'reactions')
            react_files = [rf for rf in glob(os.path.join(react_dir, '*.json')) if not os.path.basename(rf).startswith('.')]
            reactions = []
            for rf in react_files:
                with open(rf, encoding='utf8') as f:
                    reactions.append(json.load(f))
            # collect
            output.append({
                'event': event,
                'label': label,
                'thread_id': thread_id,
                'annotation': annotation,
                'structure': structure,
                'source_tweet': source_tweet,
                'reactions': reactions
            })
            count += 1
            if count % 10 == 0:
                print(f"Parsed {count} threads so far...")

with open('pheme_parsed_threads.json', 'w', encoding='utf8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"Parsed {len(output)} threads. Output: pheme_parsed_threads.json")
