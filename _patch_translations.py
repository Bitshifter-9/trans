import json

with open('output/step3_meta.json') as f:
    d = json.load(f)

# seg3: 2s window, XTTS generated 4.17s for "यह सब क्यों?" — use single word
d['segments'][3]['hindi'] = 'क्यों?'

with open('output/step3_meta.json', 'w') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print('Updated seg3:', d['segments'][3]['hindi'])
