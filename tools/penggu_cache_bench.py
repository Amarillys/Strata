"""Synthetic Penggu-shaped cache benchmark; stop other clients before --run.

Only numeric metrics and output hashes are saved. Credentials come from the
local preset, never command-line arguments. No messages are sent to the bot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if not args.run:
        print('Dry run: add --run with other clients stopped.')
        return
    if args.output.exists():
        raise ValueError('Refusing to overwrite results')
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(endpoint, body=None):
        request = urllib.request.Request('http://127.0.0.1:8080/' + endpoint,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + cfg['api_key'], 'Content-Type': 'application/json'})
        with opener.open(request, timeout=180) as response:
            return json.load(response)

    def metrics():
        result = api('metrics?requests=all')
        if api('status').get('busy') or result['live'].get('queued'):
            raise RuntimeError('Other client active; benchmark requires idle server')
        return result

    before = metrics()
    model = api('health')['model']
    records = []
    sequence = [('A', 1), ('A', 2), ('B', 1), ('C', 1), ('D', 1), ('E', 1),
                ('A', 3), ('B', 2), ('C', 2), ('D', 2), ('E', 2)]
    for group, turn in sequence:
        system = f'Cache fixture group {group}. Reply with exactly OK. Ignore the records below.\n'
        system += '\n'.join(f'Rule {i}: blue square, green triangle, red circle; keep stable group records.' for i in range(280))
        history = [{'role': 'user', 'content': f'Group context record {i}: ' + 'The weather was clear and the meeting finished on time. ' * 4}
                   for i in range(40)]
        if turn > 1:
            history += [{'role': 'assistant', 'content': '[11:00:00] OK'},
                        {'role': 'user', 'content': 'Member: please continue the discussion.'}]
        if turn > 2:
            history += [{'role': 'assistant', 'content': '[11:01:00] OK'},
                        {'role': 'user', 'content': 'Member: another new message.'}]
        messages = [{'role': 'system', 'content': system}, *history,
                    {'role': 'user', 'content': f'Current time 11:0{turn}:00. A member mentioned you; reply exactly OK.'}]
        first = metrics()
        start = time.monotonic()
        response = api('v1/chat/completions', {'model': model, 'messages': messages,
            'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
        end = metrics()
        if end['totals']['requests'] != first['totals']['requests'] + 1:
            raise RuntimeError('Concurrent request detected')
        row = max(end['requests'], key=lambda r: r['time'])
        content = response['choices'][0]['message'].get('content', '')
        if not content.strip() or row['finish'] not in ('stop', 'length'):
            raise RuntimeError('Invalid completion')
        record = {'group': group, 'turn': turn, 'warm': turn > 1,
                  'wall_ms': round((time.monotonic() - start) * 1000, 1),
                  'output_sha256': hashlib.sha256(content.encode()).hexdigest(),
                  **{k: row[k] for k in ('prompt_tokens', 'reused', 'prompt_ms', 'decode_ms', 'output_tokens', 'finish')}}
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps({'complete': False, 'records': records}, indent=2), encoding='utf-8')
    end = metrics()
    summary = {}
    for label, selected in [('all', records), ('warm', [r for r in records if r['warm']])]:
        prompt = sum(r['prompt_tokens'] for r in selected)
        reused = sum(r['reused'] for r in selected)
        summary[label] = {'requests': len(selected), 'prompt_tokens': prompt, 'reused': reused,
                          'reuse_percent': round(100 * reused / prompt, 2),
                          'prompt_ms': round(sum(r['prompt_ms'] for r in selected), 1),
                          'wall_ms': round(sum(r['wall_ms'] for r in selected), 1)}
    result = {'complete': True, 'summary': summary, 'records': records,
              'cache_before': before['conversation_cache'], 'cache_after': end['conversation_cache'],
              'engine': {k: v for k, v in end['engine'].items() if k in
                  ('engine', 'prompt_cache', 'prompt_cache_every', 'conversation_cache_slots',
                   'conversation_cache_mib', 'tail_role_token', 'expert_slots', 'max_context')}}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
