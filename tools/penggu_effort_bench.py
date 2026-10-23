"""Compare local thinking levels on synthetic Penggu conversations.

Reads the bot's prompt source, never its private chat history. Calls only Strata;
does not send bot messages or change settings. Saves final answers and timing,
not reasoning text or credentials. Run from the Strata repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import strata_tokenizer as ST


def fixtures():
    # Rubrics are fixed before generation. All people and conversations are invented.
    return [
        dict(id='judge_other_people', kind='judge', rubric='只输出 FALSE；两人在互相答话，没有向砰咕提问。',
             turns=[('user', '[10月23日 14:00] 阿禾：@小林 你周末几点到？'),
                    ('user', '[10月23日 14:01] 小林：[回复 阿禾: 你周末几点到] 下午三点，车票买好了。'),
                    ('user', '[10月23日 14:01] 阿禾：那我三点去接你。')]),
        dict(id='judge_followup', kind='judge', rubric='TRUE 并接住阿禾的追问；说先少放一点糖、尝后再加，不能当成别人在问。',
             turns=[('user', '[10月23日 14:00] 阿禾：砰咕，番茄炒蛋我总觉得太酸。'),
                    ('assistant', '可以出锅前放一点点糖，压酸味就好。'),
                    ('user', '[10月23日 14:01] 阿禾：两颗番茄的话放半勺会不会太多？我不想吃成甜口。')]),
        dict(id='judge_stale_topic', kind='judge', rubric='只输出 FALSE；不回头回答上午旧问题、不点评当前表情图。',
             turns=[('user', '[10月23日 09:00] 小林：2100年是不是闰年？'),
                    ('user', '[10月23日 09:02] 阿禾：查到了，不是，整百年份还得能被400整除。'),
                    ('user', '[10月23日 14:00] 小林：[表情图: 猫猫打滚，表示开心]'),
                    ('user', '[10月23日 14:01] 阿禾：[表情图: 小狗鼓掌，表示附和]')]),
        dict(id='reply_math', kind='reply', rubric='A=61.2 元，B=63.2 元，A 便宜2元；运算顺序正确。',
             turns=[('user', '[10月23日 14:00] 阿禾：@砰咕 一件89元的东西，A店先打八折再减10元，B店先减10元再打八折。分别实付多少，哪家便宜多少？没有别的费用。')]),
        dict(id='reply_schedule', kind='reply', rubric='不存在连续30分钟；公共空闲为18:45–18:50和19:10–19:30，最长20分钟。不得把19:10–19:40当作可行。',
             turns=[('user', '[10月23日 14:00] 阿禾：我18:00到20:00都有空。'),
                    ('user', '[10月23日 14:01] 小林：我18:30到19:30有空，19:30必须走。'),
                    ('user', '[10月23日 14:02] 小雨：我18:45到20:00有空。'),
                    ('user', '[10月23日 14:03] 阿禾：@砰咕 帮找最早能一起开会的连续30分钟。会议室18:50到19:10被占用，其余时间空着。如果没有30分钟，告诉我最长能开多久、几点到几点。')]),
        dict(id='reply_empathy', kind='reply', rubric='接住被裁员后的难过；不嘲笑、不把玩笑建议当成命令、不硬给长篇解决方案。',
             turns=[('user', '[10月23日 14:00] 小林：今天突然被裁了，回家坐了半小时，饭也没吃。'),
                    ('user', '[10月23日 14:01] 小林：@砰咕 我知道要找工作，但现在不想听建议，就想有人陪我说两句。')]),
        dict(id='reply_uncertain', kind='reply', rubric='明确不知道哪个人拿快递，不根据在场信息断言；可建议群里问或核对签收信息。',
             turns=[('user', '[10月23日 14:00] 阿禾：我的快递显示签收了，门口没找到。'),
                    ('user', '[10月23日 14:01] 小林：我下午路过门口，但没看到快递。'),
                    ('user', '[10月23日 14:02] 小雨：我今天一直在外面。'),
                    ('user', '[10月23日 14:03] 阿禾：@砰咕 所以肯定是小林拿了对吧？你直接说谁拿的。')]),
        dict(id='recap_corrections', kind='recap', rubric='保留10月25日周日15点东门集合、阿禾小林线下、小雨仅线上、是否下雨待确认；旧日期地点须更新；小林尚未离职。最多300字，无标题。',
             previous='10月23日上午，阿禾、小林、小雨约好10月24日周六15点在公园南门见面。小林似乎已经辞职。',
             turns=[('user', '[10月23日 13:00] 阿禾：周六有事，改周日吧，10月25日，还是下午三点。'),
                    ('user', '[10月23日 13:01] 小林：同意，南门封了，走东门。'),
                    ('user', '[10月23日 13:02] 小雨：我来不了现场，可以线上陪聊。'),
                    ('user', '[10月23日 13:03] 阿禾：那线下就我和小林，周日15点东门。'),
                    ('user', '[10月23日 13:04] 小林：还有我没辞职，只是在等另一家的offer，别传歪了。'),
                    ('user', '[10月23日 13:05] 小雨：天气预报还没查，如果下雨要不要改室内？'),
                    ('user', '[10月23日 13:06] 阿禾：这个明天确认，现在先不定。')]),
        dict(id='recap_long', kind='recap', rubric='保留10月25日19:30线上联机、阿禾主持、小雨带新手、当前缺1人、小林不参加、周六压力测试取消；不保留错误19点/缺2人/小林参加。最多300字。',
             turns=[('user', '[10月23日 12:00] 阿禾：周日10月25日晚上七点开联机？暂时缺两个人。'),
                    ('user', '[10月23日 12:01] 小林：我应该能来。'),
                    ('user', '[10月23日 12:02] 小雨：我可以带新手，阿禾当主持。')]
             + [('user', f'[10月23日 12:{i+3:02d}] ' + ('小林' if i % 2 else '阿禾') + '：' +
                 ['昨天那个关卡的隐藏门我没找到，绕了好几圈。', '原来入口在瀑布后面，我还以为地图没画完。',
                  '晚饭吃面还是炒饭，我又开始纠结了。', '[表情图: 猫猫打滚，表示开心]',
                  '刚才键盘没电，换了电池就好了。', '这个贴纸好像之前见过，不记得在哪了。'][i % 6]) for i in range(36)]
             + [('user', '[10月23日 13:00] 阿禾：确认下，周日改19:30，19点我赶不及，周六那场压力测试取消。'),
                ('user', '[10月23日 13:01] 小林：我周日临时加班，不参加了。'),
                ('user', '[10月23日 13:02] 小雨：我又喊来了两个人，算上小林退出，现在只缺一个。'),
                ('user', '[10月23日 13:03] 阿禾：好，我主持，小雨带新手，还缺一个继续找。')]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--efforts', nargs='+', default=['low', 'medium', 'xhigh'])
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--budget', type=int)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    cases = [c for c in fixtures() if not args.cases or c['id'] in args.cases]
    if not args.run:
        print(json.dumps({'cases': [c['id'] for c in cases], 'efforts': args.efforts, 'requests': len(cases)*len(args.efforts)}))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    source = args.prompts.read_text(encoding='utf8')
    prompts = {}
    for key in ('profilePrompt', 'judgeReplyPrompt', 'recapPrompt'):
        prompts[key] = re.search(r'\b' + key + r': `([\s\S]*?)`', source).group(1)
        assert '${' not in prompts[key], 'Unexpected template interpolation'
    tpath = Path(cfg['tokenizer'])
    vocab = json.loads((tpath / 'vocab.json').read_text(encoding='utf8'))
    tokens = [None] * len(vocab)
    for token, i in vocab.items(): tokens[i] = token
    tok = ST.Tokenizer(tokens, (tpath / 'merges.txt').read_text(encoding='utf8').split('\n'),
                       json.loads((tpath / 'token_type.json').read_text(encoding='utf8')))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, body=None):
        req = urllib.request.Request('http://127.0.0.1:8080/' + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + cfg['api_key'], 'Content-Type': 'application/json'})
        with opener.open(req, timeout=240) as response: return json.load(response)

    health = api('health')
    manifest = {'model': health['model'], 'temperature': 0.8, 'max_tokens': 4096,
                'seed': args.seed, 'reasoning_budget_tokens': args.budget,
                'strata_checkpoint': False, 'synthetic': True,
                'prompt_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k,v in prompts.items()},
                'template_sha256': hashlib.sha256((tpath / 'chat_template.jinja').read_bytes()).hexdigest(),
                'cases': cases, 'efforts': args.efforts,
                'engine': {k:v for k,v in api('metrics')['engine'].items() if k in
                           ('engine','kv','ctx','conversation_cache_slots','conversation_cache_mib','tail_role_token')}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    for index, case in enumerate(cases):
        if case['kind'] == 'recap':
            system = prompts['recapPrompt']
            previous = ('前一段的前情摘要:\n' + case['previous'] + '\n\n') if case.get('previous') else ''
            history = [{'role': 'user', 'content': previous + '需要总结的群聊片段:\n' +
                       '\n'.join(text for _,text in case['turns']) + '\n\n请输出前情摘要。'}]
        else:
            system = prompts['profilePrompt']
            if case['kind'] == 'judge': system += '\n\n' + prompts['judgeReplyPrompt']
            history = [{'role': role, 'content': text} for role,text in case['turns']]
            if case['kind'] == 'judge':
                history += [{'role': 'user', 'content': '[系统通知] 现在是10月23日14:02。新消息是14:00起的消息。请判断是否接话。'}]
        efforts = args.efforts[index % len(args.efforts):] + args.efforts[:index % len(args.efforts)]
        for effort in efforts:
            deadline = time.monotonic() + 900
            while True:
                state = api('status')
                if not state.get('busy') and not state.get('queued'): break
                if time.monotonic() > deadline: raise RuntimeError('Server remained busy')
                time.sleep(3)
            body = {'model': health['model'], 'messages': [{'role':'system','content':system}, *history],
                    'temperature': 0.8, 'seed': args.seed, 'max_tokens': 4096,
                    'reasoning_effort': effort, 'strata_checkpoint': False}
            if args.budget is not None: body['reasoning_budget_tokens'] = args.budget
            start = time.monotonic()
            response = api('v1/chat/completions', body)
            elapsed = time.monotonic() - start
            choice = response['choices'][0]
            answer = choice['message'].get('content') or ''
            reasoning = choice['message'].get('reasoning_content') or ''
            record = {'case': case['id'], 'kind': case['kind'], 'effort': effort,
                      'seed': args.seed, 'wall_seconds': round(elapsed,3), 'answer': answer,
                      'answer_chars': len(answer), 'answer_retokenized': len(tok.encode(answer, parse_special=True)),
                      'reasoning_retokenized': len(tok.encode(reasoning, parse_special=True)),
                      'finish': choice['finish_reason'], 'usage': response.get('usage'),
                      'timings': response.get('timings'), 'budget_requested': args.budget}
            (args.out / (case['id'] + '-' + effort + '.json')).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
            print(json.dumps({k:record[k] for k in ('case','effort','wall_seconds','answer_chars','reasoning_retokenized','finish')}), flush=True)
            time.sleep(1)  # Allow an already waiting bot request to take the next slot.
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
